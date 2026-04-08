"""Ollama adapter - translates between internal format and Ollama API."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import httpx

from app.models import (
    Message,
    MessageRole,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
    create_assistant_message,
)
from app.services.llm_adapter import (
    LLMAdapter,
    LLMError,
    LLMErrorCategory,
    StreamEvent,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY_S = 0.5


def classify_ollama_error(error: Exception) -> LLMError:
    """Classify an Ollama error into a recovery category."""
    msg = str(error).lower()

    if isinstance(error, httpx.ConnectError):
        return LLMError(
            category=LLMErrorCategory.RETRYABLE,
            message=f"Cannot connect to Ollama: {error}",
            original=error,
        )

    if isinstance(error, httpx.TimeoutException):
        return LLMError(
            category=LLMErrorCategory.RETRYABLE,
            message=f"Ollama timeout: {error}",
            original=error,
        )

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 404:
            return LLMError(
                category=LLMErrorCategory.TERMINAL,
                message=f"Model not found: {error}",
                status_code=404,
                original=error,
            )
        if status >= 500:
            return LLMError(
                category=LLMErrorCategory.RETRYABLE,
                message=str(error),
                status_code=status,
                original=error,
            )
        return LLMError(
            category=LLMErrorCategory.TERMINAL,
            message=str(error),
            status_code=status,
            original=error,
        )

    if "context length" in msg or "too long" in msg:
        return LLMError(
            category=LLMErrorCategory.CONTEXT_TOO_LONG,
            message=str(error),
            original=error,
        )

    return LLMError(
        category=LLMErrorCategory.TERMINAL,
        message=str(error),
        original=error,
    )


class OllamaAdapter(LLMAdapter):
    """Translates internal format <-> Ollama /api/chat API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def create_completion(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition],
        model: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        ollama_messages = self._convert_messages(messages, system_prompt)
        ollama_tools = self._convert_tools(tools) if tools else None

        for attempt in range(MAX_RETRIES + 1):
            try:
                async for event in self._stream_completion(
                    ollama_messages, ollama_tools, model, max_tokens, **kwargs,
                ):
                    yield event
                return
            except Exception as e:
                classified = classify_ollama_error(e)
                if classified.category != LLMErrorCategory.RETRYABLE:
                    raise classified from e
                if attempt >= MAX_RETRIES:
                    raise classified from e
                delay = BASE_DELAY_S * (2 ** attempt)
                logger.warning(
                    "Ollama error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, classified.message[:200], delay,
                )
                await asyncio.sleep(delay)

    async def _stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Single streaming attempt against Ollama /api/chat."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        text_accum = ""
        tool_calls_accum: list[dict[str, Any]] = []
        finish_reason = "stop"

        async with self._http.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Text content
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                if content:
                    text_accum += content
                    yield StreamEvent(type="text_delta", data={"content": content})

                # Tool calls (Ollama sends them in message.tool_calls)
                if msg.get("tool_calls"):
                    for i, tc in enumerate(msg["tool_calls"]):
                        func = tc.get("function", {})
                        tc_id = f"call_{len(tool_calls_accum)}"
                        tool_calls_accum.append({
                            "id": tc_id,
                            "name": func.get("name", ""),
                            "arguments": json.dumps(func.get("arguments", {})),
                        })
                        yield StreamEvent(
                            type="tool_call_start",
                            data={"index": len(tool_calls_accum) - 1, "id": tc_id, "name": func.get("name", "")},
                        )
                        yield StreamEvent(
                            type="tool_call_delta",
                            data={"index": len(tool_calls_accum) - 1, "arguments_delta": json.dumps(func.get("arguments", {}))},
                        )

                # Check if done
                if chunk.get("done"):
                    if chunk.get("done_reason") == "length":
                        finish_reason = "length"
                    break

        # Emit final done event
        usage_data = None
        yield StreamEvent(
            type="done",
            data={
                "finish_reason": finish_reason,
                "text": text_accum,
                "tool_calls": tool_calls_accum,
                "usage": usage_data,
            },
        )

    def build_assistant_message(self, events: list[StreamEvent]) -> Message:
        done_event = next((e for e in events if e.type == "done"), None)
        if not done_event:
            return create_assistant_message(text="")

        data = done_event.data
        text = data.get("text") or None
        raw_tool_calls = data.get("tool_calls", [])

        tool_calls = []
        for tc in raw_tool_calls:
            try:
                parsed_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                parsed_input = {"_raw": tc["arguments"]}
            tool_calls.append(
                ToolUseBlock(id=tc["id"], name=tc["name"], input=parsed_input)
            )

        return create_assistant_message(text=text, tool_calls=tool_calls)

    # -- Side query support (for memory, compaction) --

    async def side_query(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 256,
    ) -> str:
        """Non-streaming auxiliary LLM call."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        resp = await self._http.post("/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    # -- Internal --

    def _convert_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        """Convert internal messages to Ollama format."""
        result: list[dict[str, Any]] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            result.extend(self._convert_single_message(msg))

        return result

    def _convert_single_message(self, msg: Message) -> list[dict[str, Any]]:
        """Convert one internal message to Ollama format."""
        if msg.role == MessageRole.SYSTEM:
            return [{"role": "system", "content": msg.get_text()}]

        if msg.role == MessageRole.USER:
            return [{"role": "user", "content": msg.get_text()}]

        if msg.role == MessageRole.ASSISTANT:
            out: dict[str, Any] = {"role": "assistant"}
            text = msg.get_text()
            out["content"] = text or ""
            if msg.tool_calls:
                out["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.input,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return [out]

        if msg.role == MessageRole.TOOL:
            return [{"role": "tool", "content": msg.get_text()}]

        return []

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to Ollama format (OpenAI-compatible)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
