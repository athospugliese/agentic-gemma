"""LLM adapter layer - translates between internal format and OpenAI API."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI, NOT_GIVEN

from app.models import (
    Message,
    MessageRole,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
    create_assistant_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class LLMErrorCategory:
    RETRYABLE = "retryable"           # 429, 500, 502, 503, timeout, connection
    CONTEXT_TOO_LONG = "context_too_long"  # 400 context_length_exceeded
    OUTPUT_TRUNCATED = "output_truncated"  # finish_reason=length
    TERMINAL = "terminal"             # 400 (other), 401, 403, etc.


@dataclass
class LLMError(Exception):
    """Classified LLM error with metadata for recovery decisions."""
    category: str
    message: str
    status_code: int | None = None
    retry_after: float | None = None  # From Retry-After header
    original: Exception | None = None

    def __str__(self) -> str:
        return f"LLMError({self.category}, status={self.status_code}): {self.message}"


def classify_llm_error(error: Exception) -> LLMError:
    """Classify an OpenAI exception into a recovery category."""
    from openai import (
        RateLimitError,
        APIStatusError,
        APITimeoutError,
        APIConnectionError,
    )

    if isinstance(error, RateLimitError):
        retry_after = None
        if hasattr(error, 'response') and error.response is not None:
            ra = error.response.headers.get("retry-after")
            if ra:
                try:
                    retry_after = float(ra)
                except ValueError:
                    pass
        return LLMError(
            category=LLMErrorCategory.RETRYABLE,
            message=str(error),
            status_code=429,
            retry_after=retry_after,
            original=error,
        )

    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return LLMError(
            category=LLMErrorCategory.RETRYABLE,
            message=str(error),
            status_code=None,
            original=error,
        )

    if isinstance(error, APIStatusError):
        status = error.status_code

        # Context length exceeded → recoverable via compact
        if status == 400:
            msg = str(error).lower()
            if "context_length" in msg or "maximum context" in msg or "too long" in msg:
                return LLMError(
                    category=LLMErrorCategory.CONTEXT_TOO_LONG,
                    message=str(error),
                    status_code=400,
                    original=error,
                )
            # Other 400s are terminal
            return LLMError(
                category=LLMErrorCategory.TERMINAL,
                message=str(error),
                status_code=400,
                original=error,
            )

        # Server errors are retryable
        if status in (500, 502, 503, 529):
            retry_after = None
            if hasattr(error, 'response') and error.response is not None:
                ra = error.response.headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass
            return LLMError(
                category=LLMErrorCategory.RETRYABLE,
                message=str(error),
                status_code=status,
                retry_after=retry_after,
                original=error,
            )

        # Auth/forbidden are terminal
        return LLMError(
            category=LLMErrorCategory.TERMINAL,
            message=str(error),
            status_code=status,
            original=error,
        )

    # Unknown exceptions are terminal
    return LLMError(
        category=LLMErrorCategory.TERMINAL,
        message=str(error),
        original=error,
    )


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

BASE_DELAY_MS = 500
MAX_DELAY_MS = 32_000
MAX_RETRIES = 10


# ---------------------------------------------------------------------------
# Stream events (internal, provider-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A single event from the LLM stream."""

    type: str  # "text_delta", "tool_call_start", "tool_call_delta", "done", "error"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def create_completion(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition],
        model: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream completion events from the LLM."""
        ...  # pragma: no cover

    @abstractmethod
    def build_assistant_message(self, events: list[StreamEvent]) -> Message:
        """Accumulate stream events into a complete assistant Message."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter(LLMAdapter):
    """Translates internal format <-> OpenAI Chat Completions API."""

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    # -- Public API --------------------------------------------------------

    async def create_completion(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition],
        model: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        openai_messages = self._convert_messages(messages, system_prompt)
        openai_tools = self._convert_tools(tools) if tools else NOT_GIVEN

        # Retry loop with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                async for event in self._stream_completion(
                    openai_messages, openai_tools, model, max_tokens, **kwargs,
                ):
                    yield event
                return  # Success — exit retry loop
            except Exception as e:
                classified = classify_llm_error(e)

                if classified.category != LLMErrorCategory.RETRYABLE:
                    # Non-retryable: propagate as classified error
                    raise classified from e

                if attempt >= MAX_RETRIES:
                    raise classified from e

                # Calculate backoff with jitter
                if classified.retry_after and classified.retry_after < 60:
                    delay = classified.retry_after
                else:
                    base_delay = min(BASE_DELAY_MS * (2 ** attempt), MAX_DELAY_MS)
                    jitter = random.random() * 0.25 * base_delay
                    delay = (base_delay + jitter) / 1000.0  # Convert to seconds

                logger.warning(
                    "LLM API error (attempt %d/%d, status=%s): %s. Retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, classified.status_code, classified.message[:200], delay,
                )
                await asyncio.sleep(delay)
                last_error = e

        # Should not reach here, but just in case
        if last_error:
            raise classify_llm_error(last_error) from last_error

    async def _stream_completion(
        self,
        openai_messages: list[dict[str, Any]],
        openai_tools: Any,
        model: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Single streaming attempt (no retry)."""
        stream = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=openai_tools,
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        # Accumulate tool call deltas by index
        tool_call_accum: dict[int, dict[str, str]] = {}
        text_accum = ""
        usage_data: dict[str, Any] | None = None
        final_finish_reason: str | None = None

        async for chunk in stream:
            # Usage comes in the FINAL chunk (after finish_reason, with empty choices)
            if chunk.usage:
                usage_data = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                    "total_tokens": chunk.usage.total_tokens or 0,
                }
                if hasattr(chunk.usage, "prompt_tokens_details") and chunk.usage.prompt_tokens_details:
                    usage_data["cached_tokens"] = getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0) or 0
                if hasattr(chunk.usage, "completion_tokens_details") and chunk.usage.completion_tokens_details:
                    usage_data["reasoning_tokens"] = (
                        getattr(chunk.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                    )

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Text content
            if delta.content:
                text_accum += delta.content
                yield StreamEvent(type="text_delta", data={"content": delta.content})

            # Tool calls (incremental)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {"id": "", "name": "", "arguments": ""}
                    acc = tool_call_accum[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] = tc_delta.function.name
                            yield StreamEvent(
                                type="tool_call_start",
                                data={"index": idx, "id": acc["id"], "name": acc["name"]},
                            )
                        if tc_delta.function.arguments:
                            acc["arguments"] += tc_delta.function.arguments
                            yield StreamEvent(
                                type="tool_call_delta",
                                data={"index": idx, "arguments_delta": tc_delta.function.arguments},
                            )

            # Track finish reason but DON'T yield done yet — usage comes in a later chunk
            if finish_reason:
                final_finish_reason = finish_reason

        # Stream exhausted — now we have both finish_reason and usage
        yield StreamEvent(
            type="done",
            data={
                "finish_reason": final_finish_reason or "stop",
                "text": text_accum,
                "tool_calls": list(tool_call_accum.values()),
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
        usage_data = data.get("usage")

        tool_calls = []
        for tc in raw_tool_calls:
            try:
                parsed_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                parsed_input = {"_raw": tc["arguments"]}
            tool_calls.append(
                ToolUseBlock(id=tc["id"], name=tc["name"], input=parsed_input)
            )

        usage = None
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                cached_tokens=usage_data.get("cached_tokens", 0),
                reasoning_tokens=usage_data.get("reasoning_tokens", 0),
            )

        return create_assistant_message(text=text, tool_calls=tool_calls, usage=usage)

    # -- Internal ----------------------------------------------------------

    def _convert_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        result: list[dict[str, Any]] = []

        # System prompt as first message
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            result.extend(self._convert_single_message(msg))

        return result

    def _convert_single_message(self, msg: Message) -> list[dict[str, Any]]:
        """Convert one internal message to one or more OpenAI messages."""
        if msg.role == MessageRole.SYSTEM:
            return [{"role": "system", "content": msg.get_text()}]

        if msg.role == MessageRole.USER:
            return [{"role": "user", "content": msg.get_text()}]

        if msg.role == MessageRole.ASSISTANT:
            out: dict[str, Any] = {"role": "assistant"}
            text = msg.get_text()
            if msg.tool_calls:
                out["content"] = text if text else None
                out["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            else:
                out["content"] = text
            return [out]

        if msg.role == MessageRole.TOOL:
            return [
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.get_text(),
                }
            ]

        return []

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to OpenAI format."""
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
