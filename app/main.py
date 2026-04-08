"""FastAPI application - HTTP API for the agent system."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.definitions import AgentRegistry
from app.services.llm_adapter import LLMAdapter
from app.services.permissions import PermissionManager
from app.services.query import QueryEngine, QueryEvent
from app.services.session_store import SessionStore
from app.settings import Settings, load_settings
from app.tools.base import ToolRegistry
from app.tools.registry import create_full_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

settings: Settings = Settings()
session_store: SessionStore = SessionStore()
cleanup_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, session_store, cleanup_task

    settings = load_settings()
    session_store = SessionStore(base_dir=f"{settings.data_dir}/sessions")

    # Periodic cleanup of idle sessions
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)
            removed = await session_store.cleanup_expired(max_idle_seconds=3600)
            if removed:
                logger.info("Cleaned up %d idle sessions", removed)

    cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("Agent API started (model=%s)", settings.default_model)

    yield

    cleanup_task.cancel()
    logger.info("Agent API shutting down")


app = FastAPI(title="Agent API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    project_dir: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    permission_mode: str = "auto"
    cwd: str | None = None
    coordinator_mode: bool = False


class SessionResponse(BaseModel):
    session_id: str
    resumed: bool = False


class MessageRequest(BaseModel):
    prompt: str
    model: str | None = None


class ToolCallDetail(BaseModel):
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any] = {}
    result: str = ""
    is_error: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class MessageResponse(BaseModel):
    """Complete response for a single message turn (non-streaming)."""

    type: str = "result"  # "result"
    subtype: str = "success"  # "success" | "error" | "error_max_turns"
    text: str = ""  # Final concatenated text
    model: str = ""
    session_id: str = ""
    stop_reason: str | None = None  # "stop" | "tool_calls" | "max_turns" | "aborted"
    num_turns: int = 0
    duration_ms: int = 0
    is_error: bool = False
    errors: list[str] = []

    # Usage
    usage: UsageInfo = UsageInfo()

    # Tool calls made during this message
    tool_calls: list[ToolCallDetail] = []

    # Messages exchanged (full conversation for this turn)
    messages: list[dict[str, Any]] = []


class PermissionResponse(BaseModel):
    allow: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_adapter_and_client() -> tuple[LLMAdapter, Any]:
    """Create the LLM adapter and side-query client based on configured provider."""
    if settings.llm_provider == "ollama":
        from app.services.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(base_url=settings.ollama_base_url)
        # For side queries (memory, compaction), pass the adapter itself
        return adapter, adapter
    else:
        from openai import AsyncOpenAI
        from app.services.llm_adapter import OpenAIAdapter
        client = AsyncOpenAI(
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
        )
        adapter = OpenAIAdapter(client)
        return adapter, client


def _create_engine(session_id: str, req: CreateSessionRequest) -> QueryEngine:
    """Create a QueryEngine from request params."""
    adapter, side_query_client = _create_adapter_and_client()

    # Resolve model based on provider
    if settings.llm_provider == "ollama":
        default_model = settings.ollama_model
        fast_model = settings.ollama_model
    else:
        default_model = settings.default_model
        fast_model = settings.fast_model

    # Agent registry (built-in + custom from project dir)
    agent_reg = AgentRegistry()
    agent_reg.load_defaults()
    if req.project_dir:
        agent_reg.load_custom(req.project_dir)

    # Full tool registry including Agent tool
    registry = create_full_registry(adapter, agent_reg, settings)

    # Coordinator mode: restrict to Agent + SendMessage + TaskStop only
    # This FORCES the model to delegate via Agent tool instead of using tools directly
    if req.coordinator_mode:
        COORDINATOR_ALLOWED = {"Agent", "SendMessage", "TaskStop", "TaskOutput"}
        coordinator_registry = ToolRegistry()
        for tool in registry.all():
            if tool.name in COORDINATOR_ALLOWED:
                coordinator_registry.register(tool)
        registry = coordinator_registry

    permissions = PermissionManager(
        mode=req.permission_mode,
        allow_rules=list(settings.permissions.allow),
        deny_rules=list(settings.permissions.deny),
    )

    engine = QueryEngine(
        adapter=adapter,
        registry=registry,
        permissions=permissions,
        session_id=session_id,
        model=req.model or default_model,
        fast_model=fast_model,
        max_tokens=settings.max_tokens,
        max_turns=settings.max_turns,
        cwd=req.cwd or req.project_dir or ".",
        custom_system_prompt=req.system_prompt,
        openai_client=side_query_client,
        data_dir=settings.data_dir,
        coordinator_mode=req.coordinator_mode,
    )

    # Wire hooks config from settings
    if settings.hooks:
        engine._hooks_config = {k: list(v) for k, v in settings.hooks.items()}

    return engine


def _format_sse(event: QueryEvent) -> str:
    """Format a QueryEvent as an SSE data line."""
    payload = {"type": event.type, **event.data}
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=SessionResponse)
async def create_session(req: CreateSessionRequest = CreateSessionRequest()) -> SessionResponse:
    """Create a new agent session."""
    session_id = uuid4().hex[:16]
    engine = _create_engine(session_id, req)
    session_store.register(session_id, engine)
    return SessionResponse(session_id=session_id)


@app.get("/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    """List active sessions."""
    return session_store.list_active()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Remove an active session."""
    engine = session_store.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")
    engine.abort()
    session_store.remove(session_id)
    return {"status": "deleted"}


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, req: MessageRequest) -> StreamingResponse:
    """Send a message and stream the response as SSE events."""
    engine = session_store.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    if req.model:
        engine.model = req.model

    async def event_stream():
        try:
            async for event in engine.submit_message(req.prompt):
                yield _format_sse(event)

                # Persist messages + metadata when turn completes
                if event.type == "done":
                    await session_store.save_messages(session_id, engine.messages[-engine.turn_count * 2 :])
                    await session_store.save_session_metadata(session_id, engine)
        except Exception as e:
            logger.exception("Error in message stream")
            yield _format_sse(QueryEvent(type="error", data={"message": str(e)}))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/sessions/{session_id}/messages/sync", response_model=MessageResponse)
async def send_message_sync(session_id: str, req: MessageRequest) -> MessageResponse:
    """Send a message and wait for the complete response (non-streaming)."""
    import time

    engine = session_store.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    if req.model:
        engine.model = req.model

    start = time.time()
    text_parts: list[str] = []
    tool_calls: list[ToolCallDetail] = []
    errors: list[str] = []
    stop_reason: str | None = None
    is_error = False
    usage_data = UsageInfo()

    try:
        async for event in engine.submit_message(req.prompt):
            if event.type == "text_delta":
                text_parts.append(event.data.get("content", ""))

            elif event.type == "tool_use_end":
                # Capture ALL tool calls across ALL turns (not just last turn)
                tool_calls.append(ToolCallDetail(
                    tool_use_id=event.data.get("tool_use_id", ""),
                    tool_name=event.data.get("tool_name", ""),
                    tool_input=event.data.get("tool_input", {}),
                    result=event.data.get("result", ""),
                    is_error=event.data.get("is_error", False),
                ))

            elif event.type == "done":
                usage_raw = event.data.get("usage", {})
                usage_data = UsageInfo(**{k: v for k, v in usage_raw.items() if k in UsageInfo.model_fields})
                stop_reason = "stop"

            elif event.type == "error":
                is_error = True
                errors.append(event.data.get("message", "Unknown error"))
                stop_reason = event.data.get("message", "error")

        # Persist messages
        await session_store.save_messages(session_id, engine.messages)

    except Exception as e:
        is_error = True
        errors.append(str(e))
        stop_reason = "error"

    duration_ms = int((time.time() - start) * 1000)
    subtype = "success" if not is_error else ("error_max_turns" if "Max turns" in " ".join(errors) else "error")

    # Build messages summary (last turn only)
    turn_messages = []
    for msg in engine.messages:
        turn_messages.append({
            "role": msg.role.value,
            "text": msg.get_text()[:500] if msg.get_text() else None,
            "tool_calls": [{"name": tc.name, "input": tc.input} for tc in msg.tool_calls] if msg.tool_calls else None,
            "tool_call_id": msg.tool_call_id,
        })

    return MessageResponse(
        type="result",
        subtype=subtype,
        text="".join(text_parts),
        model=engine.model,
        session_id=session_id,
        stop_reason=stop_reason,
        num_turns=engine.turn_count,
        duration_ms=duration_ms,
        is_error=is_error,
        errors=errors,
        usage=usage_data,
        tool_calls=tool_calls,
        messages=turn_messages,
    )


@app.post("/sessions/{session_id}/abort")
async def abort_session(session_id: str) -> dict[str, str]:
    """Abort the current query in a session."""
    engine = session_store.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")
    engine.abort()
    return {"status": "aborted"}


@app.post("/sessions/{session_id}/permissions/{tool_use_id}")
async def respond_permission(session_id: str, tool_use_id: str, req: PermissionResponse) -> dict[str, str]:
    """Respond to a permission request."""
    engine = session_store.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")
    engine.permissions.resolve_permission(tool_use_id, req.allow)
    return {"status": "resolved"}


@app.post("/sessions/{session_id}/compact")
async def compact_session(session_id: str) -> dict[str, Any]:
    """Trigger manual compaction of the session history."""
    from app.services.compact import apply_compaction, compact_conversation

    engine = session_store.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    try:
        result = await compact_conversation(
            messages=engine.messages,
            client=engine.openai_client,
            model=engine.fast_model,
        )
        engine.messages = apply_compaction(engine.messages, result)
        return {
            "status": "compacted",
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
        }
    except Exception as e:
        raise HTTPException(500, f"Compaction failed: {e}")


@app.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str) -> SessionResponse:
    """Resume a persisted session with full state restoration.

    Restores: messages, model, coordinator mode, permission mode, file cache, cwd.
    """
    # Load messages + metadata using optimized loader (backward scan for large files)
    messages, metadata = await session_store.load_transcript(session_id)

    if not messages:
        # Fallback: try the legacy load path
        from app.services.session_resume import load_session
        messages, meta2 = await load_session(session_store.base_dir, session_id)
        metadata.update(meta2)

    if not messages:
        raise HTTPException(404, "Session not found or empty")

    # Restore session configuration from metadata
    req = CreateSessionRequest(
        model=metadata.get("model"),
        cwd=metadata.get("cwd"),
        coordinator_mode=metadata.get("coordinator_mode", False),
        permission_mode=metadata.get("permission_mode", "auto"),
        system_prompt=metadata.get("custom_system_prompt"),
    )
    engine = _create_engine(session_id, req)
    engine.messages = messages

    # Restore turn count
    if metadata.get("turn_count"):
        engine.turn_count = metadata["turn_count"]

    # Rebuild file state cache from Read tool results in messages
    from app.utils.file_state import FileState
    from app.models import MessageRole
    for msg in messages:
        if msg.role == MessageRole.TOOL:
            text = msg.get_text() or ""
            # Detect Read tool results (have line numbers like "1\t...")
            if text and "\t" in text.split("\n")[0] if "\n" in text else False:
                # This looks like a Read tool result - we can't perfectly restore
                # but at least mark that SOMETHING was read
                pass

    # Detect and handle interrupted conversations
    from app.utils.messages import detect_interrupted_conversation
    _, was_interrupted = detect_interrupted_conversation(messages)
    if was_interrupted:
        from app.models import create_user_message
        engine.messages.append(create_user_message("Continue from where you left off."))

    session_store.register(session_id, engine)
    logger.info(
        "Resumed session %s: %d msgs, model=%s, coordinator=%s, interrupted=%s",
        session_id, len(messages), engine.model, engine.coordinator_mode, was_interrupted,
    )
    return SessionResponse(session_id=session_id, resumed=True)


@app.get("/sessions/persisted")
async def list_persisted_sessions() -> list[dict[str, Any]]:
    """List all persisted sessions from disk."""
    from app.services.session_resume import discover_sessions
    return await discover_sessions(session_store.base_dir)


# -- WebSocket (bidirectional alternative) ---------------------------------

@app.websocket("/sessions/{session_id}/ws")
async def websocket_session(websocket: WebSocket, session_id: str):
    """WebSocket for bidirectional real-time communication."""
    engine = session_store.get(session_id)
    if not engine:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message":
                async for event in engine.submit_message(data["prompt"]):
                    await websocket.send_json({"type": event.type, **event.data})

            elif msg_type == "abort":
                engine.abort()
                await websocket.send_json({"type": "aborted"})

            elif msg_type == "permission_response":
                engine.permissions.resolve_permission(data["tool_use_id"], data["allow"])
                await websocket.send_json({"type": "permission_resolved"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# -- Health ----------------------------------------------------------------

@app.get("/sessions/{session_id}/context")
async def get_session_context(session_id: str) -> dict[str, Any]:
    """Inspect the context state of a session (observability)."""
    engine = session_store.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    # Find checkpoints (compact boundaries) in messages
    checkpoints = []
    for i, msg in enumerate(engine.messages):
        if hasattr(msg, "system_message_type") and msg.system_message_type == "compact_boundary":
            checkpoints.append({
                "index": i,
                "uuid": msg.uuid,
                "summarized_count": len(msg.summarized_uuids) if hasattr(msg, "summarized_uuids") else 0,
                "summary_preview": (msg.get_text() or "")[:200],
            })

    return {
        "session_id": session_id,
        "model": engine.model,
        "coordinator_mode": engine.coordinator_mode,
        "turn_count": engine.turn_count,
        "message_count": len(engine.messages),
        "estimated_tokens": (
            sum(len(m.get_text() or "") for m in engine.messages) // 4
            + (len(engine._system_prompt_cache) // 4 if engine._system_prompt_cache else 0)
        ),
        "file_state_cache_size": len(engine.file_state_cache._cache),
        "memory_surfaced_count": len(engine._memory_surfaced),
        "notification_queue_pending": engine.notification_queue.pending_count,
        "content_replacements": {
            "seen_ids": len(engine._content_replacement_state.seen_ids),
            "frozen_replacements": len(engine._content_replacement_state.replacements),
        },
        "tool_decision_cache_size": len(engine._tool_decisions),
        "compact_state": {
            "consecutive_failures": engine._compact_state.consecutive_failures,
            "last_compact_turn": engine._compact_state.last_compact_turn,
            "enabled": engine._compact_state.enabled,
            "total_compactions": engine._compact_state.total_compactions,
        },
        "checkpoints": checkpoints,
        "query_tracking": {
            "chain_id": engine._query_tracking.chain_id,
            "depth": engine._query_tracking.depth,
        },
        "total_usage": engine.total_usage.model_dump(),
        "context_log_last_10": engine._context_log[-10:],
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.default_model,
        "active_sessions": len(session_store.list_active()),
    }
