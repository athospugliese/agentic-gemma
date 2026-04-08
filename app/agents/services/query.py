"""Core query loop - the heart of the agent system.

Flow: user message -> system prompt + tools -> LLM API (streaming) -> tool dispatch -> loop
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from app.models import (
    Message,
    TokenUsage,
    ToolUseBlock,
    create_assistant_message,
    create_tool_result_message,
    create_user_message,
)
from app.services.llm_adapter import LLMAdapter, StreamEvent, ToolDefinition
from app.services.permissions import PermissionDecision, PermissionManager
from app.services.prompt import build_system_prompt, load_claude_md
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from app.services.compact import AutoCompactState, apply_compaction, compact_conversation, estimate_tokens, should_auto_compact
from app.utils.file_state import FileStateCache
from app.utils.messages import normalize_messages_for_api
from app.utils.notification_queue import NotificationQueue
from app.utils.tool_result_storage import get_threshold_for_tool, maybe_truncate_result

# Characters-per-token estimate for context window tracking
CHARS_PER_TOKEN = 4
GIT_REFRESH_INTERVAL = 5  # Re-fetch git context every N turns
MAX_MEMORY_SURFACED = 50  # Cap on _memory_surfaced set size


@dataclass
class ContentReplacementState:
    """Tracks tool result budget decisions. Once frozen, never changes (cache stability)."""
    seen_ids: set[str] = field(default_factory=set)
    replacements: dict[str, str] = field(default_factory=dict)  # tool_use_id -> preview

    def clone(self) -> "ContentReplacementState":
        return ContentReplacementState(
            seen_ids=set(self.seen_ids),
            replacements=dict(self.replacements),
        )


@dataclass
class QueryTracking:
    """Tracks agent chain depth for telemetry and recursion detection."""
    chain_id: str
    depth: int = 0

    def child(self, child_id: str) -> "QueryTracking":
        return QueryTracking(chain_id=child_id, depth=self.depth + 1)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE event types sent to the client
# ---------------------------------------------------------------------------

@dataclass
class QueryEvent:
    """Event yielded from the query loop to the HTTP layer."""

    type: str  # text_delta, tool_use_start, tool_use_end, agent_spawned, agent_completed, error, done, permission_request
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

class QueryEngine:
    """Manages one conversation session with the LLM.

    Handles: message accumulation, system prompt, tool dispatch, error recovery.
    """

    def __init__(
        self,
        adapter: LLMAdapter,
        registry: ToolRegistry,
        permissions: PermissionManager,
        session_id: str,
        model: str = "gpt-4o",
        fast_model: str = "gpt-4o-mini",
        max_tokens: int = 8192,
        max_turns: int = 200,
        cwd: str = ".",
        custom_system_prompt: str | None = None,
        openai_client: Any = None,
        data_dir: str = "~/.agent",
        coordinator_mode: bool = False,
    ):
        self.adapter = adapter
        self.registry = registry
        self.permissions = permissions
        self.session_id = session_id
        self.model = model
        self.fast_model = fast_model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.cwd = cwd
        self.custom_system_prompt = custom_system_prompt
        self.openai_client = openai_client
        self.data_dir = data_dir
        self.coordinator_mode = coordinator_mode

        self.messages: list[Message] = []
        self.turn_count = 0
        self.total_usage = TokenUsage()
        self._abort_event = asyncio.Event()
        self.file_state_cache = FileStateCache()
        self._session_dir: str | None = None
        self._git_context: str | None = None
        self._git_context_turn: int = 0  # Turn when git was last fetched
        self._system_prompt_cache: str | None = None
        self._compact_state = AutoCompactState()
        self._memory_surfaced: set[str] = set()
        self._last_extraction_uuid: str | None = None
        self.notification_queue = NotificationQueue()
        self._content_replacement_state = ContentReplacementState()
        self._tool_decisions: dict[str, bool] = {}  # tool_key -> allowed (cache)
        self._query_tracking = QueryTracking(chain_id=self.session_id, depth=0)
        self._context_log: list[dict[str, Any]] = []  # Observability log
        self._has_attempted_reactive_compact = False  # PTL recovery guard
        self._max_output_recovery_count = 0  # finish_reason=length recovery attempts
        self._effective_max_tokens = max_tokens  # Can be escalated on truncation
        self._hooks_config: dict[str, list] = {}  # Populated from settings

    def _log_context_event(self, event: str, source: str, detail: str = "", size: int = 0) -> None:
        """Log a context mutation for observability."""
        import time
        self._context_log.append({
            "timestamp": time.time(),
            "turn": self.turn_count,
            "event": event,
            "source": source,
            "detail": detail[:200],
            "size": size,
        })
        # Keep bounded
        if len(self._context_log) > 500:
            self._context_log = self._context_log[-250:]

    def abort(self) -> None:
        """Signal the query loop to abort."""
        self._abort_event.set()

    async def submit_message(self, prompt: str) -> AsyncGenerator[QueryEvent, None]:
        """Submit a user message and stream query events.

        This is the main entry point. It:
        1. Adds the user message
        2. Builds system prompt (cached, with git + memory + coordinator)
        3. Runs memory prefetch in parallel
        4. Calls LLM with streaming
        5. Dispatches tool calls
        6. Loops until done or max_turns
        7. Fires background memory extraction on completion
        """
        self._abort_event.clear()

        # Add user message
        user_msg = create_user_message(prompt)
        if self.messages:
            user_msg.parent_uuid = self.messages[-1].uuid
        self.messages.append(user_msg)

        # Build system prompt (cached per session, rebuilt only once)
        if self._system_prompt_cache is None:
            from app.utils.git import get_git_context
            self._git_context = await get_git_context(self.cwd)
            self._git_context_turn = self.turn_count

            from app.services.memory.loader import load_memory_prompt
            memory_prompt = load_memory_prompt(self.cwd, self.data_dir)

            coordinator_prompt = None
            if self.coordinator_mode:
                from app.services.coordinator import get_coordinator_prompt
                tool_names = [t.name for t in self.registry.all()]
                coordinator_prompt = get_coordinator_prompt(tool_names, cwd=self.cwd)

            agent_defs = None
            agent_tool = self.registry.get("Agent")
            if agent_tool and hasattr(agent_tool, '_agent_registry'):
                agent_defs = agent_tool._agent_registry.all()

            claude_md = load_claude_md(self.cwd)
            self._system_prompt_cache = build_system_prompt(
                tools=self.registry.all(),
                cwd=self.cwd,
                custom_prompt=self.custom_system_prompt,
                claude_md_content=claude_md,
                memory_prompt=memory_prompt,
                coordinator_prompt=coordinator_prompt,
                agent_definitions=agent_defs,
            )
            self._log_context_event("system_prompt_built", "submit_message", size=len(self._system_prompt_cache))

        # Re-fetch git context every N turns (branch may change)
        if self.turn_count - self._git_context_turn >= GIT_REFRESH_INTERVAL:
            from app.utils.git import get_git_context
            self._git_context = await get_git_context(self.cwd)
            self._git_context_turn = self.turn_count
            self._log_context_event("git_refreshed", "submit_message")

        # Assemble per-turn system prompt (git appended fresh)
        system_prompt = self._system_prompt_cache
        if self._git_context:
            system_prompt = f"{system_prompt}\n\n# Git Context\n{self._git_context}"

        # Context window check: warn if approaching limit
        estimated_tokens = estimate_tokens(self.messages) + len(system_prompt) // CHARS_PER_TOKEN
        self._log_context_event("context_estimate", "submit_message", f"~{estimated_tokens} tokens", estimated_tokens)

        # Prefetch relevant memories (inject as system reminder)
        memory_attachment = await self._prefetch_memories(prompt)
        if memory_attachment:
            from app.models import create_system_message
            reminder = create_system_message(memory_attachment)
            self.messages.insert(-1, reminder)
            self._log_context_event("memory_injected", "prefetch", size=len(memory_attachment))

        # Cap memory_surfaced to prevent unbounded growth
        if len(self._memory_surfaced) > MAX_MEMORY_SURFACED:
            # Keep newest half
            self._memory_surfaced = set(list(self._memory_surfaced)[-MAX_MEMORY_SURFACED // 2:])

        tool_defs = self.registry.definitions()

        # Main query loop
        async for event in self._query_loop(system_prompt, tool_defs):
            yield event

        # Background memory extraction (fire-and-forget)
        asyncio.create_task(self._extract_memories_background())

    async def _prefetch_memories(self, query: str) -> str | None:
        """Prefetch relevant memories for the current query."""
        if not self.openai_client:
            return None
        try:
            from pathlib import Path
            from app.services.memory.loader import get_memory_dir
            from app.services.memory.ranker import find_relevant_memories, format_memories_as_attachment

            memory_dir = get_memory_dir(self.cwd, self.data_dir)
            if not memory_dir.exists():
                return None

            memories = await find_relevant_memories(
                query=query,
                memory_dir=memory_dir,
                client=self.openai_client,
                model=self.fast_model,
                already_surfaced=self._memory_surfaced,
            )

            if memories:
                for m in memories:
                    self._memory_surfaced.add(m["path"])
                return format_memories_as_attachment(memories)
        except Exception:
            logger.debug("Memory prefetch failed", exc_info=True)
        return None

    async def _extract_memories_background(self) -> None:
        """Background task: extract memories from recent conversation."""
        if not self.openai_client:
            return
        try:
            from pathlib import Path
            from app.services.memory.loader import get_memory_dir
            from app.services.memory.extractor import extract_memories

            memory_dir = get_memory_dir(self.cwd, self.data_dir)
            written = await extract_memories(
                messages=self.messages,
                memory_dir=memory_dir,
                client=self.openai_client,
                model=self.fast_model,
                last_cursor_uuid=self._last_extraction_uuid,
            )

            if self.messages:
                self._last_extraction_uuid = self.messages[-1].uuid

            if written:
                logger.info("Background extraction wrote %d memories", len(written))
        except Exception:
            logger.debug("Background memory extraction failed", exc_info=True)

    async def _query_loop(
        self,
        system_prompt: str,
        tool_defs: list[ToolDefinition],
    ) -> AsyncGenerator[QueryEvent, None]:
        """Core loop: call LLM -> execute tools -> repeat.

        Each iteration:
        1. Auto-compact if context too large
        2. Normalize messages (pair tool_use/result, merge, strip)
        3. Stream LLM response
        4. Execute tool calls
        5. Loop if tools were called
        """

        while self.turn_count < self.max_turns:
            if self._abort_event.is_set():
                yield QueryEvent(type="error", data={"message": "Aborted"})
                return

            self.turn_count += 1

            # --- Drain task notifications from background agents ---
            notifications = self.notification_queue.drain()
            for notif in notifications:
                notif_msg = create_user_message(notif)
                if self.messages:
                    notif_msg.parent_uuid = self.messages[-1].uuid
                self.messages.append(notif_msg)
                yield QueryEvent(type="task_notification", data={"content": notif})

            # --- Auto-compact if context is too large ---
            if should_auto_compact(self.messages, self._compact_state):
                try:
                    result = await compact_conversation(
                        messages=self.messages,
                        client=self.openai_client,
                        model=self.fast_model,
                    )
                    self.messages = apply_compaction(self.messages, result)
                    self.file_state_cache.clear()  # Prevent stale edits after compaction
                    self._compact_state.consecutive_failures = 0
                    self._compact_state.last_compact_turn = self.turn_count
                    self._compact_state.total_compactions += 1

                    self._log_context_event(
                        "compaction", "auto",
                        f"summarized={result.messages_summarized} preserved={result.messages_preserved} "
                        f"{result.tokens_before}->{result.tokens_after} tokens ({result.duration_ms}ms)",
                        result.tokens_after,
                    )

                    yield QueryEvent(type="compact", data={
                        "tokens_before": result.tokens_before,
                        "tokens_after": result.tokens_after,
                        "messages_summarized": result.messages_summarized,
                        "messages_preserved": result.messages_preserved,
                        "duration_ms": result.duration_ms,
                    })
                except Exception:
                    logger.warning("Auto-compact failed", exc_info=True)
                    self._compact_state.consecutive_failures += 1
                    self._log_context_event("compaction_failed", "auto", f"failure #{self._compact_state.consecutive_failures}")

            # --- Normalize messages before API call ---
            api_messages = normalize_messages_for_api(self.messages)

            # --- Stream LLM response ---
            stream_events: list[StreamEvent] = []
            try:
                async for event in self.adapter.create_completion(
                    messages=api_messages,
                    system_prompt=system_prompt,
                    tools=tool_defs,
                    model=self.model,
                    max_tokens=self._effective_max_tokens,
                ):
                    stream_events.append(event)

                    # Forward text deltas
                    if event.type == "text_delta":
                        yield QueryEvent(type="text_delta", data=event.data)

                    # Forward tool call starts
                    elif event.type == "tool_call_start":
                        yield QueryEvent(type="tool_use_start", data=event.data)

            except Exception as e:
                from app.services.llm_adapter import LLMError, LLMErrorCategory

                if isinstance(e, LLMError):
                    # --- PTL Recovery: context_length_exceeded → compact and retry ---
                    if (
                        e.category == LLMErrorCategory.CONTEXT_TOO_LONG
                        and not self._has_attempted_reactive_compact
                    ):
                        self._has_attempted_reactive_compact = True
                        logger.warning("Context too long — attempting reactive compact")
                        try:
                            result = await compact_conversation(
                                messages=self.messages,
                                client=self.openai_client,
                                model=self.fast_model,
                            )
                            self.messages = apply_compaction(self.messages, result)
                            self.file_state_cache.clear()
                            self._compact_state.total_compactions += 1
                            self._log_context_event(
                                "reactive_compact", "ptl_recovery",
                                f"{result.tokens_before}->{result.tokens_after} tokens",
                                result.tokens_after,
                            )
                            yield QueryEvent(type="compact", data={
                                "tokens_before": result.tokens_before,
                                "tokens_after": result.tokens_after,
                                "trigger": "reactive_ptl",
                            })
                            # Decrement turn count so this retry doesn't count as a turn
                            self.turn_count -= 1
                            continue  # Retry the LLM call with compacted context
                        except Exception:
                            logger.exception("Reactive compact failed")

                    # Classified error — report with category
                    logger.error("LLM error [%s]: %s", e.category, e.message[:300])
                    yield QueryEvent(type="error", data={
                        "message": str(e),
                        "category": e.category,
                        "status_code": e.status_code,
                    })
                else:
                    logger.exception("LLM API error (unclassified)")
                    yield QueryEvent(type="error", data={"message": str(e)})
                return

            # --- Build assistant message from stream ---
            assistant_msg = self.adapter.build_assistant_message(stream_events)
            if assistant_msg.usage:
                self._accumulate_usage(assistant_msg.usage)
            if self.messages:
                assistant_msg.parent_uuid = self.messages[-1].uuid
            self.messages.append(assistant_msg)

            # Reset reactive compact flag on successful API call
            self._has_attempted_reactive_compact = False

            # --- Check finish_reason for output truncation ---
            done_event = next((e for e in stream_events if e.type == "done"), None)
            finish_reason = done_event.data.get("finish_reason") if done_event else None

            if finish_reason == "length":
                # Output was truncated — attempt recovery
                if self._max_output_recovery_count < 3:
                    self._max_output_recovery_count += 1

                    # Escalate max_tokens on first truncation
                    if self._max_output_recovery_count == 1 and self.max_tokens < 16384:
                        self._effective_max_tokens = min(16384, self.max_tokens * 2)
                        self._log_context_event("max_tokens_escalated", "length_recovery",
                                                f"{self.max_tokens}->{self._effective_max_tokens}")

                    # Inject continuation nudge and loop
                    nudge = create_user_message(
                        "Output token limit hit. Resume directly from where you stopped — "
                        "no apology, no repetition, no summary of what you already said."
                    )
                    if self.messages:
                        nudge.parent_uuid = self.messages[-1].uuid
                    self.messages.append(nudge)
                    self._log_context_event("output_truncation_recovery", "length",
                                            f"attempt {self._max_output_recovery_count}/3")
                    continue  # Loop back for continuation

            # --- Check if we need to execute tools ---
            if not assistant_msg.tool_calls:
                # No tool calls = final response
                # Reset output recovery state
                self._max_output_recovery_count = 0
                self._effective_max_tokens = self.max_tokens
                yield QueryEvent(type="done", data={
                    "usage": self.total_usage.model_dump(),
                    "turn_count": self.turn_count,
                })
                return

            # Reset output recovery on successful tool-call turn
            self._max_output_recovery_count = 0

            # --- Execute tools ---
            tool_results = await self._execute_tools(assistant_msg.tool_calls)

            # Yield tool results and add to messages
            for tool_call, result, result_msg in tool_results:
                yield QueryEvent(type="tool_use_end", data={
                    "tool_use_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.input,
                    "result": result.output[:2000],  # Preview
                    "is_error": result.is_error,
                })
                if self.messages:
                    result_msg.parent_uuid = self.messages[-1].uuid
                self.messages.append(result_msg)

            # Continue loop -> next LLM call with tool results

        # Max turns reached
        yield QueryEvent(type="error", data={"message": f"Max turns ({self.max_turns}) reached"})

    async def _execute_tools(
        self,
        tool_calls: list[ToolUseBlock],
    ) -> list[tuple[ToolUseBlock, ToolResult, Message]]:
        """Execute tool calls, respecting concurrency rules.

        Read-only tools run concurrently, mutating tools run serially.
        """
        # Partition into concurrent (read-only) and serial (mutating) batches
        read_only_batch: list[ToolUseBlock] = []
        serial_queue: list[ToolUseBlock] = []

        for tc in tool_calls:
            tool = self.registry.get(tc.name)
            if tool and tool.is_concurrency_safe(tc.input):
                read_only_batch.append(tc)
            else:
                serial_queue.append(tc)

        results: list[tuple[ToolUseBlock, ToolResult, Message]] = []

        # Run read-only tools concurrently
        if read_only_batch:
            coros = [self._run_single_tool(tc) for tc in read_only_batch]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)
            for tc, res in zip(read_only_batch, batch_results):
                if isinstance(res, Exception):
                    tr = ToolResult(output=f"Tool error: {res}", is_error=True)
                    msg = create_tool_result_message(tc.id, tr.output, is_error=True)
                    results.append((tc, tr, msg))
                else:
                    tr, msg = res
                    results.append((tc, tr, msg))

        # Run mutating tools serially
        for tc in serial_queue:
            try:
                tr, msg = await self._run_single_tool(tc)
                results.append((tc, tr, msg))
            except Exception as e:
                tr = ToolResult(output=f"Tool error: {e}", is_error=True)
                msg = create_tool_result_message(tc.id, tr.output, is_error=True)
                results.append((tc, tr, msg))

        return results

    async def _run_single_tool(self, tc: ToolUseBlock) -> tuple[ToolResult, Message]:
        """Run a single tool call with permission checking, budget tracking, and result truncation."""
        tool = self.registry.get(tc.name)
        if not tool:
            result = ToolResult(output=f"Unknown tool: {tc.name}", is_error=True)
            return result, create_tool_result_message(tc.id, result.output, is_error=True)

        # Permission check with decision cache
        cache_key = f"{tc.name}:{hash(str(sorted(tc.input.items())) if tc.input else '')}"
        if cache_key in self._tool_decisions:
            allowed = self._tool_decisions[cache_key]
            if not allowed:
                result = ToolResult(output=f"Permission denied (cached)", is_error=True)
                return result, create_tool_result_message(tc.id, result.output, is_error=True)
        else:
            decision = await self.permissions.check(tc.name, tc.input, tool.is_read_only(tc.input))
            # Cache the decision for this tool+input pattern
            self._tool_decisions[cache_key] = decision.allowed
            if not decision.allowed:
                result = ToolResult(output=f"Permission denied: {decision.reason}", is_error=True)
                return result, create_tool_result_message(tc.id, result.output, is_error=True)

        # Build ToolContext with COPIED messages (prevent tool from mutating conversation)
        context = ToolContext(
            session_id=self.session_id,
            cwd=self.cwd,
            messages=list(self.messages),  # Shallow copy prevents mutation
            permission_mode=self.permissions.mode,
            file_state_cache=self.file_state_cache,
            session_dir=self._session_dir,
            notification_queue=self.notification_queue,
            abort_signal=self._abort_event,
        )

        self._log_context_event("tool_call", tc.name, f"input_keys={list(tc.input.keys())}")

        # Execute PreToolUse hooks
        if self._hooks_config:
            from app.utils.hooks import execute_hooks
            hook_context = {"tool_name": tc.name, "tool_input": tc.input, "event": "PreToolUse"}
            hook_results = await execute_hooks("PreToolUse", self._hooks_config, hook_context, self.cwd)
            for hr in hook_results:
                if hr.blocked:
                    result = ToolResult(output=f"Blocked by PreToolUse hook: {hr.reason}", is_error=True)
                    return result, create_tool_result_message(tc.id, result.output, is_error=True)

        result = await tool.call(tc.input, context)

        # Execute PostToolUse hooks (fire-and-forget, non-blocking)
        if self._hooks_config:
            from app.utils.hooks import execute_hooks
            hook_context = {
                "tool_name": tc.name, "tool_input": tc.input,
                "tool_result": result.output[:5000], "event": "PostToolUse",
            }
            asyncio.create_task(execute_hooks("PostToolUse", self._hooks_config, hook_context, self.cwd))

        # Content replacement: freeze budget decision per tool_use_id
        output = result.output
        if not result.is_error and tc.id not in self._content_replacement_state.seen_ids:
            self._content_replacement_state.seen_ids.add(tc.id)
            threshold = get_threshold_for_tool(tc.name)
            if len(output) > threshold and self._session_dir:
                output = maybe_truncate_result(
                    output=output,
                    tool_name=tc.name,
                    tool_use_id=tc.id,
                    session_dir=self._session_dir,
                    max_chars=threshold,
                )
                # Freeze the replacement preview (reuse on subsequent turns)
                self._content_replacement_state.replacements[tc.id] = output
                self._log_context_event("content_replaced", tc.name, f"id={tc.id}", len(result.output))
        elif tc.id in self._content_replacement_state.replacements:
            # Re-use frozen replacement (cache-stable)
            output = self._content_replacement_state.replacements[tc.id]

        msg = create_tool_result_message(tc.id, output, is_error=result.is_error)
        return result, msg

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        self.total_usage.prompt_tokens += usage.prompt_tokens
        self.total_usage.completion_tokens += usage.completion_tokens
        self.total_usage.total_tokens += usage.total_tokens
        self.total_usage.cached_tokens += usage.cached_tokens
        self.total_usage.reasoning_tokens += usage.reasoning_tokens
