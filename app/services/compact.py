"""Compaction service - summarizes old messages to free context window.

Supports:
- Full compaction: summarize everything, keep nothing
- Partial compaction: summarize old messages, preserve recent ones
- Session metadata entries alongside boundaries
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.models import (
    CompactBoundary,
    Message,
    MessageRole,
    TokenUsage,
    create_system_message,
)
from app.utils.side_query import side_query

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_CONTEXT_WINDOW = 128_000
AUTOCOMPACT_BUFFER = 13_000
MAX_CONSECUTIVE_FAILURES = 3

# Partial compaction: preserve at least this many recent messages
MIN_PRESERVE_MESSAGES = 4
# Partial compaction: preserve at least this many tokens of recent context
MIN_PRESERVE_TOKENS = 2000

COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far.
This summary will replace the conversation history, so capture ALL important information.

CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

## Summary Sections
1. **Primary Request and Intent**: All explicit user requests and goals
2. **Key Technical Concepts**: Technologies, frameworks, patterns discussed
3. **Files and Code Sections**: Specific files examined/modified with relevant code snippets
4. **Errors and Fixes**: Problems encountered and how they were resolved
5. **Problem Solving**: Approaches tried, what worked, what didn't
6. **All User Messages**: List ALL non-tool-result user messages (preserve exact intent)
7. **Pending Tasks**: Explicitly requested tasks not yet completed
8. **Current Work**: What was being worked on immediately before this summary, with file names and code context
9. **Next Step**: If applicable, the immediate next action with direct quotes showing where you left off

Be thorough - information not captured here will be lost."""

COMPACT_SYSTEM = (
    "You are a conversation summarizer. Create a detailed, structured summary "
    "of the conversation. Respond with text only - do NOT use any tools."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    boundary: CompactBoundary
    tokens_before: int
    tokens_after: int
    summary: str
    messages_summarized: int = 0
    messages_preserved: int = 0
    trigger: str = "auto"
    duration_ms: int = 0


@dataclass
class AutoCompactState:
    consecutive_failures: int = 0
    last_compact_turn: int = 0
    enabled: bool = True
    total_compactions: int = 0


@dataclass
class SessionMetadataEntry:
    """Metadata entry persisted in JSONL alongside messages."""
    type: str = "session_metadata"
    model: str = ""
    coordinator_mode: bool = False
    permission_mode: str = "auto"
    custom_system_prompt: str | None = None
    cwd: str = ""
    timestamp: float = 0.0

    def to_message(self) -> Message:
        return create_system_message(json.dumps({
            "type": self.type,
            "model": self.model,
            "coordinator_mode": self.coordinator_mode,
            "permission_mode": self.permission_mode,
            "custom_system_prompt": self.custom_system_prompt[:200] if self.custom_system_prompt else None,
            "cwd": self.cwd,
            "timestamp": self.timestamp or time.time(),
        }))

    @classmethod
    def from_message(cls, msg: Message) -> SessionMetadataEntry | None:
        text = msg.get_text()
        if not text:
            return None
        try:
            data = json.loads(text)
            if data.get("type") != "session_metadata":
                return None
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(messages: list[Message]) -> int:
    total_chars = 0
    for msg in messages:
        text = msg.get_text()
        total_chars += len(text) if text else 0
        for tc in msg.tool_calls:
            total_chars += len(str(tc.input))
    return total_chars // CHARS_PER_TOKEN


def should_auto_compact(
    messages: list[Message],
    state: AutoCompactState,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> bool:
    if not state.enabled:
        return False
    if state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        logger.warning("Auto-compact disabled: %d consecutive failures", state.consecutive_failures)
        return False
    threshold = context_window - AUTOCOMPACT_BUFFER
    current = estimate_tokens(messages)
    return current > threshold


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

async def compact_conversation(
    messages: list[Message],
    client: Any,
    model: str,
    custom_instructions: str | None = None,
    trigger: str = "auto",
) -> CompactionResult:
    """Compact conversation with partial preservation of recent messages.

    1. Find split point (keep recent messages totaling >= MIN_PRESERVE_TOKENS)
    2. Summarize old messages via LLM
    3. Create CompactBoundary
    4. Return boundary + preserved messages
    """
    start_time = time.time()
    tokens_before = estimate_tokens(messages)

    # Find split point: walk backward to find where to cut
    split_idx = _find_split_point(messages)
    old_messages = messages[:split_idx]
    preserved_messages = messages[split_idx:]

    # If nothing to summarize, skip
    if not old_messages:
        logger.info("Nothing to compact (all messages are recent)")
        boundary = CompactBoundary(summary="(no prior context)", summarized_uuids=[])
        return CompactionResult(
            boundary=boundary,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            summary="(no prior context)",
            trigger=trigger,
        )

    # Build prompt and call LLM
    prompt = COMPACT_PROMPT
    if custom_instructions:
        prompt += f"\n\n## Additional Instructions\n{custom_instructions}"

    conversation_text = _format_conversation_for_compact(old_messages)
    user_prompt = f"{conversation_text}\n\n---\n\n{prompt}"

    summary = await side_query(
        client=client,
        model=model,
        system=COMPACT_SYSTEM,
        user=user_prompt,
        max_tokens=4096,
    )
    summary = _format_summary(summary)

    # Create boundary
    summarized_uuids = [msg.uuid for msg in old_messages]
    boundary = CompactBoundary(summary=summary, summarized_uuids=summarized_uuids)

    tokens_after = estimate_tokens([boundary] + preserved_messages)
    duration_ms = int((time.time() - start_time) * 1000)

    logger.info(
        "Compacted: %d msgs summarized, %d preserved, %d->%d tokens (%dms)",
        len(old_messages), len(preserved_messages), tokens_before, tokens_after, duration_ms,
    )

    return CompactionResult(
        boundary=boundary,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        summary=summary,
        messages_summarized=len(old_messages),
        messages_preserved=len(preserved_messages),
        trigger=trigger,
        duration_ms=duration_ms,
    )


def apply_compaction(messages: list[Message], result: CompactionResult) -> list[Message]:
    """Replace old messages with boundary + preserved recent messages.

    Returns: [boundary] + [preserved recent messages]
    """
    split_idx = _find_split_point(messages)
    preserved = messages[split_idx:]

    # Fix parent_uuid chain: first preserved message points to boundary
    new_messages = [result.boundary]
    for msg in preserved:
        if not new_messages[-1:]:
            msg.parent_uuid = result.boundary.uuid
        else:
            msg.parent_uuid = new_messages[-1].uuid
        new_messages.append(msg)

    return new_messages


def _find_split_point(messages: list[Message]) -> int:
    """Find where to split messages: old (to summarize) | recent (to keep).

    Walks backward, preserving at least MIN_PRESERVE_MESSAGES and MIN_PRESERVE_TOKENS.
    Also ensures we don't split in the middle of a tool_use/tool_result pair.
    """
    if len(messages) <= MIN_PRESERVE_MESSAGES:
        return 0  # Nothing to compact

    # Walk backward accumulating tokens
    preserve_tokens = 0
    preserve_count = 0
    split_idx = len(messages)

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        text = msg.get_text() or ""
        msg_tokens = len(text) // CHARS_PER_TOKEN

        preserve_tokens += msg_tokens
        preserve_count += 1
        split_idx = i

        # Stop when we have enough preserved context
        if preserve_count >= MIN_PRESERVE_MESSAGES and preserve_tokens >= MIN_PRESERVE_TOKENS:
            break

    # Adjust split point to avoid breaking tool_use/tool_result pairs
    # Walk forward from split_idx to find a safe boundary (not mid-tool-call)
    while split_idx > 0 and split_idx < len(messages):
        msg = messages[split_idx]
        # Don't split right before a tool_result (it needs its tool_use)
        if msg.role == MessageRole.TOOL:
            split_idx -= 1
            continue
        # Don't split right after an assistant with tool_calls (results follow)
        if split_idx > 0:
            prev = messages[split_idx - 1]
            if prev.role == MessageRole.ASSISTANT and prev.tool_calls:
                split_idx -= 1
                continue
        break

    return max(0, split_idx)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_conversation_for_compact(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.role.value.upper()
        text = msg.get_text()

        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            tool_names = ", ".join(tc.name for tc in msg.tool_calls)
            if text:
                parts.append(f"[{role}]: {text}\n[Used tools: {tool_names}]")
            else:
                parts.append(f"[{role}]: [Used tools: {tool_names}]")
        elif msg.role == MessageRole.TOOL:
            preview = text[:2000] if text else "(empty)"
            if text and len(text) > 2000:
                preview += f"\n... ({len(text) - 2000} more chars)"
            parts.append(f"[TOOL RESULT]: {preview}")
        elif text:
            parts.append(f"[{role}]: {text}")

    return "\n\n".join(parts)


def _format_summary(raw: str) -> str:
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"</?summary>", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
