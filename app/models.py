"""Internal message types and JSONL serialization."""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    type: str = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: str = "tool_use"
    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:24]}")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: str = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


# ---------------------------------------------------------------------------
# Messages (internal representation)
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class Message(BaseModel):
    """Internal message representation, provider-agnostic."""

    uuid: str = Field(default_factory=lambda: uuid4().hex)
    parent_uuid: str | None = None
    role: MessageRole
    content: list[ContentBlock] = Field(default_factory=list)
    text: str | None = None  # Shorthand for single text content
    tool_calls: list[ToolUseBlock] = Field(default_factory=list)
    tool_call_id: str | None = None  # For role=tool messages
    usage: TokenUsage | None = None
    timestamp: float = Field(default_factory=time.time)
    agent_id: str | None = None
    is_sidechain: bool = False

    def get_text(self) -> str:
        """Extract concatenated text from content blocks or text field."""
        if self.text is not None:
            return self.text
        parts = [b.text for b in self.content if isinstance(b, TextBlock)]
        return "\n".join(parts)


def create_user_message(text: str, **kwargs: Any) -> Message:
    return Message(role=MessageRole.USER, text=text, **kwargs)


def create_assistant_message(
    text: str | None = None,
    tool_calls: list[ToolUseBlock] | None = None,
    usage: TokenUsage | None = None,
    **kwargs: Any,
) -> Message:
    msg = Message(role=MessageRole.ASSISTANT, text=text, usage=usage, **kwargs)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


def create_tool_result_message(tool_call_id: str, content: str, is_error: bool = False, **kwargs: Any) -> Message:
    return Message(
        role=MessageRole.TOOL,
        tool_call_id=tool_call_id,
        text=content,
        content=[ToolResultBlock(type="tool_result", tool_call_id=tool_call_id, content=content, is_error=is_error)],
        **kwargs,
    )


def create_system_message(text: str, **kwargs: Any) -> Message:
    return Message(role=MessageRole.SYSTEM, text=text, **kwargs)


# ---------------------------------------------------------------------------
# Compact boundary (checkpoint)
# ---------------------------------------------------------------------------

class CompactBoundary(Message):
    """Marks a compaction point. Everything before is summarized."""

    system_message_type: str = "compact_boundary"
    summarized_uuids: list[str] = Field(default_factory=list)

    def __init__(self, summary: str, summarized_uuids: list[str] | None = None, **kwargs: Any):
        super().__init__(
            role=MessageRole.SYSTEM,
            text=summary,
            parent_uuid=None,  # Breaks the chain
            **kwargs,
        )
        self.summarized_uuids = summarized_uuids or []


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------

def message_to_jsonl(msg: Message) -> str:
    """Serialize a message to a single JSONL line."""
    return msg.model_dump_json(exclude_none=True)


def message_from_jsonl(line: str) -> Message:
    """Deserialize a message from a JSONL line."""
    data = json.loads(line)
    if data.get("system_message_type") == "compact_boundary":
        return CompactBoundary.model_validate(data)
    return Message.model_validate(data)
