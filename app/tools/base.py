"""Tool interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.services.llm_adapter import ToolDefinition

if TYPE_CHECKING:
    from app.models import Message


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}  # JSON Schema

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return False

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        """Whether this tool can safely run in parallel with other concurrency-safe tools.

        Separate from is_read_only: a tool can be read-only but not concurrency-safe
        (e.g., Bash with complex state dependencies), or concurrency-safe but not read-only
        (e.g., Agent tool that delegates permission checks internally).

        Defaults to is_read_only for backward compatibility.
        """
        return self.is_read_only(input)

    def is_enabled(self) -> bool:
        return True

    @abstractmethod
    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with given input."""
        ...

    def to_definition(self) -> ToolDefinition:
        """Convert to provider-agnostic tool definition for the LLM."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass
class ToolContext:
    """Context available to tools during execution."""

    session_id: str
    cwd: str
    messages: list[Message] = field(default_factory=list)
    agent_id: str | None = None
    permission_mode: str = "auto"
    abort_signal: Any = None  # asyncio.Event
    file_state_cache: Any = None  # FileStateCache (avoid circular import)
    session_dir: str | None = None  # For tool result persistence
    notification_queue: Any = None  # NotificationQueue (avoid circular import)


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.is_enabled()]

    def definitions(self) -> list[ToolDefinition]:
        return [t.to_definition() for t in self.all()]

    def filter(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        read_only: bool = False,
    ) -> list[Tool]:
        """Filter tools by name lists or read-only flag."""
        tools = self.all()
        if include is not None:
            include_set = set(include)
            tools = [t for t in tools if t.name in include_set]
        if exclude is not None:
            exclude_set = set(exclude)
            tools = [t for t in tools if t.name not in exclude_set]
        if read_only:
            # Return only tools that are read-only for empty input (heuristic)
            tools = [t for t in tools if t.is_read_only({})]
        return tools

    def filtered_definitions(self, **kwargs: Any) -> list[ToolDefinition]:
        return [t.to_definition() for t in self.filter(**kwargs)]
