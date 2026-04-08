"""Tool registry factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.tools.base import ToolRegistry
from app.tools.bash import BashTool
from app.tools.filesystem import EditTool, GlobTool, GrepTool, ReadTool, WriteTool
from app.tools.plan_mode import EnterPlanModeTool, ExitPlanModeTool
from app.tools.todo import TodoWriteTool
from app.tools.web_fetch import WebFetchTool
from app.tools.web_search import WebSearchTool

if TYPE_CHECKING:
    from app.agents.definitions import AgentRegistry
    from app.services.llm_adapter import LLMAdapter
    from app.settings import Settings


def create_default_registry() -> ToolRegistry:
    """Create a registry with core tools (no Agent tool - needs extra deps)."""
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(EditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())
    registry.register(TodoWriteTool())
    registry.register(EnterPlanModeTool())
    registry.register(ExitPlanModeTool())
    return registry


def create_full_registry(
    adapter: LLMAdapter,
    agent_registry: AgentRegistry,
    settings: Settings,
) -> ToolRegistry:
    """Create a registry with all tools including Agent, SendMessage, TaskStop, TaskOutput."""
    from app.tools.agent_tool import AgentTool
    from app.tools.send_message import SendMessageTool
    from app.tools.task_tools import TaskOutputTool, TaskStopTool

    registry = create_default_registry()

    agent_tool = AgentTool(
        adapter=adapter,
        tool_registry=registry,
        agent_registry=agent_registry,
        settings=settings,
    )
    registry.register(agent_tool)
    registry.register(SendMessageTool(agent_tool=agent_tool))
    registry.register(TaskStopTool(agent_tool=agent_tool))
    registry.register(TaskOutputTool(agent_tool=agent_tool))

    return registry
