"""Agent tool - spawns sub-agents to handle delegated tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.agents.definitions import AgentDefinition, AgentRegistry
from app.models import Message, create_user_message
from app.services.llm_adapter import LLMAdapter
from app.services.permissions import PermissionManager
from app.services.prompt import build_system_prompt
from app.services.query import QueryEngine, QueryEvent
from app.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class AgentTool(Tool):
    """Spawns a sub-agent to handle a delegated task.

    The sub-agent gets its own query loop with filtered tools and
    its own system prompt based on the agent definition.
    """

    name = "Agent"
    description = (
        "Launch a sub-agent to handle a complex task. The agent runs its own query loop "
        "with access to tools and returns its findings."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task for the sub-agent to perform.",
            },
            "description": {
                "type": "string",
                "description": "Short (3-5 word) description of the task.",
            },
            "agent_type": {
                "type": "string",
                "description": "Type of agent to use (e.g. 'Explore', 'Plan', 'general-purpose').",
            },
            "model": {
                "type": "string",
                "description": "Model override for this agent.",
            },
            "background": {
                "type": "boolean",
                "description": "Run the agent in the background.",
            },
        },
        "required": ["prompt", "description"],
    }

    def __init__(
        self,
        adapter: LLMAdapter,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        settings: Any,  # app.settings.Settings
    ):
        self._adapter = adapter
        self._tool_registry = tool_registry
        self._agent_registry = agent_registry
        self._settings = settings
        self._background_tasks: dict[str, asyncio.Task] = {}
        self._agent_results: dict[str, dict[str, Any]] = {}  # agent_id -> result data
        self._agent_engines: dict[str, Any] = {}  # agent_id -> QueryEngine (for SendMessage continuation)

    def is_read_only(self, input: dict[str, Any]) -> bool:
        # Agent delegates permission checks to its sub-tools internally
        return True

    def is_concurrency_safe(self, input: dict[str, Any]) -> bool:
        # Agents can run in parallel — each has isolated context
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        prompt = input["prompt"]
        description = input.get("description", "sub-agent task")
        agent_type = input.get("agent_type", "general-purpose")
        model_override = input.get("model")
        run_background = input.get("background", False)

        # Resolve agent definition
        agent_def = self._agent_registry.get(agent_type)
        if not agent_def:
            return ToolResult(output=f"Unknown agent type: {agent_type}", is_error=True)

        # Resolve model
        model = self._resolve_model(model_override, agent_def)

        # Filter tools for sub-agent
        sub_registry = self._build_sub_registry(agent_def)

        # Build sub-agent engine with isolated context
        is_bg = run_background or agent_def.background
        sub_engine = self._create_sub_engine(
            agent_def=agent_def,
            model=model,
            registry=sub_registry,
            context=context,
            is_background=is_bg,
        )

        if is_bg:
            return await self._run_background(sub_engine, prompt, description, context)
        else:
            return await self._run_foreground(sub_engine, prompt, description)

    async def _run_foreground(self, engine: QueryEngine, prompt: str, description: str) -> ToolResult:
        """Run sub-agent synchronously and return its output."""
        text_parts: list[str] = []
        tool_use_count = 0

        try:
            async for event in engine.submit_message(prompt):
                if event.type == "text_delta":
                    text_parts.append(event.data.get("content", ""))
                elif event.type == "tool_use_end":
                    tool_use_count += 1
                elif event.type == "error":
                    return ToolResult(
                        output=f"Agent error: {event.data.get('message', 'unknown')}",
                        is_error=True,
                    )
        except Exception as e:
            return ToolResult(output=f"Agent execution error: {e}", is_error=True)

        result_text = "".join(text_parts).strip()
        if not result_text:
            # Fallback: get text from last assistant message
            for msg in reversed(engine.messages):
                if msg.role.value == "assistant":
                    result_text = msg.get_text()
                    break

        return ToolResult(
            output=result_text or "(no output)",
            metadata={
                "agent_type": engine.custom_system_prompt is not None,
                "tool_use_count": tool_use_count,
                "turn_count": engine.turn_count,
                "usage": engine.total_usage.model_dump() if engine.total_usage else {},
            },
        )

    async def _run_background(
        self, engine: QueryEngine, prompt: str, description: str, context: ToolContext,
    ) -> ToolResult:
        """Run sub-agent in background with notification on completion.

        Stores result in _agent_results for TaskOutput retrieval.
        Enqueues task-notification XML for coordinator queue drain.
        """
        agent_id = engine.session_id
        notification_queue = context.notification_queue
        start_time = asyncio.get_event_loop().time()

        async def _run():
            text_parts: list[str] = []
            tool_use_count = 0
            status = "completed"
            try:
                async for event in engine.submit_message(prompt):
                    if event.type == "text_delta":
                        text_parts.append(event.data.get("content", ""))
                    elif event.type == "tool_use_end":
                        tool_use_count += 1
            except Exception as e:
                logger.exception("Background agent %s failed", agent_id)
                status = "failed"
                text_parts.append(f"Error: {e}")

            # Get final text
            result_text = "".join(text_parts).strip()
            if not result_text:
                for msg in reversed(engine.messages):
                    if msg.role.value == "assistant":
                        result_text = msg.get_text()
                        break

            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Store result for TaskOutput retrieval
            self._agent_results[agent_id] = {
                "status": status,
                "result": result_text or "(no output)",
                "description": description,
                "total_tokens": engine.total_usage.total_tokens if engine.total_usage else 0,
                "tool_uses": tool_use_count,
                "duration_ms": elapsed_ms,
            }

            # Enqueue task notification for coordinator/parent
            if notification_queue:
                from app.services.coordinator import format_task_notification
                notification = format_task_notification(
                    agent_id=agent_id,
                    status=status,
                    result=result_text or "(no output)",
                    total_tokens=engine.total_usage.total_tokens if engine.total_usage else 0,
                    tool_uses=tool_use_count,
                    duration_ms=elapsed_ms,
                )
                await notification_queue.enqueue(notification)
            else:
                logger.debug("Background agent %s completed but no notification queue", agent_id)

        task = asyncio.create_task(_run())
        self._background_tasks[agent_id] = task
        self._agent_engines[agent_id] = engine

        return ToolResult(
            output=f"Agent launched in background (id: {agent_id}, description: {description})",
            metadata={"agent_id": agent_id, "status": "async_launched", "description": description},
        )

    def _resolve_model(self, override: str | None, agent_def: AgentDefinition) -> str:
        """Resolve model: override > agent definition > settings default."""
        if override:
            return self._alias_to_model(override)
        if agent_def.model:
            return self._alias_to_model(agent_def.model)
        if self._settings.llm_provider == "ollama":
            return self._settings.ollama_model
        if self._settings.llm_provider == "koboldcpp":
            return self._settings.koboldcpp_model
        return self._settings.default_model

    def _alias_to_model(self, alias: str) -> str:
        if self._settings.llm_provider in ("ollama", "koboldcpp"):
            model = (
                self._settings.ollama_model
                if self._settings.llm_provider == "ollama"
                else self._settings.koboldcpp_model
            )
            aliases = {
                "smart": model,
                "fast": model,
                "reasoning": model,
                "inherit": model,
            }
        else:
            aliases = {
                "smart": self._settings.default_model,
                "fast": self._settings.fast_model,
                "reasoning": self._settings.reasoning_model,
                "inherit": self._settings.default_model,
            }
        return aliases.get(alias, alias)

    def _build_sub_registry(self, agent_def: AgentDefinition) -> ToolRegistry:
        """Build a filtered tool registry for the sub-agent.

        Applies:
        1. Global sub-agent restrictions (Agent, SendMessage blocked by default)
        2. Agent definition's disallowed_tools
        3. Agent definition's explicit tool allowlist (if not wildcard)
        """
        registry = ToolRegistry()

        all_tools = self._tool_registry.all()
        disallowed = set(agent_def.disallowed_tools or [])
        # Sub-agents cannot spawn other agents or send messages (coordinator-only)
        disallowed.add("Agent")
        disallowed.add("SendMessage")

        for tool in all_tools:
            if tool.name in disallowed:
                continue
            if agent_def.has_wildcard_tools():
                registry.register(tool)
            elif agent_def.tools and tool.name in agent_def.tools:
                registry.register(tool)

        return registry

    def _create_sub_engine(
        self,
        agent_def: AgentDefinition,
        model: str,
        registry: ToolRegistry,
        context: ToolContext,
        is_background: bool = False,
    ) -> QueryEngine:
        """Create a QueryEngine for the sub-agent with isolated context.

        Isolation:
        - Cloned file_state_cache (sub-agent reads don't affect parent)
        - Fresh abort event (sub-agent abort doesn't kill parent)
        - Shares notification_queue (so parent receives task notifications)
        """
        from uuid import uuid4

        from app.utils.file_state import FileStateCache

        adapter = self._adapter

        # Permission mode: agent definition > parent context
        perm_mode = agent_def.permission_mode or context.permission_mode
        # Read-only agents get plan mode (blocks mutations)
        if agent_def.read_only:
            perm_mode = "plan"

        permissions = PermissionManager(mode=perm_mode)

        # Clone file state cache from parent (isolation)
        cloned_cache = FileStateCache()
        if context.file_state_cache:
            for path, state in context.file_state_cache._cache.items():
                cloned_cache._cache[path] = state

        child_id = f"agent-{uuid4().hex[:8]}"
        engine = QueryEngine(
            adapter=adapter,
            registry=registry,
            permissions=permissions,
            session_id=child_id,
            model=model,
            fast_model=(
                self._settings.ollama_model if self._settings.llm_provider == "ollama"
                else self._settings.koboldcpp_model if self._settings.llm_provider == "koboldcpp"
                else self._settings.fast_model
            ),
            max_tokens=self._settings.max_tokens,
            max_turns=agent_def.max_turns,
            cwd=context.cwd,
            custom_system_prompt=agent_def.system_prompt,
            openai_client=self._adapter,
            data_dir=getattr(self._settings, 'data_dir', '~/.agent'),
        )

        # Replace with cloned cache
        engine.file_state_cache = cloned_cache

        # Share notification queue so background completions reach parent
        if context.notification_queue:
            engine.notification_queue = context.notification_queue

        # Propagate query tracking (increment depth)
        from app.services.query import QueryTracking
        engine._query_tracking = QueryTracking(chain_id=child_id, depth=is_background and 1 or 0)

        return engine

    def get_background_task(self, agent_id: str) -> asyncio.Task | None:
        return self._background_tasks.get(agent_id)

    def get_agent_result(self, agent_id: str) -> dict[str, Any] | None:
        """Get stored result from a completed background agent."""
        return self._agent_results.get(agent_id)

    def get_agent_engine(self, agent_id: str) -> Any | None:
        """Get the QueryEngine of a background agent (for SendMessage continuation)."""
        return self._agent_engines.get(agent_id)
