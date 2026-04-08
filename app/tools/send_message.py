"""SendMessage tool - sends messages to continue existing agents."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.tools.base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class SendMessageTool(Tool):
    """Send a follow-up message to a running or completed background agent.

    Used by the coordinator to:
    - Continue workers with synthesized instructions
    - Ask workers for clarification
    - Send shutdown requests

    Unlike respawning, this preserves the agent's full conversation context.
    """

    name = "SendMessage"
    description = (
        "Send a message to an existing background agent to continue its work. "
        "Use the agent's ID (returned when it was launched) as the 'to' field."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Agent ID or name to send the message to.",
            },
            "message": {
                "type": "string",
                "description": "The message content to send to the agent.",
            },
            "summary": {
                "type": "string",
                "description": "Short (5-10 word) preview of the message.",
            },
        },
        "required": ["to", "message"],
    }

    def __init__(self, agent_tool: Any = None):
        """Initialize with reference to AgentTool for background task lookup."""
        self._agent_tool = agent_tool

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        to = input["to"]
        message = input["message"]

        if not self._agent_tool:
            return ToolResult(output="SendMessage not available (no agent tool configured)", is_error=True)

        # Find the background task
        task = self._agent_tool.get_background_task(to)
        if task is None:
            return ToolResult(
                output=f"Agent '{to}' not found. Available agents: {list(self._agent_tool._background_tasks.keys())}",
                is_error=True,
            )

        # Get the agent's engine (preserved reference)
        engine = self._agent_tool.get_agent_engine(to)
        if engine is None:
            # Fallback: respawn if engine was not stored (shouldn't happen)
            logger.warning("No engine found for agent %s, falling back to respawn", to)
            return await self._respawn_agent(to, message, context)

        if task.done():
            # Agent completed — continue with its full context by submitting a new message
            return await self._continue_completed_agent(to, engine, message, context)
        else:
            # Agent is running — inject message via notification queue for next drain
            return await self._inject_into_running_agent(to, engine, message)

    async def _continue_completed_agent(
        self, agent_id: str, engine: Any, message: str, context: ToolContext,
    ) -> ToolResult:
        """Continue a completed agent by submitting a new message to its engine.

        The engine retains full conversation history, file state, and context.
        """
        notification_queue = context.notification_queue
        start_time = asyncio.get_event_loop().time()

        async def _run():
            text_parts: list[str] = []
            tool_use_count = 0
            status = "completed"
            try:
                async for event in engine.submit_message(message):
                    if event.type == "text_delta":
                        text_parts.append(event.data.get("content", ""))
                    elif event.type == "tool_use_end":
                        tool_use_count += 1
            except Exception as e:
                logger.exception("Continued agent %s failed", agent_id)
                status = "failed"
                text_parts.append(f"Error: {e}")

            result_text = "".join(text_parts).strip()
            if not result_text:
                for msg in reversed(engine.messages):
                    if msg.role.value == "assistant":
                        result_text = msg.get_text()
                        break

            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Update stored result
            self._agent_tool._agent_results[agent_id] = {
                "status": status,
                "result": result_text or "(no output)",
                "description": f"continuation of {agent_id}",
                "total_tokens": engine.total_usage.total_tokens if engine.total_usage else 0,
                "tool_uses": tool_use_count,
                "duration_ms": elapsed_ms,
            }

            # Notify coordinator
            if notification_queue:
                from app.agents.services.coordinator import format_task_notification
                notification = format_task_notification(
                    agent_id=agent_id,
                    status=status,
                    result=result_text or "(no output)",
                    total_tokens=engine.total_usage.total_tokens if engine.total_usage else 0,
                    tool_uses=tool_use_count,
                    duration_ms=elapsed_ms,
                )
                await notification_queue.enqueue(notification)

        # Run continuation in background
        new_task = asyncio.create_task(_run())
        self._agent_tool._background_tasks[agent_id] = new_task

        return ToolResult(
            output=f"Message sent to agent '{agent_id}' (continuing with full context, {len(engine.messages)} messages preserved)",
            metadata={"agent_id": agent_id, "action": "continued", "messages_preserved": len(engine.messages)},
        )

    async def _inject_into_running_agent(self, agent_id: str, engine: Any, message: str) -> ToolResult:
        """Inject a message into a running agent via its notification queue.

        The message will be drained at the start of the agent's next turn.
        """
        wrapped = f"<coordinator-message>\n{message}\n</coordinator-message>"
        await engine.notification_queue.enqueue(wrapped)
        logger.info("Injected message into running agent %s", agent_id)

        return ToolResult(
            output=f"Message injected into running agent '{agent_id}' (will be processed at next turn)",
            metadata={"agent_id": agent_id, "action": "injected"},
        )

    async def _respawn_agent(self, agent_id: str, message: str, context: ToolContext) -> ToolResult:
        """Fallback: respawn an agent with a continuation message (loses context)."""
        if not self._agent_tool:
            return ToolResult(output="Cannot respawn: no agent tool", is_error=True)

        continuation_prompt = (
            f"You are continuing work from a previous agent (id: {agent_id}). "
            f"Here is the follow-up instruction:\n\n{message}"
        )

        result = await self._agent_tool.call(
            {
                "prompt": continuation_prompt,
                "description": f"Continue {agent_id}",
                "agent_type": "general-purpose",
                "background": True,
            },
            context,
        )

        if result.is_error:
            return result

        return ToolResult(
            output=f"Message sent to agent '{agent_id}' (respawned as new agent — context not preserved)",
            metadata={"original_agent": agent_id, "action": "respawned"},
        )
