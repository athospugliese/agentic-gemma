"""Task management tools - stop and query background tasks."""

from __future__ import annotations

import asyncio
from typing import Any

from app.tools.base import Tool, ToolContext, ToolResult


class TaskStopTool(Tool):
    name = "TaskStop"
    description = "Stop a running background task/agent."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task/agent ID to stop.",
            },
        },
        "required": ["task_id"],
    }

    def __init__(self, agent_tool: Any = None):
        self._agent_tool = agent_tool

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = input["task_id"]

        if not self._agent_tool:
            return ToolResult(output="TaskStop unavailable (no agent tool)", is_error=True)

        task = self._agent_tool.get_background_task(task_id)
        if task is None:
            available = list(self._agent_tool._background_tasks.keys())
            return ToolResult(output=f"Task '{task_id}' not found. Available: {available}", is_error=True)

        if task.done():
            return ToolResult(output=f"Task '{task_id}' already completed.")

        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        return ToolResult(output=f"Task '{task_id}' stopped.", metadata={"task_id": task_id})


class TaskOutputTool(Tool):
    name = "TaskOutput"
    description = "Get the output/status of a background task/agent."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task/agent ID to query.",
            },
            "block": {
                "type": "boolean",
                "description": "Wait for the task to complete (default false).",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds when blocking (default 30).",
            },
        },
        "required": ["task_id"],
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    def __init__(self, agent_tool: Any = None):
        self._agent_tool = agent_tool

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = input["task_id"]
        block = input.get("block", False)
        timeout = input.get("timeout", 30)

        if not self._agent_tool:
            return ToolResult(output="TaskOutput unavailable (no agent tool)", is_error=True)

        task = self._agent_tool.get_background_task(task_id)
        if task is None:
            available = list(self._agent_tool._background_tasks.keys())
            return ToolResult(output=f"Task '{task_id}' not found. Available: {available}", is_error=True)

        if block and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                return ToolResult(output=f"Task '{task_id}' still running after {timeout}s timeout.")
            except asyncio.CancelledError:
                return ToolResult(output=f"Task '{task_id}' was cancelled.")

        if task.done():
            try:
                task.result()
                status = "completed"
            except asyncio.CancelledError:
                status = "cancelled"
            except Exception as e:
                status = f"failed: {e}"

            # Get stored agent result (includes full output text)
            agent_result = self._agent_tool.get_agent_result(task_id)
            if agent_result:
                result_text = agent_result.get("result", "(no output)")
                description = agent_result.get("description", "")
                tokens = agent_result.get("total_tokens", 0)
                tool_uses = agent_result.get("tool_uses", 0)
                duration = agent_result.get("duration_ms", 0)

                output = (
                    f"Task '{task_id}' ({description}): {status}\n"
                    f"Duration: {duration}ms | Tokens: {tokens} | Tool uses: {tool_uses}\n\n"
                    f"Result:\n{result_text}"
                )
                return ToolResult(output=output, metadata={
                    "task_id": task_id, "status": status,
                    "total_tokens": tokens, "tool_uses": tool_uses,
                })

            return ToolResult(output=f"Task '{task_id}': {status} (no result stored)", metadata={"task_id": task_id, "status": status})

        return ToolResult(output=f"Task '{task_id}' is still running.", metadata={"task_id": task_id, "status": "running"})
