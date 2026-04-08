"""TodoWrite tool - manage structured task lists."""

from __future__ import annotations

from typing import Any

from app.tools.base import Tool, ToolContext, ToolResult

# Global todo state per session
_session_todos: dict[str, list[dict[str, Any]]] = {}


class TodoWriteTool(Tool):
    name = "TodoWrite"
    description = "Create and manage a structured task list for tracking progress."
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Task description."},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "activeForm": {"type": "string", "description": "Present continuous form (e.g. 'Running tests')."},
                    },
                    "required": ["content", "status"],
                },
                "description": "The complete updated todo list.",
            },
        },
        "required": ["todos"],
    }

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        todos = input["todos"]
        session_key = context.agent_id or context.session_id

        old_todos = _session_todos.get(session_key, [])
        _session_todos[session_key] = todos

        # Count statuses
        pending = sum(1 for t in todos if t.get("status") == "pending")
        in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
        completed = sum(1 for t in todos if t.get("status") == "completed")

        return ToolResult(
            output=f"Todo list updated: {pending} pending, {in_progress} in progress, {completed} completed.",
            metadata={
                "old_count": len(old_todos),
                "new_count": len(todos),
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
            },
        )
