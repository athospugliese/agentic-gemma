"""Plan mode tools - enter/exit read-only planning mode."""

from __future__ import annotations

from typing import Any

from app.tools.base import Tool, ToolContext, ToolResult


class EnterPlanModeTool(Tool):
    name = "EnterPlanMode"
    description = "Switch to plan mode (read-only). Use this before designing an implementation strategy."
    parameters = {
        "type": "object",
        "properties": {},
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        # Store current mode for restoration
        context._pre_plan_mode = context.permission_mode  # type: ignore[attr-defined]

        # The actual mode switch happens in the query engine via permission manager
        return ToolResult(
            output=(
                "Plan mode activated. You are now in READ-ONLY mode.\n\n"
                "In plan mode you should:\n"
                "1. Explore the codebase to understand existing patterns\n"
                "2. Identify similar features as reference\n"
                "3. Consider multiple approaches and their trade-offs\n"
                "4. Design a concrete implementation strategy\n"
                "5. When ready, use ExitPlanMode to present your plan"
            ),
            metadata={"action": "enter_plan_mode", "previous_mode": context.permission_mode},
        )


class ExitPlanModeTool(Tool):
    name = "ExitPlanMode"
    description = "Exit plan mode and present your implementation plan for approval."
    parameters = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The implementation plan (markdown).",
            },
        },
        "required": ["plan"],
    }

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        plan = input["plan"]

        return ToolResult(
            output=f"Plan submitted for review:\n\n{plan}",
            metadata={"action": "exit_plan_mode", "plan_length": len(plan)},
        )
