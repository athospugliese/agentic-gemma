"""Permission system for tool execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str = ""


class PermissionManager:
    """Manages tool execution permissions.

    Modes:
        - bypass: allow everything
        - auto: allow read-only tools + known-safe commands, ask for the rest
        - plan: only read-only tools allowed
        - default: ask for everything not explicitly allowed
    """

    def __init__(self, mode: str = "auto", allow_rules: list[str] | None = None, deny_rules: list[str] | None = None):
        self.mode = mode
        self.allow_rules = set(allow_rules or [])
        self.deny_rules = set(deny_rules or [])

        # Pending permission requests: tool_use_id -> asyncio.Event + decision
        self._pending: dict[str, tuple[asyncio.Event, PermissionDecision | None]] = {}

    async def check(self, tool_name: str, tool_input: dict[str, Any], is_read_only: bool) -> PermissionDecision:
        """Check if a tool call is allowed."""
        # Deny rules always win
        if tool_name in self.deny_rules:
            return PermissionDecision(allowed=False, reason=f"Tool {tool_name} is denied by policy")

        # Bypass mode: allow everything
        if self.mode == "bypass":
            return PermissionDecision(allowed=True, reason="bypass mode")

        # Plan mode: only read-only
        if self.mode == "plan":
            if is_read_only:
                return PermissionDecision(allowed=True, reason="read-only in plan mode")
            return PermissionDecision(allowed=False, reason="mutation not allowed in plan mode")

        # Explicit allow rules
        if tool_name in self.allow_rules:
            return PermissionDecision(allowed=True, reason=f"Tool {tool_name} explicitly allowed")

        # Auto mode: allow read-only
        if self.mode == "auto" and is_read_only:
            return PermissionDecision(allowed=True, reason="read-only auto-approved")

        # Auto mode: classify mutating operations for safety
        if self.mode == "auto":
            if _is_safe_mutation(tool_name, tool_input):
                return PermissionDecision(allowed=True, reason="auto mode (safe mutation)")
            # Dangerous mutation in auto mode — needs explicit permission
            return PermissionDecision(allowed=False, reason=f"Tool {tool_name} requires permission (potentially destructive)")

        # Default mode: needs explicit permission
        return PermissionDecision(allowed=False, reason=f"Tool {tool_name} requires permission")

    def create_permission_request(self, tool_use_id: str) -> asyncio.Event:
        """Create a pending permission request for interactive approval."""
        event = asyncio.Event()
        self._pending[tool_use_id] = (event, None)
        return event

    def resolve_permission(self, tool_use_id: str, allowed: bool) -> None:
        """Resolve a pending permission request (called from HTTP endpoint)."""
        if tool_use_id in self._pending:
            event, _ = self._pending[tool_use_id]
            self._pending[tool_use_id] = (event, PermissionDecision(allowed=allowed))
            event.set()

    async def request_and_wait(self, tool_use_id: str, tool_name: str, tool_input: dict[str, Any], timeout: float = 300) -> PermissionDecision:
        """Create a permission request, emit event, and wait for resolution."""
        event = self.create_permission_request(tool_use_id)
        return await self.wait_for_permission(tool_use_id, timeout)

    async def wait_for_permission(self, tool_use_id: str, timeout: float = 300) -> PermissionDecision:
        """Wait for a permission to be resolved."""
        if tool_use_id not in self._pending:
            return PermissionDecision(allowed=False, reason="No pending request")

        event, _ = self._pending[tool_use_id]
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(tool_use_id, None)
            return PermissionDecision(allowed=False, reason="Permission request timed out")

        _, decision = self._pending.pop(tool_use_id)
        return decision or PermissionDecision(allowed=False, reason="No decision")


# ---------------------------------------------------------------------------
# Auto-mode safety classification
# ---------------------------------------------------------------------------

# Bash commands considered dangerous (destructive, irreversible, or network-impacting)
_DANGEROUS_BASH_PATTERNS = (
    "rm ", "rm\t", "rmdir ",
    "git push", "git reset --hard", "git clean", "git rebase",
    "git checkout .", "git restore .", "git branch -D", "git branch -d",
    "docker run", "docker exec", "docker rm",
    "kubectl delete", "kubectl apply",
    "chmod ", "chown ", "chgrp ",
    "mkfs", "dd if=", "fdisk",
    "curl | ", "curl |", "wget -O- |", "wget -O-|",
    "eval ", "source ",
    "> /", ">/",
    "sudo ",
    "kill ", "killall ", "pkill ",
    "shutdown", "reboot", "halt",
    "pip install", "npm install", "yarn add",
    "mv ", "cp -r",
)

# Substrings in commands that indicate output redirection (destructive write)
_REDIRECT_PATTERNS = (" > ", " >> ", "\t>", " >|")

# Tools that are always safe mutations in auto mode
_SAFE_MUTATING_TOOLS = {"Write", "Edit", "TodoWrite", "EnterPlanMode", "ExitPlanMode"}


def _is_safe_mutation(tool_name: str, tool_input: dict[str, Any]) -> bool:
    """Classify whether a mutating tool call is safe for auto-approval.

    Returns True for safe mutations, False for potentially dangerous ones.
    """
    # File write/edit tools are generally safe (agent-initiated, not arbitrary commands)
    if tool_name in _SAFE_MUTATING_TOOLS:
        return True

    # Agent tool — delegates permission checks internally
    if tool_name == "Agent":
        return True

    # Bash tool — needs command-level analysis
    if tool_name == "Bash":
        command = tool_input.get("command", "").strip()
        return _is_safe_bash_command(command)

    # Unknown mutating tools — require permission
    return False


def _is_safe_bash_command(command: str) -> bool:
    """Check if a bash command is safe for auto-approval.

    This is a safety classifier, not a read-only classifier.
    Returns False for commands that could be destructive or irreversible.
    """
    if not command:
        return False

    cmd_lower = command.lower()

    # Check dangerous patterns
    for pattern in _DANGEROUS_BASH_PATTERNS:
        if pattern in cmd_lower:
            return False

    # Check for output redirection to files
    for redir in _REDIRECT_PATTERNS:
        if redir in command:
            return False

    # Check for pipe to shell execution
    if "|" in command:
        parts = command.split("|")
        for part in parts[1:]:
            stripped = part.strip().lower()
            if stripped.startswith(("sh", "bash", "zsh", "dash", "tee ")):
                return False

    # Check for compound commands with dangerous second part
    for separator in ("&&", "||", ";"):
        if separator in command:
            parts = command.split(separator)
            for part in parts[1:]:
                stripped = part.strip().lower()
                for pattern in _DANGEROUS_BASH_PATTERNS:
                    if stripped.startswith(pattern.strip()):
                        return False

    # Passed all checks — consider safe
    return True
