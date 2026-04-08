"""Hook execution engine - runs shell hooks at tool lifecycle points.

Hooks are defined in settings as:
  hooks:
    PreToolUse:
      - command: "echo 'about to run tool'"
        matcher: "Bash"
        timeout: 30
    PostToolUse:
      - command: "notify-send 'tool done'"
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HOOK_TIMEOUT = 30  # seconds


@dataclass
class HookResult:
    """Result from a hook execution."""
    success: bool
    output: str = ""
    blocked: bool = False  # Hook explicitly blocked the operation
    reason: str = ""


async def execute_hooks(
    event: str,
    hooks_config: dict[str, list[Any]],
    context: dict[str, Any] | None = None,
    cwd: str = ".",
) -> list[HookResult]:
    """Execute all hooks matching an event.

    Args:
        event: Hook event name (e.g., "PreToolUse", "PostToolUse")
        hooks_config: The hooks dict from Settings
        context: Context dict passed to hook as env vars / stdin
        cwd: Working directory for hook execution

    Returns:
        List of HookResult for each executed hook.
    """
    hook_entries = hooks_config.get(event, [])
    if not hook_entries:
        return []

    results: list[HookResult] = []
    for entry in hook_entries:
        # entry is a HookEntry or dict with command, matcher, timeout
        if hasattr(entry, 'command'):
            command = entry.command
            matcher = entry.matcher
            timeout = entry.timeout or DEFAULT_HOOK_TIMEOUT
        else:
            command = entry.get("command", "")
            matcher = entry.get("matcher")
            timeout = entry.get("timeout", DEFAULT_HOOK_TIMEOUT)

        if not command:
            continue

        # Check matcher against context
        if matcher and context:
            tool_name = context.get("tool_name", "")
            if matcher != tool_name and not tool_name.startswith(matcher):
                continue  # Matcher doesn't match — skip this hook

        result = await _run_hook_command(command, context, timeout, cwd)
        results.append(result)

        # If a hook blocks, stop executing further hooks
        if result.blocked:
            logger.info("Hook blocked operation: %s (event=%s)", result.reason, event)
            break

    return results


async def _run_hook_command(
    command: str,
    context: dict[str, Any] | None,
    timeout: int,
    cwd: str,
) -> HookResult:
    """Run a single hook command as a subprocess."""
    import os

    env = os.environ.copy()
    # Inject hook context as env vars
    if context:
        env["HOOK_TOOL_NAME"] = str(context.get("tool_name", ""))
        env["HOOK_TOOL_INPUT"] = json.dumps(context.get("tool_input", {}))
        env["HOOK_EVENT"] = str(context.get("event", ""))
        if context.get("tool_result"):
            env["HOOK_TOOL_RESULT"] = str(context["tool_result"])[:10000]

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode(errors="replace").strip() if stdout else ""

        # Exit code 0 = success, 1 = error (non-blocking), 2 = blocked
        if proc.returncode == 2:
            return HookResult(success=False, output=output, blocked=True, reason=output or "Hook blocked operation")
        elif proc.returncode != 0:
            logger.warning("Hook failed (exit %d): %s", proc.returncode, output[:200])
            return HookResult(success=False, output=output)
        else:
            return HookResult(success=True, output=output)

    except asyncio.TimeoutError:
        logger.warning("Hook timed out after %ds: %s", timeout, command[:100])
        return HookResult(success=False, output=f"Hook timed out after {timeout}s")
    except Exception as e:
        logger.warning("Hook execution error: %s", e)
        return HookResult(success=False, output=str(e))
