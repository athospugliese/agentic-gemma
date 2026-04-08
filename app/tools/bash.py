"""Bash tool - executes shell commands."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from typing import Any

from app.tools.base import Tool, ToolContext, ToolResult

DEFAULT_TIMEOUT = 120  # seconds
MAX_OUTPUT_CHARS = 64_000

# Commands that are always read-only (no state mutation)
_READ_ONLY_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "grep", "rg", "find", "echo", "pwd", "which", "wc",
    "file", "stat", "du", "df", "env", "printenv", "whoami", "hostname", "uname",
    "date", "cal", "uptime", "free", "top", "ps", "id", "groups", "type", "command",
    "readlink", "realpath", "basename", "dirname", "diff", "comm", "sort", "uniq",
    "tr", "cut", "awk", "sed", "jq", "yq", "xargs", "tee",
    "tree", "less", "more", "strings", "hexdump", "xxd", "md5sum", "sha256sum",
    "man", "help", "info", "whatis", "apropos",
})

# Git subcommands that are read-only
_GIT_READ_ONLY_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "branch", "show", "blame", "stash list",
    "remote", "tag", "describe", "shortlog", "reflog", "rev-parse",
    "ls-files", "ls-tree", "cat-file", "rev-list", "name-rev",
    "config --get", "config --list", "config -l",
})

# Commands known to be destructive (never read-only)
_DANGEROUS_COMMANDS = frozenset({
    "rm", "rmdir", "mv", "chmod", "chown", "chgrp", "mkfs", "dd",
    "fdisk", "shutdown", "reboot", "halt", "kill", "killall", "pkill",
    "sudo", "su", "docker", "kubectl",
})

# Operators that chain commands — anything after these must also be checked
_CHAIN_OPERATORS = re.compile(r'\s*(?:&&|\|\||;)\s*')
_PIPE_OPERATOR = re.compile(r'\s*\|\s*')


class BashTool(Tool):
    name = "Bash"
    description = "Execute a bash command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (max 600).",
            },
        },
        "required": ["command"],
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        cmd = input.get("command", "").strip()
        return _is_command_read_only(cmd)

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        command = input["command"]
        timeout = min(input.get("timeout", DEFAULT_TIMEOUT), 600)

        # Build subprocess environment
        env = _build_env(context)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                cwd=context.cwd,
                env=env,
            )

            # Check abort signal during execution if available
            abort_signal = context.abort_signal
            if abort_signal is not None:
                # Race between command completion and abort
                comm_task = asyncio.create_task(proc.communicate())
                abort_task = asyncio.create_task(abort_signal.wait())

                done, pending = await asyncio.wait(
                    {comm_task, abort_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for p in pending:
                    p.cancel()

                if abort_task in done:
                    # Abort received — kill process
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return ToolResult(output="Command aborted by user", is_error=True)

                if comm_task in done:
                    stdout, _ = comm_task.result()
                else:
                    # Timeout
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return ToolResult(output=f"Command timed out after {timeout}s", is_error=True)
            else:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            output = stdout.decode(errors="replace").strip() if stdout else ""

            if not output:
                output = f"(exit code {proc.returncode})"

            # Truncate very large outputs
            if len(output) > MAX_OUTPUT_CHARS:
                truncated = output[:MAX_OUTPUT_CHARS]
                remaining = len(output) - MAX_OUTPUT_CHARS
                output = f"{truncated}\n\n... ({remaining} more characters truncated)"

            is_error = proc.returncode != 0
            if is_error:
                output = f"Exit code {proc.returncode}\n{output}"

            return ToolResult(output=output, is_error=is_error)

        except asyncio.TimeoutError:
            # Try to kill the process
            try:
                proc.kill()
            except Exception:
                pass
            return ToolResult(output=f"Command timed out after {timeout}s", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Error executing command: {e}", is_error=True)


# ---------------------------------------------------------------------------
# Read-only command classification
# ---------------------------------------------------------------------------

def _is_command_read_only(command: str) -> bool:
    """Check if a bash command is read-only (no state mutation).

    Handles compound commands (&&, ||, ;), pipes, and redirections.
    """
    if not command:
        return False

    # Output redirection is never read-only
    if re.search(r'[^|]>|>>|\d+>', command):
        return False

    # Split on chain operators (&&, ||, ;) and check each part
    parts = _CHAIN_OPERATORS.split(command)
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # For piped commands, check each segment
        pipe_segments = _PIPE_OPERATOR.split(part)
        for segment in pipe_segments:
            segment = segment.strip()
            if not segment:
                continue
            if not _is_single_command_read_only(segment):
                return False

    return True


def _is_single_command_read_only(command: str) -> bool:
    """Check if a single command (no pipes, no chains) is read-only."""
    # Extract the base command (first word)
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Malformed command — assume not read-only
        return False

    if not tokens:
        return False

    base_cmd = os.path.basename(tokens[0])

    # Known dangerous commands — never read-only
    if base_cmd in _DANGEROUS_COMMANDS:
        return False

    # Known read-only commands
    if base_cmd in _READ_ONLY_COMMANDS:
        # Special case: some read-only commands have dangerous flags
        if base_cmd == "find" and any(f in tokens for f in ("-exec", "-execdir", "-delete")):
            return False
        if base_cmd == "xargs" and len(tokens) > 1:
            # xargs wraps another command — check if that command is read-only
            return _is_single_command_read_only(" ".join(tokens[1:]))
        if base_cmd == "tee":
            # tee writes to files — not read-only
            return False
        if base_cmd == "sed" and any(f in tokens for f in ("-i", "--in-place")):
            return False
        if base_cmd == "awk" and any(f in tokens for f in ("-i",)):
            return False
        return True

    # Git: check subcommand
    if base_cmd == "git" and len(tokens) > 1:
        git_sub = tokens[1]
        # Check compound git subcommands (e.g., "stash list", "config --get")
        full_sub = " ".join(tokens[1:3]) if len(tokens) > 2 else git_sub
        if git_sub in _GIT_READ_ONLY_SUBCOMMANDS or full_sub in _GIT_READ_ONLY_SUBCOMMANDS:
            return True
        return False

    # Test runners
    if base_cmd in ("pytest", "python", "node", "npm", "npx", "cargo", "go", "make"):
        if base_cmd == "npm" and len(tokens) > 1 and tokens[1] in ("test", "run"):
            return True
        if base_cmd == "python" and len(tokens) > 1 and tokens[1] in ("-c", "-m"):
            if tokens[1] == "-m" and len(tokens) > 2 and tokens[2] == "pytest":
                return True
            if tokens[1] == "-c":
                return True
        if base_cmd == "cargo" and len(tokens) > 1 and tokens[1] == "test":
            return True
        if base_cmd == "go" and len(tokens) > 1 and tokens[1] == "test":
            return True
        return False

    # Unknown command — assume not read-only
    return False


def _build_env(context: ToolContext) -> dict[str, str]:
    """Build environment for subprocess with useful defaults."""
    env = os.environ.copy()
    env["AGENT_SESSION_ID"] = context.session_id
    env["GIT_EDITOR"] = "true"  # Prevent git from opening editor
    if context.agent_id:
        env["AGENT_ID"] = context.agent_id
    return env
