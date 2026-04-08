"""Git integration - captures git context for system prompt."""

from __future__ import annotations

import asyncio
from pathlib import Path


async def _run_git(args: list[str], cwd: str) -> str:
    """Run a git command and return stdout (empty string on failure)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode(errors="replace").strip() if stdout else ""
    except Exception:
        return ""


async def get_git_context(cwd: str) -> str | None:
    """Capture git branch, status, and recent commits for system prompt injection.

    Returns None if not in a git repo.
    """
    # Check if we're in a git repo
    git_dir = await _run_git(["rev-parse", "--git-dir"], cwd)
    if not git_dir:
        return None

    # Run all git commands in parallel
    branch_coro = _run_git(["branch", "--show-current"], cwd)
    main_branch_coro = _run_git(["symbolic-ref", "refs/remotes/origin/HEAD", "--short"], cwd)
    status_coro = _run_git(["--no-optional-locks", "status", "--short"], cwd)
    log_coro = _run_git(["--no-optional-locks", "log", "--oneline", "-n", "5"], cwd)
    user_coro = _run_git(["config", "user.name"], cwd)

    branch, main_branch, status, log, user_name = await asyncio.gather(
        branch_coro, main_branch_coro, status_coro, log_coro, user_coro
    )

    # Clean up main branch (remove origin/ prefix)
    if main_branch:
        main_branch = main_branch.replace("origin/", "")

    parts = []
    if branch:
        parts.append(f"Current branch: {branch}")
    if main_branch:
        parts.append(f"Main branch: {main_branch}")
    if user_name:
        parts.append(f"Git user: {user_name}")
    if status:
        # Truncate status to 2000 chars
        if len(status) > 2000:
            status = status[:2000] + "\n... (truncated)"
        parts.append(f"Status:\n{status}")
    if log:
        parts.append(f"Recent commits:\n{log}")

    return "\n".join(parts) if parts else None
