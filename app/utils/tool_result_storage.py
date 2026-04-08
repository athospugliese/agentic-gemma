"""Tool result storage - persists large outputs to disk."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MAX_RESULT_CHARS = 50_000
PREVIEW_SIZE = 2000


def maybe_truncate_result(
    output: str,
    tool_name: str,
    tool_use_id: str,
    session_dir: str,
    max_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> str:
    """Truncate large tool results, persisting full output to disk.

    Returns the output as-is if under threshold, or a truncated version
    with a reference to the persisted file.
    """
    if len(output) <= max_chars:
        return output

    # Persist to disk
    results_dir = Path(session_dir) / "tool-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{tool_use_id}.txt"
    result_path.write_text(output)

    # Build preview
    preview = output[:PREVIEW_SIZE]
    remaining = len(output) - PREVIEW_SIZE

    return (
        f"<large-output persisted-to=\"{result_path}\">\n"
        f"{preview}\n"
        f"... ({remaining} more characters, use Read tool on {result_path} to see full output)\n"
        f"</large-output>"
    )


# Per-tool thresholds (tools that commonly produce large output)
TOOL_THRESHOLDS: dict[str, int] = {
    "Bash": 30_000,
    "Read": 1_000_000,  # Effectively no limit for Read
    "Edit": 100_000,
    "Grep": 50_000,
    "Glob": 50_000,
    "Write": 100_000,
}


def get_threshold_for_tool(tool_name: str) -> int:
    return TOOL_THRESHOLDS.get(tool_name, DEFAULT_MAX_RESULT_CHARS)
