"""Memory ranker - selects relevant memories per turn using a fast model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.services.memory.loader import read_memory_file, scan_memory_files
from app.utils.side_query import side_query_json

logger = logging.getLogger(__name__)

SELECT_MEMORIES_SYSTEM = """You are selecting memories that will be useful to an AI assistant as it processes a user's query. \
You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected_memories" array of filenames for memories that will clearly be useful (up to 5). \
Only include memories you are certain will be helpful based on their name and description.
- If unsure, do not include the memory.
- If no memories are clearly useful, return an empty array.
- If a list of recently-used tools is provided, do not select API docs for those tools \
(but DO select warnings/gotchas about those tools)."""

MAX_RELEVANT_MEMORIES = 5


async def find_relevant_memories(
    query: str,
    memory_dir: Path,
    client: Any,  # AsyncOpenAI
    model: str,
    recent_tools: list[str] | None = None,
    already_surfaced: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Find memories relevant to the current user query.

    Uses a fast model (gpt-4o-mini) as a ranker to select up to 5 memories
    from the memory directory based on filename and description.

    Returns list of {path, mtime_ms, content} for selected memories.
    """
    memories = scan_memory_files(memory_dir)
    if not memories:
        return []

    # Filter already surfaced
    if already_surfaced:
        memories = [m for m in memories if m["file_path"] not in already_surfaced]

    if not memories:
        return []

    # Build manifest for ranker
    manifest_lines = [f"{m['filename']}: {m['description']}" for m in memories if m["description"]]
    if not manifest_lines:
        # No descriptions - return first few memories as fallback
        return [
            {"path": m["file_path"], "mtime_ms": m["mtime_ms"], "content": read_memory_file(m["file_path"])}
            for m in memories[:3]
        ]

    manifest = "\n".join(manifest_lines)

    # Build user prompt
    user_parts = [f"Query: {query}", f"\nAvailable memories:\n{manifest}"]
    if recent_tools:
        user_parts.append(f"\nRecently used tools: {', '.join(recent_tools)}")

    user_prompt = "\n".join(user_parts)

    # Call ranker
    try:
        result = await side_query_json(
            client=client,
            model=model,
            system=SELECT_MEMORIES_SYSTEM,
            user=user_prompt,
            max_tokens=256,
        )
    except Exception:
        logger.warning("Memory ranker failed, returning empty", exc_info=True)
        return []

    selected_filenames = result.get("selected_memories", [])
    if not isinstance(selected_filenames, list):
        return []

    # Map filenames back to full info
    filename_map = {m["filename"]: m for m in memories}
    valid = [filename_map[fn] for fn in selected_filenames if fn in filename_map]

    # Read content of selected memories
    results = []
    for mem in valid[:MAX_RELEVANT_MEMORIES]:
        content = read_memory_file(mem["file_path"])
        if content:
            results.append({
                "path": mem["file_path"],
                "mtime_ms": mem["mtime_ms"],
                "content": content,
            })

    return results


def format_memories_as_attachment(memories: list[dict[str, Any]]) -> str | None:
    """Format selected memories as a system reminder to inject into the user message."""
    if not memories:
        return None

    parts = ["<relevant-memories>"]
    for mem in memories:
        filename = Path(mem["path"]).name
        parts.append(f"\n### {filename}\n{mem['content']}")
    parts.append("\n</relevant-memories>")

    return "\n".join(parts)
