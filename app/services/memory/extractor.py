"""Background memory extraction - automatically extracts memories after each turn.

After the main query loop completes a turn, a background task analyzes
recent messages and extracts key facts into memory files.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.models import Message, MessageRole
from app.utils.side_query import side_query

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are a memory extraction agent. Your job is to analyze a conversation \
and identify information worth persisting as memories for future conversations.

You have access to a memory directory where you can write files. Each memory file should have \
YAML frontmatter with name, description, and type fields.

## What to extract
- User preferences, role, goals, knowledge (type: user)
- Corrections or confirmations of approaches (type: feedback)
- Project context: deadlines, architecture decisions, ongoing work (type: project)
- Pointers to external systems (type: reference)

## What NOT to extract
- Code patterns derivable from reading the code
- Git history (use git log)
- Debugging solutions (the fix is in the code)
- Ephemeral task details

## Output format
For each memory to save, output a fenced code block with the filename and content:

```memory:filename.md
---
name: Memory Name
description: One-line description
type: user|feedback|project|reference
---
Content here...
```

If there are no memories worth extracting, respond with: NO_MEMORIES_TO_EXTRACT"""

EXTRACTION_USER_TEMPLATE = """Here are the recent conversation messages to analyze:

{conversation}

Identify any information worth persisting as memories. Remember:
- Only extract non-obvious, durable information
- Skip anything derivable from code or git history
- Check if a similar memory might already exist (existing memories listed below)

{existing_memories}"""


async def extract_memories(
    messages: list[Message],
    memory_dir: Path,
    client: Any,  # AsyncOpenAI
    model: str,
    last_cursor_uuid: str | None = None,
) -> list[str]:
    """Extract memories from recent messages and write to disk.

    Returns list of written file paths.
    """
    # Count new messages since last extraction
    new_messages = _get_new_messages(messages, last_cursor_uuid)
    if len(new_messages) < 2:  # Need at least a user + assistant pair
        return []

    # Check if main agent already wrote memories this turn
    if _has_memory_writes(new_messages, memory_dir):
        return []

    # Build conversation text
    conversation = _format_messages_for_extraction(new_messages)

    # List existing memories for dedup
    existing = _list_existing_memories(memory_dir)
    existing_text = f"Existing memories:\n{existing}" if existing else "No existing memories."

    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        conversation=conversation,
        existing_memories=existing_text,
    )

    # Call LLM
    try:
        response = await side_query(
            client=client,
            model=model,
            system=EXTRACTION_SYSTEM,
            user=user_prompt,
            max_tokens=2048,
        )
    except Exception:
        logger.warning("Memory extraction failed", exc_info=True)
        return []

    if "NO_MEMORIES_TO_EXTRACT" in response:
        return []

    # Parse and write memories
    written = _parse_and_write_memories(response, memory_dir)
    if written:
        _update_index(memory_dir, written)
        logger.info("Extracted %d memories: %s", len(written), written)

    return written


def _get_new_messages(messages: list[Message], cursor_uuid: str | None) -> list[Message]:
    """Get messages since the last extraction cursor."""
    if cursor_uuid is None:
        return messages

    found = False
    result: list[Message] = []
    for msg in messages:
        if found:
            result.append(msg)
        elif msg.uuid == cursor_uuid:
            found = True

    return result if found else messages


def _has_memory_writes(messages: list[Message], memory_dir: Path) -> bool:
    """Check if any messages contain writes to the memory directory."""
    memory_str = str(memory_dir)
    for msg in messages:
        if msg.role == MessageRole.TOOL:
            text = msg.get_text()
            if text and memory_str in text and ("File written" in text or "Edited" in text):
                return True
    return False


def _format_messages_for_extraction(messages: list[Message]) -> str:
    """Format messages for the extraction prompt."""
    parts: list[str] = []
    for msg in messages:
        if msg.role in (MessageRole.SYSTEM, MessageRole.TOOL):
            continue
        text = msg.get_text()
        if text:
            role = msg.role.value
            # Truncate long messages
            if len(text) > 2000:
                text = text[:2000] + "..."
            parts.append(f"[{role}]: {text}")
    return "\n\n".join(parts)


def _list_existing_memories(memory_dir: Path) -> str:
    """List existing memory filenames for dedup."""
    if not memory_dir.is_dir():
        return ""

    files = [f.name for f in memory_dir.iterdir() if f.suffix == ".md" and f.name != "MEMORY.md"]
    return "\n".join(f"- {f}" for f in sorted(files))


def _parse_and_write_memories(response: str, memory_dir: Path) -> list[str]:
    """Parse memory blocks from response and write to disk."""
    import re

    memory_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Find all ```memory:filename.md ... ``` blocks
    pattern = r"```memory:(\S+\.md)\s*\n(.*?)```"
    for match in re.finditer(pattern, response, re.DOTALL):
        filename = match.group(1)
        content = match.group(2).strip()

        if not content or not filename:
            continue

        # Sanitize filename
        filename = filename.replace("/", "_").replace("\\", "_")
        file_path = memory_dir / filename

        try:
            file_path.write_text(content)
            written.append(filename)
        except OSError:
            logger.warning("Failed to write memory file: %s", filename)

    return written


def _update_index(memory_dir: Path, new_files: list[str]) -> None:
    """Append new memory entries to MEMORY.md index."""
    index_path = memory_dir / "MEMORY.md"

    # Read existing index
    existing = ""
    if index_path.exists():
        try:
            existing = index_path.read_text()
        except OSError:
            pass

    # Append new entries
    new_lines: list[str] = []
    for filename in new_files:
        file_path = memory_dir / filename
        if not file_path.exists():
            continue

        text = file_path.read_text(errors="replace")
        from app.services.memory.loader import _extract_frontmatter_fields
        name, description = _extract_frontmatter_fields(text)
        title = name or filename.replace(".md", "").replace("_", " ").title()
        hook = description or ""
        new_lines.append(f"- [{title}]({filename}) — {hook}")

    if new_lines:
        separator = "\n" if existing and not existing.endswith("\n") else ""
        updated = existing + separator + "\n".join(new_lines) + "\n"
        try:
            index_path.write_text(updated)
        except OSError:
            logger.warning("Failed to update MEMORY.md index")
