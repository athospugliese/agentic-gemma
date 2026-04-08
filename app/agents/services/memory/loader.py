"""Memory loader - builds memory prompt from MEMORY.md and individual memory files.

Memory directory structure:
  ~/.agent/projects/<project>/memory/
    MEMORY.md           - Index file (max 200 lines)
    user_role.md        - Individual memory files
    feedback_testing.md
    project_deadline.md
    ...

Each memory file has YAML frontmatter:
---
name: memory name
description: one-line description
type: user|feedback|project|reference
---
Content here...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000

MEMORY_TYPES_SECTION = """## Types of memory

- **user**: Information about the user's role, goals, preferences, and knowledge.
- **feedback**: Guidance on what approaches to avoid or repeat, with reasoning.
- **project**: Ongoing work context - goals, deadlines, decisions, bugs.
- **reference**: Pointers to external systems (Linear, Slack, Grafana, etc.)."""

SAVE_INSTRUCTIONS = """## How to save memories

**Step 1**: Write the memory to its own file with frontmatter:
```markdown
---
name: {{memory name}}
description: {{one-line description}}
type: {{user|feedback|project|reference}}
---
{{memory content}}
```

**Step 2**: Add a pointer to MEMORY.md (one line, under 150 chars):
`- [Title](file.md) — one-line hook`"""

WHEN_TO_ACCESS = """## When to access memories
- When memories seem relevant to the current task
- When the user explicitly asks you to check, recall, or remember
- Memory records can become stale - verify against current state before acting"""

VERIFICATION = """## Before recommending from memory
- If the memory names a file path: check the file exists
- If the memory names a function: grep for it
- Memory is a snapshot in time - if it conflicts with current state, trust what you observe now"""


def get_memory_dir(project_dir: str, data_dir: str = "~/.agent") -> Path:
    """Get the memory directory for a project."""
    # Sanitize project path for use as directory name
    sanitized = project_dir.replace("/", "_").replace("\\", "_").strip("_")
    return Path(data_dir).expanduser() / "projects" / sanitized / "memory"


def load_memory_prompt(project_dir: str, data_dir: str = "~/.agent") -> str | None:
    """Build the memory system prompt section.

    Returns None if memory is disabled or directory doesn't exist.
    """
    memory_dir = get_memory_dir(project_dir, data_dir)

    if not memory_dir.exists():
        return None

    sections: list[str] = []

    # Header
    sections.append(f"# Memory\n\nMemory directory: `{memory_dir}`")

    # Save instructions
    sections.append(SAVE_INSTRUCTIONS)

    # Types
    sections.append(MEMORY_TYPES_SECTION)

    # When to access
    sections.append(WHEN_TO_ACCESS)

    # Verification
    sections.append(VERIFICATION)

    # MEMORY.md index content
    index_path = memory_dir / "MEMORY.md"
    if index_path.exists():
        try:
            content = index_path.read_text()
            content = _truncate_index(content)
            sections.append(f"## Current MEMORY.md\n\n{content}")
        except OSError:
            pass

    return "\n\n".join(sections)


def _truncate_index(content: str) -> str:
    """Truncate MEMORY.md to max lines/bytes with warning."""
    lines = content.splitlines()
    truncated_by_lines = False
    truncated_by_bytes = False

    if len(lines) > MAX_INDEX_LINES:
        lines = lines[:MAX_INDEX_LINES]
        truncated_by_lines = True

    result = "\n".join(lines)

    if len(result.encode()) > MAX_INDEX_BYTES:
        # Find last newline before byte limit
        encoded = result.encode()[:MAX_INDEX_BYTES]
        last_newline = encoded.rfind(b"\n")
        if last_newline > 0:
            result = encoded[:last_newline].decode(errors="replace")
        truncated_by_bytes = True

    if truncated_by_lines or truncated_by_bytes:
        cap = "lines" if truncated_by_lines else "bytes"
        result += f"\n\n⚠️ MEMORY.md truncated (exceeded {cap} limit). Keep the index concise."

    return result


def scan_memory_files(memory_dir: Path, max_files: int = 200) -> list[dict[str, Any]]:
    """Scan memory directory for individual memory files.

    Returns list of {filename, description, file_path, mtime_ms} sorted newest-first.
    """
    if not memory_dir.is_dir():
        return []

    files: list[dict[str, Any]] = []

    for path in memory_dir.iterdir():
        if path.name == "MEMORY.md" or not path.suffix == ".md":
            continue
        if not path.is_file():
            continue

        try:
            stat = path.stat()
            # Quick frontmatter extraction (no full parse)
            text = path.read_text(errors="replace")
            name, description = _extract_frontmatter_fields(text)

            files.append({
                "filename": path.name,
                "name": name or path.stem,
                "description": description or "",
                "file_path": str(path),
                "mtime_ms": stat.st_mtime * 1000,
            })
        except OSError:
            continue

    # Sort newest first
    files.sort(key=lambda f: f["mtime_ms"], reverse=True)
    return files[:max_files]


def _extract_frontmatter_fields(text: str) -> tuple[str | None, str | None]:
    """Quick extraction of name and description from YAML frontmatter."""
    if not text.startswith("---"):
        return None, None

    end = text.find("---", 3)
    if end == -1:
        return None, None

    frontmatter = text[3:end]
    name = None
    description = None

    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line[5:].strip().strip("'\"")
        elif line.startswith("description:"):
            description = line[12:].strip().strip("'\"")

    return name, description


def read_memory_file(path: str, max_lines: int = 500, max_bytes: int = 10_000) -> str:
    """Read a memory file with truncation limits."""
    try:
        content = Path(path).read_text(errors="replace")
        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            content += f"\n\n... ({len(lines) - max_lines} more lines)"
        if len(content.encode()) > max_bytes:
            content = content[:max_bytes] + "\n\n... (truncated)"
        return content
    except OSError:
        return ""
