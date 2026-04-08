"""Session resume - discover, load, and restore sessions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.models import Message, message_from_jsonl
from app.utils.messages import detect_interrupted_conversation, normalize_messages_for_api

logger = logging.getLogger(__name__)


async def discover_sessions(
    sessions_dir: Path,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Discover persisted sessions from disk.

    Reads head/tail of JSONL files to extract metadata without full parsing.
    Returns sorted by last modified (newest first).
    """
    if not sessions_dir.is_dir():
        return []

    entries: list[dict[str, Any]] = []

    # Sort by mtime (stat-only pass)
    jsonl_files = sorted(
        sessions_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for path in jsonl_files[:limit]:
        try:
            stat = path.stat()
            session_id = path.stem
            metadata = _extract_metadata(path)

            entries.append({
                "session_id": session_id,
                "last_modified": stat.st_mtime,
                "size_bytes": stat.st_size,
                "title": metadata.get("title"),
                "last_prompt": metadata.get("last_prompt"),
                "message_count": metadata.get("message_count", 0),
                "model": metadata.get("model"),
            })
        except OSError:
            continue

    return entries


def _extract_metadata(path: Path, tail_bytes: int = 65536) -> dict[str, Any]:
    """Extract metadata from session file without full parsing.

    Reads first and last 64KB to find title, last prompt, and message count.
    """
    metadata: dict[str, Any] = {}

    try:
        size = path.stat().st_size

        with open(path, "r") as f:
            # Read head for first user message
            head_lines: list[str] = []
            for _ in range(100):
                line = f.readline()
                if not line:
                    break
                head_lines.append(line.strip())

            # Count and find first prompt
            message_count = 0
            for line in head_lines:
                if not line:
                    continue
                message_count += 1
                if '"role":"user"' in line or '"role": "user"' in line:
                    if "last_prompt" not in metadata:
                        try:
                            msg = message_from_jsonl(line)
                            text = msg.get_text()
                            if text and len(text) > 5:
                                metadata["last_prompt"] = text[:200]
                        except Exception:
                            pass

            # Read tail for last activity
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                f.readline()  # Skip partial line
                tail_lines = f.readlines()
            else:
                f.seek(0)
                tail_lines = f.readlines()

            # Count remaining lines
            message_count += len([l for l in tail_lines if l.strip()])
            metadata["message_count"] = message_count

            # Extract last user prompt from tail
            for line in reversed(tail_lines):
                line = line.strip()
                if not line:
                    continue
                if '"role":"user"' in line or '"role": "user"' in line:
                    try:
                        msg = message_from_jsonl(line)
                        text = msg.get_text()
                        if text and len(text) > 5 and not msg.tool_call_id:
                            metadata["last_prompt"] = text[:200]
                            break
                    except Exception:
                        pass

    except OSError:
        pass

    return metadata


async def load_session(
    sessions_dir: Path,
    session_id: str,
) -> tuple[list[Message], dict[str, Any]]:
    """Load a full session from JSONL file.

    Returns (messages, metadata).
    Handles interruption detection and message cleanup.
    """
    path = sessions_dir / f"{session_id}.jsonl"
    if not path.exists():
        return [], {}

    messages: list[Message] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = message_from_jsonl(line)
                messages.append(msg)
            except Exception:
                logger.warning("Failed to parse JSONL line in session %s", session_id)
                continue

    # Detect interruption and clean up
    messages, was_interrupted = detect_interrupted_conversation(messages)

    metadata = {
        "was_interrupted": was_interrupted,
        "message_count": len(messages),
    }

    return messages, metadata
