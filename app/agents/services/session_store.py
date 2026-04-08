"""Session store - manages active QueryEngine instances and persistence."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import aiofiles

from app.models import Message, MessageRole, message_from_jsonl, message_to_jsonl

logger = logging.getLogger(__name__)


class SessionStore:
    """Manages active sessions and JSONL transcript persistence."""

    def __init__(self, base_dir: str = "~/.agent/sessions"):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, QueryEngine] = {}
        self._last_access: dict[str, float] = {}

    def register(self, session_id: str, engine: QueryEngine) -> None:
        self._active[session_id] = engine
        self._last_access[session_id] = time.time()

    def get(self, session_id: str) -> QueryEngine | None:
        engine = self._active.get(session_id)
        if engine:
            self._last_access[session_id] = time.time()
        return engine

    def remove(self, session_id: str) -> None:
        self._active.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def list_active(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": sid,
                "turn_count": engine.turn_count,
                "message_count": len(engine.messages),
                "model": engine.model,
                "last_access": self._last_access.get(sid, 0),
            }
            for sid, engine in self._active.items()
        ]

    # -- Persistence (JSONL) -----------------------------------------------

    def _transcript_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    async def save_message(self, session_id: str, message: Message) -> None:
        """Append a single message to the session transcript."""
        path = self._transcript_path(session_id)
        line = message_to_jsonl(message) + "\n"
        async with aiofiles.open(path, "a") as f:
            await f.write(line)

    async def save_messages(self, session_id: str, messages: list[Message]) -> None:
        """Append multiple messages to the session transcript."""
        path = self._transcript_path(session_id)
        lines = [message_to_jsonl(m) + "\n" for m in messages]
        async with aiofiles.open(path, "a") as f:
            await f.writelines(lines)

    async def save_session_metadata(self, session_id: str, engine: Any) -> None:
        """Persist session configuration metadata to JSONL for recovery on resume."""
        import json
        metadata = {
            "type": "session_metadata",
            "model": getattr(engine, "model", ""),
            "fast_model": getattr(engine, "fast_model", ""),
            "coordinator_mode": getattr(engine, "coordinator_mode", False),
            "permission_mode": getattr(engine, "permissions", None) and engine.permissions.mode or "auto",
            "cwd": getattr(engine, "cwd", "."),
            "custom_system_prompt": (engine.custom_system_prompt[:500] if engine.custom_system_prompt else None),
            "turn_count": getattr(engine, "turn_count", 0),
            "total_usage": engine.total_usage.model_dump() if hasattr(engine, "total_usage") else {},
            "timestamp": time.time(),
        }
        line = json.dumps({"_meta": True, **metadata}) + "\n"
        path = self._transcript_path(session_id)
        async with aiofiles.open(path, "a") as f:
            await f.write(line)

    async def load_transcript(self, session_id: str) -> tuple[list[Message], dict[str, Any]]:
        """Load messages and metadata from a session transcript.

        Returns (messages, metadata).
        For large files (>1MB), uses backward scan to only load post-boundary messages.
        Extracts session metadata entries for state restoration.
        """
        import json

        path = self._transcript_path(session_id)
        if not path.exists():
            return [], {}

        file_size = path.stat().st_size
        metadata: dict[str, Any] = {}
        messages: list[Message] = []

        # For large files, use backward scan to find last compact boundary
        if file_size > 1_000_000:
            messages, metadata = await self._load_transcript_optimized(path)
        else:
            messages, metadata = await self._load_transcript_full(path)

        return messages, metadata

    async def _load_transcript_full(self, path: Path) -> tuple[list[Message], dict[str, Any]]:
        """Load entire JSONL file, extracting metadata entries."""
        import json

        messages: list[Message] = []
        metadata: dict[str, Any] = {}
        last_boundary_idx: int = -1

        async with aiofiles.open(path, "r") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    # Extract metadata entries (not messages)
                    if data.get("_meta"):
                        metadata.update(data)
                        continue

                    msg = message_from_jsonl(line)
                    messages.append(msg)

                    # Track last compact boundary position
                    if hasattr(msg, "system_message_type") and msg.system_message_type == "compact_boundary":
                        last_boundary_idx = len(messages) - 1

                except Exception:
                    logger.warning("Failed to parse JSONL line in %s", path.name)
                    continue

        # If there's a boundary, only return post-boundary messages
        if last_boundary_idx >= 0:
            messages = messages[last_boundary_idx:]  # Include boundary itself

        return messages, metadata

    async def _load_transcript_optimized(self, path: Path) -> tuple[list[Message], dict[str, Any]]:
        """Backward scan for large files: find last boundary, load only post-boundary.

        Reads last 64KB first to find metadata, then scans backward for boundary.
        """
        import json

        TAIL_SIZE = 65536
        metadata: dict[str, Any] = {}
        file_size = path.stat().st_size

        # Read tail for metadata
        async with aiofiles.open(path, "r") as f:
            if file_size > TAIL_SIZE:
                await f.seek(file_size - TAIL_SIZE)
                await f.readline()  # Skip partial line
            tail_lines = await f.readlines()

        # Extract metadata from tail
        for line in reversed(tail_lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("_meta") and "model" not in metadata:
                    metadata.update(data)
            except (json.JSONDecodeError, TypeError):
                continue

        # Find last compact boundary by scanning backward
        boundary_offset = -1
        async with aiofiles.open(path, "r") as f:
            # Read in reverse chunks
            chunk_size = min(262144, file_size)  # 256KB chunks
            pos = file_size

            while pos > 0:
                read_start = max(0, pos - chunk_size)
                await f.seek(read_start)
                if read_start > 0:
                    await f.readline()  # Skip partial line
                chunk = await f.read(pos - read_start)

                if "compact_boundary" in chunk:
                    # Found a boundary in this chunk - find exact line
                    for line in reversed(chunk.splitlines()):
                        if "compact_boundary" in line:
                            boundary_offset = read_start
                            break
                    if boundary_offset >= 0:
                        break

                pos = read_start

        # Load from boundary (or from start if no boundary)
        messages: list[Message] = []
        async with aiofiles.open(path, "r") as f:
            if boundary_offset > 0:
                await f.seek(boundary_offset)
                await f.readline()  # Skip partial line

            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("_meta"):
                        continue
                    msg = message_from_jsonl(line)
                    messages.append(msg)
                except Exception:
                    continue

        logger.info("Optimized load: %d bytes -> %d messages (boundary at %d)", file_size, len(messages), boundary_offset)
        return messages, metadata

    async def list_persisted(self) -> list[dict[str, Any]]:
        """List all persisted sessions from disk."""
        sessions = []
        for path in sorted(self.base_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            session_id = path.stem
            stat = path.stat()
            sessions.append({
                "session_id": session_id,
                "size_bytes": stat.st_size,
                "last_modified": stat.st_mtime,
                "is_active": session_id in self._active,
            })
        return sessions

    async def cleanup_expired(self, max_idle_seconds: int = 3600) -> int:
        """Remove sessions idle longer than max_idle_seconds."""
        now = time.time()
        expired = [
            sid for sid, last in self._last_access.items()
            if now - last > max_idle_seconds
        ]
        for sid in expired:
            self.remove(sid)
        return len(expired)
