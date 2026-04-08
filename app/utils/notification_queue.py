"""Notification queue - thread-safe queue for task notifications.

Background agents enqueue notifications when they complete/fail.
The query loop drains the queue before each API call and injects
notifications as user-role messages.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class NotificationQueue:
    """Async queue for task notifications between agents and the main loop."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def enqueue(self, notification: str) -> None:
        """Add a notification to the queue."""
        await self._queue.put(notification)
        logger.debug("Notification enqueued (%d pending)", self._queue.qsize())

    def drain(self) -> list[str]:
        """Non-blocking drain of all pending notifications.

        Returns all queued notifications without waiting.
        """
        items: list[str] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
