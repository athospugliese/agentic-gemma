"""Side query utility - auxiliary LLM calls for memory ranking, compaction, etc.

Supports both OpenAI and Ollama adapters transparently.
"""

from __future__ import annotations

import json
from typing import Any


async def side_query(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Make an auxiliary LLM call (non-streaming, no tools).

    Used for: memory ranking, compaction summaries, permission explanations.
    Accepts either an AsyncOpenAI client or an OllamaAdapter.
    """
    # Check if client is an OllamaAdapter (has side_query method)
    if hasattr(client, "side_query"):
        return await client.side_query(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
        )

    # Otherwise, use OpenAI client
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }

    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    return choice.message.content or ""


async def side_query_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Side query that returns parsed JSON."""
    result = await side_query(
        client=client,
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        response_format={"type": "json_object"} if not hasattr(client, "side_query") else None,
    )
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {}
