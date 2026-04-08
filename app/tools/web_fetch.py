"""WebFetch tool - fetch and process web content."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import httpx

from app.tools.base import Tool, ToolContext, ToolResult

CACHE_TTL = 900  # 15 minutes
MAX_CONTENT_SIZE = 500_000  # 500KB
_url_cache: dict[str, tuple[float, str]] = {}


class WebFetchTool(Tool):
    name = "WebFetch"
    description = "Fetch a URL and return its content as markdown text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (must be a valid http/https URL).",
            },
            "prompt": {
                "type": "string",
                "description": "Optional: how to process/filter the content.",
            },
        },
        "required": ["url"],
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        url = input["url"]
        prompt = input.get("prompt")

        if not url.startswith(("http://", "https://")):
            return ToolResult(output=f"Invalid URL (must start with http:// or https://): {url}", is_error=True)

        # Check cache
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cache_key in _url_cache:
            cached_time, cached_content = _url_cache[cache_key]
            if time.time() - cached_time < CACHE_TTL:
                return ToolResult(output=cached_content, metadata={"cached": True, "url": url})

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                response = await client.get(url, headers={"User-Agent": "AgentAPI/1.0"})

            status = response.status_code
            content_type = response.headers.get("content-type", "")

            if status >= 400:
                return ToolResult(output=f"HTTP {status}: {response.reason_phrase}", is_error=True)

            body = response.text[:MAX_CONTENT_SIZE]

            # Convert HTML to markdown
            if "html" in content_type:
                try:
                    from markdownify import markdownify
                    body = markdownify(body, heading_style="ATX", strip=["script", "style", "nav", "footer"])
                except ImportError:
                    pass  # Fallback to raw text

            # Clean up excessive whitespace
            import re
            body = re.sub(r"\n{3,}", "\n\n", body).strip()

            # Truncate if still too long
            if len(body) > MAX_CONTENT_SIZE:
                body = body[:MAX_CONTENT_SIZE] + "\n\n... (truncated)"

            # Cache result
            _url_cache[cache_key] = (time.time(), body)

            # Evict old cache entries
            if len(_url_cache) > 100:
                oldest_key = min(_url_cache, key=lambda k: _url_cache[k][0])
                del _url_cache[oldest_key]

            output = f"URL: {url}\nStatus: {status}\nContent-Type: {content_type}\n\n{body}"
            return ToolResult(output=output, metadata={"url": url, "status": status, "bytes": len(body)})

        except httpx.TimeoutException:
            return ToolResult(output=f"Timeout fetching {url}", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Error fetching {url}: {e}", is_error=True)
