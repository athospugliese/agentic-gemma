"""WebSearch tool - search the web using OpenAI's web search or fallback."""

from __future__ import annotations

from typing import Any

from app.tools.base import Tool, ToolContext, ToolResult


class WebSearchTool(Tool):
    name = "WebSearch"
    description = "Search the web for current information. Returns search results with titles, snippets, and URLs."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    }

    def is_read_only(self, input: dict[str, Any]) -> bool:
        return True

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        query = input["query"]
        num_results = min(input.get("num_results", 5), 10)

        # Use OpenAI's web search via responses API if available,
        # otherwise fall back to WebFetch on a search engine
        try:
            return await self._search_via_fetch(query, num_results, context)
        except Exception as e:
            return ToolResult(output=f"Search failed: {e}", is_error=True)

    async def _search_via_fetch(self, query: str, num_results: int, context: ToolContext) -> ToolResult:
        """Fallback: use DuckDuckGo HTML search via httpx."""
        import httpx
        import re

        url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
                response = await client.get(url, headers={"User-Agent": "AgentAPI/1.0"})

            if response.status_code != 200:
                return ToolResult(output=f"Search returned HTTP {response.status_code}", is_error=True)

            html = response.text

            # Parse results from DuckDuckGo HTML
            results = []
            # Find result blocks
            result_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                re.DOTALL,
            )

            for match in result_pattern.finditer(html):
                if len(results) >= num_results:
                    break
                href = match.group(1)
                title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()

                # DuckDuckGo wraps URLs in a redirect
                if "uddg=" in href:
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]

                results.append({"title": title, "url": href, "snippet": snippet})

            if not results:
                # Simpler fallback: just extract links
                link_pattern = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
                for match in link_pattern.finditer(html):
                    if len(results) >= num_results:
                        break
                    href = match.group(1)
                    title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                    if href and title:
                        results.append({"title": title, "url": href, "snippet": ""})

            if not results:
                return ToolResult(output=f"No results found for: {query}")

            # Format output
            lines = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. **{r['title']}**")
                lines.append(f"   URL: {r['url']}")
                if r["snippet"]:
                    lines.append(f"   {r['snippet']}")
                lines.append("")

            return ToolResult(
                output="\n".join(lines),
                metadata={"query": query, "num_results": len(results)},
            )

        except httpx.TimeoutException:
            return ToolResult(output=f"Search timed out for: {query}", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Search error: {e}", is_error=True)
