import datetime
import logging
import re

from ddgs import DDGS

from tools.base import BaseTool, ToolCallResult

log = logging.getLogger("aide.web_search")

MAX_RESULTS = 10


def refresh_years(query: str) -> str:
    """
    Planners tend to write a stale "current" year into queries. Bump any year from the
    last three to this year, and leave older ones alone ("events of 1989").
    """
    now = datetime.datetime.now().year

    def bump(m: re.Match) -> str:
        year = int(m.group(0))
        return str(now) if now - 3 <= year < now else m.group(0)

    return re.sub(r"\b(?:19|20)\d{2}\b", bump, query)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web (DuckDuckGo) for current information: news, facts, prices, weather, "
        "documentation, anything recent or outside the model's training data. Use it whenever "
        "fresh information could help, even if the user didn't ask for it, but don't overdo it. "
        "Not for the user's local files (use file_search). For deep research, chain several searches."
    )
    input_schema = (
        "A concise search query, 3-8 words, phrased like a web search. Only include a year if it "
        "matters, and then use the current year. "
        "Examples: 'Python 3.13 new features', 'weather Rome this week'."
    )

    def run(self, query: str, data: dict) -> ToolCallResult:
        query = refresh_years(query)
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=MAX_RESULTS))
        except Exception as e:  # the library raises assorted errors (rate limits, network)
            return ToolCallResult(False, f"Web search failed: {e}")

        if not raw:
            return ToolCallResult(False, f"No results found for '{query}'.")

        results = [
            {"title": r.get("title", "").strip(), "snippet": r.get("body", "").strip(), "url": r.get("href", "").strip()}
            for r in raw
        ]
        lines = [f"{i}. {r['title']}\n   {r['snippet']}\n   Source: {r['url']}" for i, r in enumerate(results, 1)]
        log.info("%d results for %r", len(results), query)
        return ToolCallResult(
            True,
            f"Web search results for '{query}':\n\n" + "\n\n".join(lines),
            {"results": results, "query": query},
        )
