import http.client
import logging
import re
import urllib.error
import urllib.request

from bs4 import BeautifulSoup

from tools.base import BaseTool, ToolCallResult

log = logging.getLogger("aide.open_url")

MAX_CHARS = 8_000
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# Bare homepages of big sites make poor answers; prefer deep links when we have a choice.
_GENERIC_HOMEPAGE = re.compile(
    r"^https?://(?:www\.)?(amazon|ebay|google|youtube|twitter|x|facebook|instagram|reddit|"
    r"wikipedia|github|linkedin)\.[a-z.]+/?$",
    re.IGNORECASE,
)


def pick_url(query: str, data: dict) -> str | None:
    """Choose what to fetch: a URL in the query, else the best unvisited search result."""
    text = query.strip().strip("'\"")
    if text.startswith(("http://", "https://")):
        return text

    attempted = set(data.get("attempted_urls", []))
    urls = [r["url"] for r in data.get("results", []) if r.get("url")]
    fresh = [u for u in urls if u not in attempted]
    specific = [u for u in fresh if not _GENERIC_HOMEPAGE.match(u)]
    for candidates in (specific, fresh, urls):
        if candidates:
            return candidates[0]

    match = re.search(r"https?://\S+", query)
    return match.group(0).rstrip(".,)") if match else None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="ignore")


class OpenUrlTool(BaseTool):
    name = "open_url"
    description = (
        "Fetch the text of a web page. Use it to read or summarize a specific page, especially to "
        "explore URLs returned by web_search. Not for searching (use web_search)."
    )
    input_schema = (
        "A single complete URL including the scheme, e.g. 'https://pytorch.org/docs/stable/data.html'. "
        "After web_search, pick the most relevant result that is NOT in the already-attempted list, "
        "and prefer deep links over bare homepages. Output only the URL."
    )

    def run(self, query: str, data: dict) -> ToolCallResult:
        url = pick_url(query, data)
        if not url:
            return ToolCallResult(False, "No URL to fetch. Run a web search first or provide a URL.")

        try:
            text = html_to_text(fetch(url))
        except urllib.error.HTTPError as e:
            return ToolCallResult(False, f"HTTP {e.code} fetching {url}: {e.reason}", {"url": url})
        except (OSError, ValueError, http.client.HTTPException) as e:
            return ToolCallResult(False, f"Could not fetch {url}: {e}", {"url": url})

        if not text:
            return ToolCallResult(
                False,
                f"Fetched {url} but found no readable text (JavaScript-rendered or login-only page?).",
                {"url": url},
            )

        truncated = len(text) > MAX_CHARS
        content = text[:MAX_CHARS]
        log.info("fetched %s (%d chars%s)", url, len(text), ", truncated" if truncated else "")
        note = f"\n\n[Truncated at {MAX_CHARS} characters]" if truncated else ""
        return ToolCallResult(
            True,
            f"Content from {url}:\n\n{content}{note}",
            {"url": url, "content": content, "truncated": truncated},
        )
