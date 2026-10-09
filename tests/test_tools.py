import datetime

import pytest

from tools.base import resolve_path
from tools.open_url import OpenUrlTool, html_to_text, pick_url
from tools.read_file import MAX_CHARS, ReadFileTool
from tools.web_search import WebSearchTool, refresh_years


def test_resolve_path_prefers_the_query_then_the_session(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    assert resolve_path(f"please open '{a}' now", {"path": str(b)}) == str(a)
    assert resolve_path("open it", {"path": str(b)}) == str(b)
    assert resolve_path("open it", {"path": str(tmp_path / "missing")}) is None


def test_read_file_truncates_long_files(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * (MAX_CHARS + 50))
    result = ReadFileTool().run(str(big), {})
    assert result.success and result.data["truncated"] and "Truncated" in result.output


def test_read_file_without_a_path_fails_politely():
    assert not ReadFileTool().run("that file", {}).success


def test_refresh_years_bumps_recent_years_only():
    now = datetime.datetime.now().year
    assert refresh_years(f"best laptops {now - 2}") == f"best laptops {now}"
    assert refresh_years("events of 1989") == "events of 1989"
    assert refresh_years(f"roadmap {now + 2}") == f"roadmap {now + 2}"


def test_web_search_formats_results(monkeypatch):
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, query, max_results):
            return [{"title": "T", "body": "B", "href": "https://t.dev"}]

    monkeypatch.setattr("tools.web_search.DDGS", FakeDDGS)
    result = WebSearchTool().run("python news", {})
    assert result.success and result.data["results"][0]["url"] == "https://t.dev"


def test_web_search_reports_library_errors(monkeypatch):
    class Broken:
        def __enter__(self): raise RuntimeError("rate limited")
        def __exit__(self, *a): return False

    monkeypatch.setattr("tools.web_search.DDGS", Broken)
    result = WebSearchTool().run("anything", {})
    assert not result.success and "rate limited" in result.output


def test_pick_url_skips_attempted_and_generic_pages():
    data = {
        "results": [{"url": "https://github.com/"}, {"url": "https://a.dev/x"}, {"url": "https://b.dev/y"}],
        "attempted_urls": ["https://a.dev/x"],
    }
    assert pick_url("fetch top result", data) == "https://b.dev/y"
    assert pick_url("https://direct.dev/page", data) == "https://direct.dev/page"
    assert pick_url("nothing here", {}) is None


def test_html_to_text_drops_scripts_and_nav():
    html = "<nav>menu</nav><script>evil()</script><p>Hello</p><p>World</p>"
    text = html_to_text(html)
    assert "Hello" in text and "World" in text and "evil" not in text and "menu" not in text


def test_open_url_handles_http_errors(monkeypatch):
    import urllib.error

    def boom(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr("tools.open_url.fetch", boom)
    result = OpenUrlTool().run("https://x.dev/gone", {})
    assert not result.success and "404" in result.output and result.data["url"] == "https://x.dev/gone"
