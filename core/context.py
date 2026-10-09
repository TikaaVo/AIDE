"""Session state shared by the planner, executor and orchestrator."""

import threading
from dataclasses import dataclass, field

# Output of these tools is internal scaffolding: it stays in tool_results for
# auditing but is left out of the prompts the planner and synthesizer see.
_SCAFFOLDING_TOOLS = {"deep_think", "summarize"}

# Rendered prompts keep only the most recent N results per tool, so a long
# session doesn't balloon the context window.
_MAX_PER_TOOL = 3


@dataclass
class ToolResult:
    """A finished tool step, as stored in the session."""
    tool: str
    input: str
    success: bool
    output: str
    data: dict = field(default_factory=dict)


@dataclass
class SessionContext:
    """
    Everything remembered during one conversation.

    history       visible chat turns ({"role", "content"})
    tool_results  every tool result so far, so nothing gets re-fetched
    data          key/value bag that outlives a single turn: file paths from
                  file_search, web search results, URLs already attempted
    """
    history: list[dict] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    # Threads run in parallel and all write here.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def add_turn(self, role: str, content: str) -> None:
        with self._lock:
            self.history.append({"role": role, "content": content})

    def add_tool_result(self, result: ToolResult) -> None:
        """Store a result and promote well-known fields into the session data."""
        with self._lock:
            self.tool_results.append(result)

            if result.tool == "file_search" and result.success:
                hits = result.data.get("hits", [])
                if hits:
                    self.data["path"] = hits[0]["path"]
                    self.data["name"] = hits[0]["name"]
                    self.data["all_paths"] = [h["path"] for h in hits]

            elif result.tool == "web_search" and result.success:
                query = result.data.get("query", result.input)
                self.data.setdefault("web_searches", {})[query] = result.data.get("results", [])

            elif result.tool == "open_url":
                url = result.data.get("url")
                attempted = self.data.setdefault("attempted_urls", [])
                if url and url not in attempted:
                    attempted.append(url)

    def tool_results_text(self) -> str:
        """Recent tool results as a text block for LLM prompts (scaffolding excluded)."""
        with self._lock:
            visible = [r for r in self.tool_results if r.tool not in _SCAFFOLDING_TOOLS]

        counts: dict[str, int] = {}
        kept: list[ToolResult] = []
        for r in reversed(visible):
            if counts.get(r.tool, 0) < _MAX_PER_TOOL:
                kept.append(r)
                counts[r.tool] = counts.get(r.tool, 0) + 1

        return "\n\n".join(
            f"[{'✓' if r.success else '✗'} {r.tool} — {r.input!r}]\n{r.output}"
            for r in reversed(kept)
        )

    def web_search_context(self) -> str:
        """The last few web searches, compactly, for the synthesizer to pull facts from."""
        with self._lock:
            searches = list(self.data.get("web_searches", {}).items())[-_MAX_PER_TOOL:]

        blocks = []
        for query, results in searches:
            if not results:
                continue
            lines = [f"Web search: '{query}'"]
            for i, r in enumerate(results[:5], 1):
                lines.append(f"  {i}. {r.get('title', '')}\n     {r.get('snippet', '')}\n     {r.get('url', '')}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def history_text(self, n: int = 8) -> str:
        with self._lock:
            recent = self.history[-n:]
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    def snapshot_data(self) -> dict:
        """Shallow copy of the session data, safe to hand to a worker thread."""
        with self._lock:
            return dict(self.data)
