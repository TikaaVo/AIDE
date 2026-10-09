import sys
import threading
import types

import pytest

from core.llm import LLM
from tools.base import BaseTool, ToolCallResult


class FakeLLM(LLM):
    """
    Scripted stand-in for the model.

    `routes` maps a marker string to a reply. The first marker found anywhere in the prompt wins,
    so tests don't depend on call order (the orchestrator makes some calls in parallel).
    A list value is consumed one reply per call. Every call is recorded in `calls`.
    """

    def __init__(self, routes: dict[str, str | list[str]]):
        super().__init__("http://fake", "fake")
        self.routes = {k: list(v) if isinstance(v, list) else v for k, v in routes.items()}
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def chat(self, messages, **kwargs) -> str:
        text = "\n".join(m["content"] for m in messages)
        with self._lock:
            self.calls.append({"text": text, **kwargs})
            for marker, reply in self.routes.items():
                if marker in text:
                    if not isinstance(reply, list):
                        return reply
                    if reply:
                        return reply.pop(0)
                    raise AssertionError(f"route {marker!r} ran out of replies")
        raise AssertionError(f"no route matches prompt: ...{text[-300:]!r}")

    def prompts_containing(self, marker: str) -> list[str]:
        return [c["text"] for c in self.calls if marker in c["text"]]


class EchoTool(BaseTool):
    """Returns its input; optionally fails or asks for confirmation. Records every call."""
    description = "echo"

    def __init__(self, name="echo", success=True, confirm=False, data=None):
        self.name = name
        self.success = success
        self.confirm = confirm
        self.data = data or {}
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, data):
        self.calls.append((query, data))
        if self.confirm and not data.get("confirmed"):
            return ToolCallResult(False, "", requires_confirmation=True, confirmation_prompt=f"run {query}?")
        return ToolCallResult(self.success, f"{self.name} says {query}", self.data)


@pytest.fixture(autouse=True)
def _stub_ddgs(monkeypatch):
    """Keep the suite importable and offline even if the search library isn't installed."""
    try:
        import ddgs  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=object))
