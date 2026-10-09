"""Tool interface. Subclass BaseTool, then register it in registry.py."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolCallResult:
    success: bool
    output: str                                   # text the resolver and summarizer read
    data: dict = field(default_factory=dict)      # structured data for downstream tools
    requires_confirmation: bool = False
    confirmation_prompt: str = ""


class BaseTool(ABC):
    name: str
    # Shown to the planner: say what the tool does and WHEN to pick it.
    description: str
    # Shown to the input resolver when this tool is chained after another:
    # describe the format of the `query` string that run() expects.
    input_schema: str = "A natural-language instruction for this tool."
    # True for tools the orchestrator answers itself; run() is never called.
    is_conversational: bool = False

    @abstractmethod
    def run(self, query: str, data: dict) -> ToolCallResult:
        """
        query: the step's input string (planner-written, or built by the resolver).
        data:  session data plus the structured output of this step's dependencies.
        """


def resolve_path(query: str, data: dict) -> str | None:
    """Find an existing absolute path in the query text, else fall back to the session's current file."""
    for token in [query.strip(), *query.split()]:
        token = token.strip("'\"")
        if token.startswith("/") and Path(token).exists():
            return token
    path = data.get("path")
    return path if path and Path(path).exists() else None
