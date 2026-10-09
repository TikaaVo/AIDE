import logging

from core.llm import LLM
from tools.base import BaseTool, ToolCallResult

log = logging.getLogger("aide.deep_think")

PROMPT = """\
Think through the request below and the data provided, but do NOT answer the user.
Write a concise internal monologue that a later model can use to answer well: weigh different
perspectives, question your own ideas, point out risks, and add new angles and details rather
than just restating the input.

{query}

Reason step by step inside <thinking> tags. Output only the reasoning."""


class DeepThinkTool(BaseTool):
    name = "deep_think"
    description = (
        "Reason carefully before acting. Use for complex requests, conflicting sources, or risky "
        "steps such as deciding which shell command to run. Not for simple lookups. "
        "The output is reasoning text for later steps."
    )
    input_schema = (
        "What to think about, including the user's request and the relevant data. "
        "Example: 'Given the web results and file content, what is the exact command to run this script?'"
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, query: str, data: dict) -> ToolCallResult:
        reply = self.llm.ask(PROMPT.format(query=query), think=True, temperature=0.5, max_tokens=2048)
        reasoning = reply.replace("<thinking>", "").replace("</thinking>", "").strip()
        log.debug("reasoning (%d chars): %s", len(reasoning), reasoning)
        return ToolCallResult(success=True, output=reasoning, data={"reasoning": reasoning})
