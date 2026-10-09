from core.llm import LLM
from tools.base import BaseTool, ToolCallResult

SYSTEM = (
    "You write concise, factual summaries of tool outputs. Keep every important fact, number, "
    "file path, command output and error. Do not address the user; output only the summary."
)


class SummarizeTool(BaseTool):
    name = "summarize"
    description = "Summarize all earlier step outputs of a thread. Appended automatically by the planner."
    input_schema = "The thread goal followed by the tool outputs to summarize (built by the executor)."

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, query: str, data: dict) -> ToolCallResult:
        summary = self.llm.ask(query, system=SYSTEM, temperature=0.2, max_tokens=600)
        return ToolCallResult(success=True, output=summary, data={"summary": summary})
