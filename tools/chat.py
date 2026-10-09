from tools.base import BaseTool, ToolCallResult


class ChatTool(BaseTool):
    name = "chat"
    description = (
        "Answer questions, explain things, write text, brainstorm, or hold a conversation, "
        "using general knowledge or data already retrieved. For purely conversational turns "
        "that need no external tools."
    )
    is_conversational = True

    def run(self, query: str, data: dict) -> ToolCallResult:
        # Chat-only plans never get here (the orchestrator answers them directly). In a mixed
        # plan the step just passes its resolved input along for the final answer.
        return ToolCallResult(True, query)
