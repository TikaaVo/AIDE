"""The single place tools are registered. See docs/adding-a-tool.md."""

from core.llm import LLM
from tools.base import BaseTool
from tools.chat import ChatTool
from tools.deep_think import DeepThinkTool
from tools.file_search import FileSearchTool
from tools.open_file import OpenFileTool
from tools.open_url import OpenUrlTool
from tools.read_file import ReadFileTool
from tools.summarize import SummarizeTool
from tools.terminal import TerminalTool
from tools.web_search import WebSearchTool


def build_registry(llm: LLM, embedder: LLM) -> dict[str, BaseTool]:
    tools: list[BaseTool] = [
        FileSearchTool(embedder),
        OpenFileTool(),
        ReadFileTool(),
        TerminalTool(llm),
        ChatTool(),
        WebSearchTool(),
        OpenUrlTool(),
        DeepThinkTool(llm),
        SummarizeTool(llm),
    ]
    return {tool.name: tool for tool in tools}
