import logging
from pathlib import Path

from tools.base import BaseTool, ToolCallResult, resolve_path

log = logging.getLogger("aide.read_file")

MAX_CHARS = 20_000


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a local text file so AIDE itself can examine, summarize or answer questions about it. "
        "To launch a file in an external app use open_file instead. "
        "If the path isn't known, run file_search first."
    )
    input_schema = (
        "An absolute path to an existing file, e.g. /home/user/docs/notes.txt. "
        "After file_search, pick the match that best fits the user's intent, not just the top hit."
    )

    def run(self, query: str, data: dict) -> ToolCallResult:
        path = resolve_path(query, data)
        if not path:
            return ToolCallResult(False, "No file path available. Run a file search first.")
        try:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return ToolCallResult(False, f"Failed to read file: {e}")

        truncated = len(content) > MAX_CHARS
        shown = content[:MAX_CHARS] + (f"\n\n[Truncated at {MAX_CHARS} characters]" if truncated else "")
        log.info("read %s (%d chars, truncated=%s)", path, len(content), truncated)
        return ToolCallResult(
            True,
            f"Content of {Path(path).name}:\n\n{shown}",
            {"path": path, "content": shown, "truncated": truncated},
        )
