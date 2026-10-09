import logging
import subprocess
from pathlib import Path

from tools.base import BaseTool, ToolCallResult, resolve_path

log = logging.getLogger("aide.open_file")


class OpenFileTool(BaseTool):
    name = "open_file"
    description = (
        "Open a file in the system's default application (xdg-open). Use when the user wants a file "
        "opened and its path is known from context; otherwise run file_search first. "
        "This does not execute code."
    )
    input_schema = (
        "An absolute path to an existing file, e.g. /home/user/docs/report.pdf. "
        "After file_search, pick the match that best fits the user's intent, not just the top hit."
    )

    def run(self, query: str, data: dict) -> ToolCallResult:
        path = resolve_path(query, data)
        if not path:
            return ToolCallResult(False, "No file path available. Run a file search first.")
        try:
            subprocess.Popen(["xdg-open", path])
        except OSError as e:
            return ToolCallResult(False, f"Failed to open file: {e}")
        log.info("opened %s", path)
        return ToolCallResult(True, f"Opened {Path(path).name}.", {"path": path})
