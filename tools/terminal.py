"""
Shell access with safety tiers. These are guard rails, not a sandbox.

  READ_ONLY     runs immediately (ls, cat, grep, git status, ...)
  SCRIPT_RUN    interpreters and executables: always asks first
  REVERSIBLE    changes something but can be undone: asks first
  IRREVERSIBLE  never runs (rm -rf /, mkfs, dd, shutdown, ...)

Classification is regex first, with the LLM as a fallback for anything ambiguous.
A command that prints nothing for SILENCE_TIMEOUT seconds is killed.
"""

import logging
import os
import re
import signal
import subprocess
import threading
import time

from core.llm import LLM, clean_reply
from core.prompts import os_name
from tools.base import BaseTool, ToolCallResult

log = logging.getLogger("aide.terminal")

SILENCE_TIMEOUT = 30        # seconds without output before the process is killed
MAX_OUTPUT_CHARS = 8_000

_IRREVERSIBLE = re.compile(
    r"\b("
    r"rm\s+(-[rfRF]+\s+)?(/|~|\.\.)"        # rm aimed at root, home or parent
    r"|rm\s+.*\*"                           # rm with a glob
    r"|mkfs|fdisk|parted|gparted|shred"
    r"|dd\s+if="
    r"|shutdown|reboot|poweroff|halt"
    r"|systemctl\s+(stop|disable|mask)\s+\S"
    r"|passwd\s|deluser|userdel"
    r"|iptables\s+(-F|--flush)"
    r"|cryptsetup\s+luksFormat"
    r")",
    re.IGNORECASE,
)
_FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{")

# Running code can do anything, so it always needs a yes (except version checks).
_SCRIPT_RUN = re.compile(
    r"^\s*((python3?|node|ruby|perl|php|bash|sh|zsh)\s+(?!-V\b|--version\b)\S|\./|/\S+\s)",
    re.IGNORECASE,
)

_READ_ONLY = re.compile(
    r"^\s*("
    r"ls|ll|la|dir|find|locate|which|whereis|type"
    r"|cat|head|tail|bat|grep|rg|awk|sed(?!\s+-i)|cut|sort|uniq|wc|tr|diff"
    r"|echo|printf|pwd|whoami|id|groups|date|cal|uptime|hostname"
    r"|ps|top|htop|pstree|lsof|netstat|ss"
    r"|df|du|lsblk|lspci|lsusb|lshw|free|vmstat|iostat"
    r"|env|printenv|file|stat|md5sum|sha\d+sum|man|info|history|alias|tree"
    r"|git\s+(log|status|diff|show|branch|tag|remote\s+-v|stash\s+list)"
    r"|python3?\s+(-V|--version)"
    r"|pip\s+(list|show|freeze|check)"
    r"|apt[-\s]?(list|show|search|policy|depends)"
    r"|dpkg\s+(-l|-s|-L|-p)"
    r"|systemctl\s+(status|is-active|is-enabled|list-units|list-timers)"
    r"|journalctl(?!.*--(rotate|vacuum|flush))"
    r"|curl\s+(-I|-i|--head)"
    r"|wget\s+--spider"
    r")\b",
    re.IGNORECASE,
)
_MUTATING_WORDS = re.compile(r"\b(rm|mv|cp|chmod|chown|install|truncate|dd|mkfs|tee)\b|-delete|-exec", re.IGNORECASE)
_SHELL_TRICKS = re.compile(r"[;&<>`$()\n]")   # chaining, redirects, substitution

_CLASSIFIER = """\
You classify shell commands for safety. Output exactly one word:
READ_ONLY: only reads data (ls, cat, grep, git log, df, ps, apt list)
REVERSIBLE: changes things, but they can be undone (mkdir, touch, cp, mv, apt install, pip install, git commit)
IRREVERSIBLE: permanently destroys data or system state (rm -rf /, dd if=, mkfs, shred, truncate, passwd)"""


def _is_read_only(command: str) -> bool:
    """Fast path: every pipe segment is a known read-only command, with no shell tricks."""
    if _SHELL_TRICKS.search(command) or _MUTATING_WORDS.search(command):
        return False
    return all(_READ_ONLY.match(part) for part in command.split("|"))


def classify_command(command: str, llm: LLM) -> str:
    """Return READ_ONLY, SCRIPT_RUN, REVERSIBLE or IRREVERSIBLE."""
    if _IRREVERSIBLE.search(command) or _FORK_BOMB.search(command):
        return "IRREVERSIBLE"
    if any(_SCRIPT_RUN.match(part) for part in command.split("|")):
        return "SCRIPT_RUN"
    if _is_read_only(command):
        return "READ_ONLY"

    verdict = llm.ask(command, system=_CLASSIFIER, temperature=0.0, max_tokens=10).upper()
    # IRREVERSIBLE contains "REVERSIBLE", so it has to be checked first.
    for tier in ("IRREVERSIBLE", "READ_ONLY", "REVERSIBLE"):
        if tier in verdict:
            return tier
    return "REVERSIBLE"   # unsure: ask the user


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(1)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:   # it exited on its own in the meantime
        pass


def run_with_watchdog(command: str, silence_timeout: float = SILENCE_TIMEOUT) -> tuple[bool, int, str]:
    """Run `command` in bash. Returns (timed_out, exit_code, combined stdout+stderr)."""
    proc = subprocess.Popen(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,    # own process group, so the kill takes children too
    )
    last_output = time.monotonic()
    timed_out = False

    def watchdog() -> None:
        nonlocal timed_out
        while proc.poll() is None:
            time.sleep(0.25)
            if time.monotonic() - last_output >= silence_timeout:
                timed_out = True
                _kill_group(proc)
                return

    threading.Thread(target=watchdog, daemon=True).start()

    lines = []
    for line in proc.stdout:
        last_output = time.monotonic()
        lines.append(line)
    exit_code = proc.wait()

    output = "".join(lines)
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n\n[Output truncated at {MAX_OUTPUT_CHARS} chars]"
    return timed_out, exit_code, output


class TerminalTool(BaseTool):
    name = "terminal"
    description = (
        f"Run a shell command on the user's system ({os_name()}): install packages, manage files, "
        "run scripts, check system state, control services. "
        "ALWAYS plan web_search and deep_think before it (to find the right command and reason "
        "about it), unless the step just runs a local file that file_search already found."
    )
    input_schema = (
        "One complete bash command or short pipeline, ready to run as-is. No placeholders, "
        "markdown or explanation. Example: 'sudo apt install -y ffmpeg'."
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, query: str, data: dict) -> ToolCallResult:
        command = clean_reply(query)
        if not command:
            return ToolCallResult(False, "No command was provided.")

        tier = classify_command(command, self.llm)
        log.info("tier=%s cmd=%r", tier, command)

        if tier == "IRREVERSIBLE":
            return ToolCallResult(
                False,
                f"Refused: `{command}`\n\nThis could permanently destroy data or system state, so AIDE "
                "never runs it. If you really need it, run it yourself in a terminal.",
            )

        if tier != "READ_ONLY" and not data.get("confirmed"):
            reason = "This will run arbitrary code." if tier == "SCRIPT_RUN" else "This will change your system."
            return ToolCallResult(
                False,
                "",
                requires_confirmation=True,
                confirmation_prompt=f"I'm about to run:\n\n```\n{command}\n```\n\n{reason} Proceed?",
            )

        timed_out, exit_code, output = run_with_watchdog(command)
        if timed_out:
            return ToolCallResult(
                False,
                f"Killed after {SILENCE_TIMEOUT}s without output.\n\nPartial output:\n{output or '(none)'}",
                {"command": command, "output": output, "timed_out": True},
            )

        status = "" if exit_code == 0 else f"\nExit code: {exit_code}"
        return ToolCallResult(
            exit_code == 0,
            f"Command: `{command}`{status}\n\nOutput:\n{output or '(no output)'}",
            {"command": command, "output": output, "exit_code": exit_code},
        )
