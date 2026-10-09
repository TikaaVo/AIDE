import pytest

from conftest import FakeLLM
from tools import terminal
from tools.terminal import TerminalTool, classify_command, run_with_watchdog


@pytest.mark.parametrize("command, tier", [
    ("ls -la ~/Documents", "READ_ONLY"),
    ("ps aux | grep python", "READ_ONLY"),
    ("git status", "READ_ONLY"),
    ("python3 --version", "READ_ONLY"),
    ("python3 script.py", "SCRIPT_RUN"),
    ("./build.sh", "SCRIPT_RUN"),
    ("rm -rf /", "IRREVERSIBLE"),
    ("rm -rf ~", "IRREVERSIBLE"),
    ("sudo shutdown now", "IRREVERSIBLE"),
    ("dd if=/dev/zero of=/dev/sda", "IRREVERSIBLE"),
    (":(){ :|:& };:", "IRREVERSIBLE"),
])
def test_regex_tiers_never_need_the_llm(command, tier):
    llm = FakeLLM({})   # any LLM call would raise
    assert classify_command(command, llm) == tier


@pytest.mark.parametrize("command", [
    "ls; rm notes.txt",           # chaining a mutation onto a read-only command
    "cat a.txt > b.txt",          # redirect writes a file
    "echo $(whoami)",             # command substitution
    "find . -name '*.tmp' -delete",
])
def test_read_only_lookalikes_are_not_trusted(command):
    llm = FakeLLM({"classify shell commands": "REVERSIBLE"})
    assert classify_command(command, llm) != "READ_ONLY"


def test_llm_fallback_and_the_irreversible_substring_trap():
    assert classify_command("mkdir foo", FakeLLM({"classify shell": "REVERSIBLE"})) == "REVERSIBLE"
    assert classify_command("weird thing", FakeLLM({"classify shell": "IRREVERSIBLE"})) == "IRREVERSIBLE"
    assert classify_command("weird thing", FakeLLM({"classify shell": "no idea"})) == "REVERSIBLE"


def test_irreversible_is_refused_and_nothing_executes(monkeypatch):
    monkeypatch.setattr(terminal, "run_with_watchdog", lambda *a, **k: pytest.fail("must not execute"))
    result = TerminalTool(FakeLLM({})).run("rm -rf /", {})
    assert not result.success and "Refused" in result.output


def test_scripts_wait_for_confirmation_before_running(monkeypatch):
    monkeypatch.setattr(terminal, "run_with_watchdog", lambda *a, **k: pytest.fail("must not execute"))
    result = TerminalTool(FakeLLM({})).run("python3 thing.py", {})
    assert result.requires_confirmation and "python3 thing.py" in result.confirmation_prompt


def test_confirmed_command_runs_for_real():
    result = TerminalTool(FakeLLM({})).run("python3 -c 'print(40 + 2)'", {"confirmed": True})
    assert result.success and "42" in result.output


def test_read_only_command_runs_immediately_and_reports_exit_code():
    ok = TerminalTool(FakeLLM({})).run("echo hello", {})
    assert ok.success and "hello" in ok.output
    bad = TerminalTool(FakeLLM({})).run("ls /definitely/not/here", {})
    assert not bad.success and "Exit code" in bad.output


def test_watchdog_kills_a_silent_process():
    timed_out, _, output = run_with_watchdog("echo started; sleep 30", silence_timeout=0.5)
    assert timed_out and "started" in output
