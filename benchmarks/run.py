"""
Benchmarks for the real models configured in config/settings.yaml (chat server + embedder).

    python -m benchmarks.run                       # every section
    python -m benchmarks.run planner safety        # just these
    python -m benchmarks.run --json results.json   # also save the numbers

Each section reports quality (did the model do the right thing?) and speed (how long, how many
tokens, how many model calls). Tools are replaced by stubs, so nothing touches your files or the
web: only the models are real. Cases live in benchmarks/cases.py.
"""

import argparse
import json
import math
import re
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

from benchmarks import cases
from core.config import load_config
from core.context import SessionContext
from core.executor import ConfirmationRequired, ExecutionError, Executor
from core.llm import LLM
from core.orchestrator import Orchestrator
from core.planner import PlanStep, Planner, Thread
from tools.base import BaseTool, ToolCallResult
from tools.chat import ChatTool
from tools.deep_think import DeepThinkTool
from tools.file_search import FileSearchTool
from tools.open_file import OpenFileTool
from tools.open_url import OpenUrlTool
from tools.read_file import ReadFileTool
from tools.summarize import SummarizeTool
from tools.terminal import TerminalTool, classify_command
from tools.web_search import WebSearchTool

TOOL_CLASSES = [FileSearchTool, OpenFileTool, ReadFileTool, TerminalTool, ChatTool, WebSearchTool,
                OpenUrlTool, DeepThinkTool, SummarizeTool]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class Section:
    metrics: dict
    rows: list[tuple[str, bool | None, str]] = field(default_factory=list)   # (label, ok or None=info, detail)


class MeteredLLM(LLM):
    """An LLM that records every call, so sections can report latency, tokens and call counts."""

    def __init__(self, base_url: str, model: str, timeout: float = 300):
        super().__init__(base_url, model, timeout)
        self.records: list[dict] = []
        self._lock = threading.Lock()

    def complete(self, messages, **kwargs):
        start = time.perf_counter()
        text, stats = super().complete(messages, **kwargs)
        record = {
            "seconds": time.perf_counter() - start,
            "completion_tokens": stats["usage"].get("completion_tokens", 0),
            "stats": stats,
        }
        with self._lock:
            self.records.append(record)
        return text, stats


class StubTool(BaseTool):
    """Looks like a real tool to the planner (same name and description) but returns canned output."""

    def __init__(self, source, output: str = "", data: dict | None = None, success: bool = True):
        self.name, self.description, self.input_schema = source.name, source.description, source.input_schema
        self.is_conversational = source.is_conversational
        self.output, self.data, self.success = output, data or {}, success
        self.calls: list[str] = []

    def run(self, query: str, data: dict) -> ToolCallResult:
        self.calls.append(query)
        return ToolCallResult(self.success, self.output, self.data)


def stub_catalog(**replacements: BaseTool) -> dict[str, BaseTool]:
    tools: dict[str, BaseTool] = {cls.name: StubTool(cls) for cls in TOOL_CLASSES}
    tools.update(replacements)
    return tools


def run_thread(llm: LLM, goal: str, query: str, steps: list[PlanStep], tools: dict) -> str | None:
    """Execute a hand-built thread with the real executor. Returns an error message, or None."""
    try:
        Executor(tools, llm).execute_thread(Thread(goal, steps), SessionContext(), query)
    except (ExecutionError, ConfirmationRequired) as e:
        return f"thread stopped early: {e}"
    return None


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def bench_chat(llm: MeteredLLM, embedder: LLM, repeats: int = 3) -> Section:
    """Raw speed of the chat model on a fixed prompt."""
    latencies, tokens, text, last = [], [], "", {}
    for _ in range(repeats):
        mark = len(llm.records)
        start = time.perf_counter()
        text = llm.ask("Explain what a mutex is in about 80 words.", max_tokens=300, temperature=0.2)
        latencies.append(time.perf_counter() - start)
        last = llm.records[mark]
        tokens.append(last["completion_tokens"])

    metrics = {"median_latency_s": _median(latencies), "completion_tokens": round(_median(tokens))}
    if metrics["completion_tokens"]:
        metrics["tokens_per_s"] = metrics["completion_tokens"] / _median(latencies)   # includes prompt processing
    timings = last["stats"]["timings"]
    for key, label in (("predicted_per_second", "server_generation_tok_s"), ("prompt_per_second", "server_prompt_tok_s")):
        if key in timings:
            metrics[label] = timings[key]
    return Section(metrics, [("reply is non-empty", bool(text), text[:60].replace("\n", " "))])


def bench_thinking(llm: MeteredLLM, embedder: LLM, repeats: int = 3) -> Section:
    """Does chat_template_kwargs.enable_thinking change anything for this model? (Informational.)"""
    counts = {}
    for think in (False, True):
        mark = len(llm.records)
        llm.ask("What is 17 * 23? Answer with just the number.", think=think, max_tokens=800, temperature=0.0)
        counts[think] = llm.records[mark]["completion_tokens"]

    off, on = counts[False], counts[True]
    if not (off or on):
        verdict = "server reported no token usage"
    elif on > 2 * off + 10:
        verdict = "toggle has a visible effect"
    else:
        verdict = "no visible effect (this model may have no thinking mode, or the template ignores the flag)"
    return Section({"tokens_think_off": off, "tokens_think_on": on}, [("thinking toggle", None, verdict)])


def _try_plan(planner: Planner, case: cases.PlannerCase, think: bool):
    """One planning attempt. Returns (seconds, valid, problems, code_added_steps, tools_used)."""
    start = time.perf_counter()
    raw = planner.ask(case.query, SessionContext(), think=think)
    seconds = time.perf_counter() - start
    try:
        plan = planner.parse(raw, case.query)
    except (AttributeError, TypeError, ValueError) as e:
        return seconds, False, [f"invalid plan: {e}"], False, set()

    used = {s.tool for s in plan.steps} - {"summarize"}
    problems = []
    if missing := case.must_use - used:
        problems.append(f"missing {sorted(missing)}")
    if banned := case.must_not_use & used:
        problems.append(f"forbidden {sorted(banned)}")
    if not case.min_threads <= len(plan.threads) <= case.max_threads:
        problems.append(f"{len(plan.threads)} threads")
    return seconds, True, problems, any(s.output_key.startswith("_auto_") for s in plan.steps), used


def bench_planner(llm: MeteredLLM, embedder: LLM, repeats: int = 2) -> Section:
    """
    Valid JSON, right tools, right thread structure. Each case runs `repeats` times (to show
    consistency), then once more with thinking off to see what reasoning buys.
    """
    planner = Planner(stub_catalog(), llm)
    rows, times = [], []
    valid = passed = injected = consistent = 0

    for case in cases.PLANNER_CASES:
        runs = [_try_plan(planner, case, think=True) for _ in range(repeats)]
        good = [r for r in runs if r[1] and not r[2]]
        times += [r[0] for r in runs]
        valid += sum(r[1] for r in runs)
        passed += len(good)
        injected += sum(r[3] for r in runs)
        consistent += len(good) == repeats

        first_bad = next((r for r in runs if r[2]), None)
        note = "; ".join(first_bad[2]) if first_bad else "tools: " + ", ".join(sorted(runs[-1][4]))
        rows.append((case.query, len(good) == repeats, f"{len(good)}/{repeats} runs ok. {note}"))

    quick = [_try_plan(planner, case, think=False) for case in cases.PLANNER_CASES]
    n, total = len(cases.PLANNER_CASES), len(cases.PLANNER_CASES) * repeats
    return Section(
        {"valid_json_rate": valid / total, "pass_rate": passed / total, "consistent_case_rate": consistent / n,
         "median_plan_s": _median(times), "plans_where_code_added_steps": injected,
         "no_think_pass_rate": sum(r[1] and not r[2] for r in quick) / n,
         "no_think_median_plan_s": _median([r[0] for r in quick])},
        rows,
    )


def bench_resolution(llm: MeteredLLM, embedder: LLM, repeats: int = 5) -> Section:
    """Can the resolver turn earlier tool output into the right file path or shell command?"""
    rows, picked, commanded = [], 0, 0

    for case in cases.PICK_FILE_CASES:
        listing, data = cases.file_listing(case.paths)
        hits, got = 0, None
        for _ in range(repeats):
            opener = StubTool(OpenFileTool)
            tools = stub_catalog(file_search=StubTool(FileSearchTool, listing, data), open_file=opener)
            steps = [PlanStep("file_search", case.search_keywords, "fs"), PlanStep("open_file", "open the file", "of", ["fs"])]
            error = run_thread(llm, case.request, case.request, steps, tools)
            got = opener.calls[0] if opener.calls else error
            hits += got == case.expected
        picked += hits
        rows.append((f"pick file: {case.request}", hits == repeats, f"{hits}/{repeats}" + ("" if hits == repeats else f", last got {got!r}")))

    by_name = {cls.name: cls for cls in TOOL_CLASSES}
    for case in cases.COMMAND_CASES:
        hits, got = 0, None
        for _ in range(repeats):
            terminal = StubTool(TerminalTool)
            tools = stub_catalog(terminal=terminal)
            steps = []
            for i, (tool_name, evidence) in enumerate(zip(case.evidence_tools, case.evidence)):
                data = case.file_search_data if tool_name == "file_search" else None
                tools[tool_name] = StubTool(by_name[tool_name], evidence, data)
                steps.append(PlanStep(tool_name, "evidence", f"e{i}"))
            steps.append(PlanStep("terminal", "run it", "term", [s.output_key for s in steps]))
            error = run_thread(llm, case.goal, case.goal, steps, tools)
            got = terminal.calls[0] if terminal.calls else error
            hits += bool(got) and re.search(case.pattern, got.strip()) is not None
        commanded += hits
        rows.append((f"command: {case.goal}", hits == repeats, f"{hits}/{repeats}, last: {got}"))

    return Section(
        {"file_pick_accuracy": picked / (len(cases.PICK_FILE_CASES) * repeats),
         "command_accuracy": commanded / (len(cases.COMMAND_CASES) * repeats)},
        rows,
    )


def bench_confirmation(llm: MeteredLLM, embedder: LLM, repeats: int = 5) -> Section:
    """Does the model read yes / no / unclear replies correctly?"""
    bot = Orchestrator(llm, {})
    rows, correct, times = [], 0, []
    for reply, expected in cases.CONFIRM_CASES:
        verdicts = []
        for _ in range(repeats):
            start = time.perf_counter()
            verdicts.append(bot.classify_confirmation(reply))
            times.append(time.perf_counter() - start)
        hits = verdicts.count(expected)
        correct += hits
        wrong = Counter(v for v in verdicts if v != expected)
        rows.append((repr(reply), hits == repeats, f"{hits}/{repeats} said {expected}" + (f" (also: {dict(wrong)})" if wrong else "")))
    return Section({"accuracy": correct / (len(cases.CONFIRM_CASES) * repeats), "median_latency_s": _median(times)}, rows)


def bench_safety(llm: MeteredLLM, embedder: LLM, repeats: int = 5) -> Section:
    """Fallback command classifier: dangerous commands must never be called READ_ONLY."""
    rows, leaks, harmless_ok = [], 0, 0
    for command in cases.DANGEROUS_COMMANDS:
        tiers = Counter(classify_command(command, llm) for _ in range(repeats))
        leaks += tiers["READ_ONLY"]
        rows.append((command, tiers["READ_ONLY"] == 0, ", ".join(f"{t} x{n}" for t, n in tiers.items())))
    for command in cases.HARMLESS_COMMANDS:
        tiers = Counter(classify_command(command, llm) for _ in range(repeats))
        harmless_ok += tiers["READ_ONLY"]
        rows.append((command, None, ", ".join(f"{t} x{n}" for t, n in tiers.items()) + " (harmless; READ_ONLY avoids a needless prompt)"))
    return Section(
        {"dangerous_misread_as_read_only": leaks,
         "harmless_recognised_rate": harmless_ok / (len(cases.HARMLESS_COMMANDS) * repeats)},
        rows,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def bench_embeddings(llm: MeteredLLM, embedder: LLM, repeats: int = 3) -> Section:
    """Embedding speed, plus top-1 retrieval accuracy on a tiny synthetic file collection."""
    times = []

    def embed(text: str) -> list[float]:
        start = time.perf_counter()
        vector = embedder.embed(text)
        times.append(time.perf_counter() - start)
        return vector

    docs = {name: embed(text) for name, text in cases.DOCS.items()}
    rows, correct, dims = [], 0, 0
    for query, expected in cases.RETRIEVAL_CASES:
        vector = embed(query)
        dims = len(vector)
        best = max(docs, key=lambda name: _cosine(vector, docs[name]))
        correct += best == expected
        rows.append((query, best == expected, f"top hit {best}" + ("" if best == expected else f", wanted {expected}")))
    return Section(
        {"top1_accuracy": correct / len(cases.RETRIEVAL_CASES), "median_embed_s": _median(times), "dimensions": dims},
        rows,
    )


def bench_turns(llm: MeteredLLM, embedder: LLM, repeats: int = 3) -> Section:
    """Whole turns through the real orchestrator: how long, and how many model calls do they take?"""
    weather = StubTool(
        WebSearchTool,
        "Web search results for 'weather Rome today':\n\n1. Rome weather\n   Sunny, 24C, light breeze.\n   Source: https://weather.example/rome",
        {"results": [{"title": "Rome weather", "snippet": "Sunny, 24C, light breeze.", "url": "https://weather.example/rome"}],
         "query": "weather Rome today"},
    )
    tools = stub_catalog(web_search=weather, summarize=SummarizeTool(llm), chat=ChatTool())

    rows, seconds, calls = [], [], []
    for query in cases.TURN_CASES:
        mark = len(llm.records)
        start = time.perf_counter()
        reply = Orchestrator(llm, tools).handle(query)
        seconds.append(time.perf_counter() - start)
        made = llm.records[mark:]
        calls.append(len(made))
        rows.append((query, bool(reply.strip()),
                     f"{seconds[-1]:.1f}s, {len(made)} model calls, {sum(r['completion_tokens'] for r in made)} tokens"))
    return Section({"median_turn_s": _median(seconds), "max_model_calls": max(calls)}, rows)


SECTIONS: dict[str, Callable[[MeteredLLM, LLM, int], Section]] = {
    "chat": bench_chat,
    "thinking": bench_thinking,
    "planner": bench_planner,
    "resolution": bench_resolution,
    "confirmation": bench_confirmation,
    "safety": bench_safety,
    "embeddings": bench_embeddings,
    "turns": bench_turns,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(key: str, value) -> str:
    if isinstance(value, float):
        return f"{value:.0%}" if key.endswith(("rate", "accuracy")) else f"{value:.2f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the models configured for AIDE.")
    parser.add_argument("sections", nargs="*", choices=list(SECTIONS), help="default: all")
    parser.add_argument("--json", metavar="PATH", help="also write the metrics as JSON")
    parser.add_argument("--repeats", type=int, default=5, help="runs per case for chat, planner, resolution, confirmation and safety (default 5)")
    args = parser.parse_args(argv)

    cfg = load_config()
    llm = MeteredLLM.from_config(cfg["llm"])
    embedder = LLM.from_config(cfg["embedder"])
    print(f"chat:     {llm.model} @ {llm.base_url}\nembedder: {embedder.model} @ {embedder.base_url}")
    for label, client, probe in (("chat", llm, lambda c: c.ask("ping", max_tokens=5)), ("embedder", embedder, lambda c: c.embed("ping"))):
        try:
            probe(LLM(client.base_url, client.model, timeout=5))
        except requests.RequestException as e:
            print(f"\nCan't reach the {label} server ({type(e).__name__}). Check config/settings.yaml and that it's running.")
            return 2

    results: dict[str, dict] = {}
    for name in args.sections or SECTIONS:
        print(f"\n== {name}")
        start = time.perf_counter()
        try:
            section = SECTIONS[name](llm, embedder, args.repeats)
        except requests.RequestException as e:
            print(f"  server error: {e}")
            results[name] = {"error": str(e)}
            continue
        for label, ok, detail in section.rows:
            print(f"  {'·' if ok is None else '✓' if ok else '✗'} {label}: {detail}")
        for key, value in section.metrics.items():
            print(f"  → {key}: {_fmt(key, value)}")
        print(f"  ({time.perf_counter() - start:.1f}s)")
        results[name] = section.metrics

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"chat_model": llm.model, "embedder": embedder.model, "results": results}, f, indent=2)
        print(f"\nsaved {args.json}")
    return 1 if any("error" in r for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())