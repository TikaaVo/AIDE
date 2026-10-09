"""Turns a user request into a Plan: independent threads of tool steps."""

import logging
from dataclasses import dataclass, field

from core.context import SessionContext
from core.llm import LLM, extract_json_objects
from core.prompts import planner_system
from tools.base import BaseTool

log = logging.getLogger("aide.planner")


@dataclass
class PlanStep:
    tool: str
    input: str
    output_key: str
    depends_on: list[str] | None = None

    @classmethod
    def from_dict(cls, raw: dict, default_key: str) -> "PlanStep":
        dep = raw.get("depends_on")
        deps = [dep] if isinstance(dep, str) else list(dep or [])
        text = raw.get("input", "")
        if isinstance(text, dict):   # models sometimes nest the input as {"query": "..."}
            text = " ".join(map(str, text.values()))
        return cls(
            tool=str(raw.get("tool", "")),
            input=str(text),
            output_key=str(raw.get("output_key", default_key)),
            depends_on=[str(d) for d in deps] or None,
        )


@dataclass
class Thread:
    """One independent sub-task: a goal plus the ordered steps that achieve it."""
    goal: str
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Plan:
    threads: list[Thread] = field(default_factory=list)

    @property
    def steps(self) -> list[PlanStep]:
        return [s for t in self.threads for s in t.steps]


# ---------------------------------------------------------------------------
# Post-processing: fix up the model's plan before anything runs
# ---------------------------------------------------------------------------

def dedup_steps(steps: list[PlanStep]) -> list[PlanStep]:
    """
    Drop steps that repeat an earlier (tool, input) pair, rewiring dependents to the survivor.
    open_url is exempt: its planned inputs are placeholders that look identical but each
    resolves to a different URL at run time.
    """
    seen: dict[tuple[str, str], str] = {}
    renamed: dict[str, str] = {}
    kept = []
    for step in steps:
        signature = (step.tool, " ".join(step.input.lower().split()))
        if step.tool != "open_url" and signature in seen:
            renamed[step.output_key] = seen[signature]
            log.info("dropped duplicate %s step %r", step.tool, step.output_key)
            continue
        seen.setdefault(signature, step.output_key)
        kept.append(step)

    for step in kept:
        if step.depends_on:
            step.depends_on = list(dict.fromkeys(renamed.get(k, k) for k in step.depends_on))
    return kept


_RUN_WORDS = ("run", "execute", "launch", "start", "compile", "build", "python", "node", "ruby", "bash",
              "g++", "gcc", "java", ".py", ".cpp", ".js", ".rb", ".sh")
_ADMIN_WORDS = ("install", "apt", "pip", "brew", "sudo", "package", "dependenc", "update", "upgrade",
                "service", "daemon", "systemctl")


def _ancestors(step: PlanStep, by_key: dict[str, PlanStep]) -> set[str]:
    seen: set[str] = set()
    stack = list(step.depends_on or [])
    while stack:
        key = stack.pop()
        if key not in seen:
            seen.add(key)
            if key in by_key:
                stack.extend(by_key[key].depends_on or [])
    return seen


def _runs_located_file(step: PlanStep, earlier: list[PlanStep], goal: str) -> bool:
    """
    True when a terminal step only runs a file that file_search already found. The planned
    input is still a placeholder ("run blackjack script"), so go by intent: run-ish words
    and no install/admin words.
    """
    if not any(s.tool in ("file_search", "read_file") for s in earlier):
        return False
    text = f"{goal} {step.input}".lower()
    return any(w in text for w in _RUN_WORDS) and not any(w in text for w in _ADMIN_WORDS)


def enforce_terminal_prereqs(steps: list[PlanStep], goal: str, query: str) -> list[PlanStep]:
    """
    Make sure every terminal step is preceded by web_search (look up the right syntax) and
    deep_think (reason about the command), wiring in existing steps or injecting new ones.
    Skipped when the step just runs a local file we've already located.
    """
    out: list[PlanStep] = []
    taken = {s.output_key for s in steps}
    task = goal or query

    def fresh_key(base: str) -> str:
        key, n = base, 0
        while key in taken:
            n += 1
            key = f"{base}_{n}"
        taken.add(key)
        return key

    for step in steps:
        if step.tool == "terminal" and not _runs_located_file(step, out, goal):
            by_key = {s.output_key: s for s in out}
            ancestor_tools = {by_key[k].tool for k in _ancestors(step, by_key) if k in by_key}
            deps = list(step.depends_on or [])

            if "web_search" not in ancestor_tools:
                search = next((s for s in reversed(out) if s.tool == "web_search"), None)
                if search is None:
                    search = PlanStep("web_search", f"how to {task} on Linux", fresh_key("_auto_ws"))
                    out.append(search)
                    log.info("injected web_search before terminal")
                deps.append(search.output_key)

            if "deep_think" not in ancestor_tools:
                think = next((s for s in reversed(out) if s.tool == "deep_think"), None)
                if think is None:
                    searches = [s.output_key for s in out if s.tool == "web_search"]
                    think = PlanStep(
                        "deep_think",
                        f"Goal: {task}. Given the search results and any file paths found, state the "
                        "single exact bash command to run. Only the command, no strategy.",
                        fresh_key("_auto_dt"),
                        searches or None,
                    )
                    out.append(think)
                    log.info("injected deep_think before terminal")
                deps.append(think.output_key)

            step.depends_on = list(dict.fromkeys(deps)) or None
        out.append(step)
    return out


def add_summary_step(thread: Thread) -> None:
    """
    Close the thread with a summarize step over everything it produced, so each thread
    reports back one digest. Chat-only threads are answered directly and skip this.
    """
    thread.steps = [s for s in thread.steps if s.tool != "summarize"]
    if not thread.steps or all(s.tool == "chat" for s in thread.steps):
        return
    thread.steps.append(PlanStep(
        tool="summarize",
        input=f"Thread goal: {thread.goal}",
        output_key=f"{thread.steps[-1].output_key}_summary",
        depends_on=[s.output_key for s in thread.steps],
    ))


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    def __init__(self, tools: dict[str, BaseTool], llm: LLM):
        self.tools = tools
        self.llm = llm

    def _prompt(self, query: str, context: SessionContext) -> str:
        # summarize is added automatically, so the planner never needs to know about it.
        menu = "\n".join(f"- {name}: {t.description}" for name, t in self.tools.items() if name != "summarize")
        paths = context.data.get("all_paths")
        sections = [
            f"Available tools:\n{menu}",
            "Files in context (open_file can use these directly):\n" + "\n".join(f"  - {p}" for p in paths)
            if paths else "Files in context: none (open_file needs file_search first).",
        ]
        if history := context.history_text(6):
            sections.append(f"Recent conversation:\n{history}")
        if prior := context.tool_results_text():
            sections.append(f"Data already retrieved:\n{prior}")
        if web := context.web_search_context():
            sections.append(f"Web search results already in context:\n{web}")
        sections.append(f'User request: "{query}"\n\nRespond with the JSON plan only.')
        return "\n\n".join(sections)

    def ask(self, query: str, context: SessionContext, think: bool = True) -> str:
        """One planning call. Returns the model's raw reply."""
        messages = [
            {"role": "system", "content": planner_system()},
            {"role": "user", "content": self._prompt(query, context)},
        ]
        return self.llm.chat(messages, temperature=0.2, max_tokens=16384, think=think)

    def plan(self, query: str, context: SessionContext) -> Plan:
        for attempt in (1, 2):
            raw = self.ask(query, context)
            try:
                return self.parse(raw, query)
            except (AttributeError, TypeError, ValueError) as e:   # model output is untrusted
                log.warning("unusable plan (attempt %d): %s\n%s", attempt, e, raw)
        return Plan([Thread(query, [PlanStep("chat", query, "t1_response")])])

    def parse(self, raw: str, query: str) -> Plan:
        """Turn a raw model reply into a fixed-up Plan. Raises ValueError if it has no usable threads."""
        raw_threads: list[dict] = []
        for obj in extract_json_objects(raw):   # tolerates several {"threads": ...} objects in a row
            if "threads" in obj:
                raw_threads += obj["threads"]
            elif "steps" in obj:                # flat plan with no threads
                raw_threads.append({"goal": query, "steps": obj["steps"]})

        threads = []
        for n, t in enumerate(raw_threads, 1):
            goal = str(t.get("goal", query))
            steps = [PlanStep.from_dict(s, f"t{n}_step{i}") for i, s in enumerate(t.get("steps", []))]
            steps = [s for s in steps if s.tool in self.tools]
            steps = enforce_terminal_prereqs(dedup_steps(steps), goal, query)
            if steps:
                thread = Thread(goal, steps)
                add_summary_step(thread)
                threads.append(thread)

        if not threads:
            raise ValueError("no usable threads in plan")
        return Plan(threads)
