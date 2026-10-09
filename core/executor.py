"""
Runs one thread of a plan.

For each step the executor
  1. turns the outputs of the step's dependencies into this tool's input (an LLM "resolver"),
  2. runs the tool, pausing the thread if the tool asks for confirmation,
  3. asks an adaptive controller whether to continue, retry, inject steps or abort.

Threads are independent, so each gets its own ThreadRun and they can execute in parallel.
"""

import logging
from dataclasses import dataclass, field, replace

from core.context import SessionContext, ToolResult
from core.llm import LLM, clean_reply, extract_json_objects
from core.planner import PlanStep, Thread
from core.prompts import adapt_system, resolver_system, terminal_resolver_system
from tools.base import BaseTool, ToolCallResult

log = logging.getLogger("aide.executor")

MAX_RETRIES = 2
_NO_ADAPT = {"summarize", "deep_think", "chat", "open_file"}   # nothing to second-guess
_ACTIONS = {"continue", "retry", "inject", "abort"}

Completed = list[tuple[PlanStep, ToolCallResult]]


@dataclass
class ThreadRun:
    """Mutable state of one thread mid-execution. Kept so a paused thread can be resumed."""
    goal: str
    query: str                                   # the user's original request
    queue: list[PlanStep]                        # steps still to run
    completed: Completed = field(default_factory=list)
    outputs: dict[str, ToolCallResult] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)


class ExecutionError(Exception):
    """A step failed in a way that makes the rest of its thread pointless."""

    def __init__(self, step: PlanStep, result: ToolCallResult):
        super().__init__(result.output)
        self.step = step
        self.result = result


class ConfirmationRequired(Exception):
    """A tool wants a yes/no from the user. Holds everything needed to resume the thread."""

    def __init__(self, run: ThreadRun, step: PlanStep, step_input: str, step_data: dict, prompt: str):
        super().__init__(prompt)
        self.run = run
        self.step = step
        self.step_input = step_input
        self.step_data = step_data
        self.prompt = prompt


def parse_injected_steps(raw_steps, tools: dict[str, BaseTool], outputs: dict[str, ToolCallResult]) -> list[PlanStep]:
    """Turn the controller's new_steps into PlanSteps, dropping unknown tools and bad dependencies."""
    search_keys = [k for k, r in outputs.items() if r.success and r.data.get("results")]
    steps = []
    for i, raw in enumerate(raw_steps if isinstance(raw_steps, list) else []):
        if not isinstance(raw, dict):
            continue
        step = PlanStep.from_dict(raw, f"_inject_{i}")
        if step.tool not in tools:
            continue
        # Models sometimes write a tool name ("web_search") where an output key belongs.
        deps = [search_keys[-1] if d == "web_search" and search_keys else d for d in step.depends_on or []]
        step.depends_on = [d for d in deps if d in outputs] or None
        steps.append(step)
    return steps


class Executor:
    def __init__(self, tools: dict[str, BaseTool], llm: LLM):
        self.tools = tools
        self.llm = llm

    # ---- public API ----

    def execute_thread(self, thread: Thread, context: SessionContext, query: str = "") -> Completed:
        """Run every step of a thread. Raises ConfirmationRequired or ExecutionError."""
        return self._drive(ThreadRun(thread.goal, query, list(thread.steps)), context)

    def resume(self, pending: ConfirmationRequired, context: SessionContext) -> Completed:
        """Run the step the user just approved, then carry on with the rest of its thread."""
        run, step = pending.run, pending.step
        data = {**pending.step_data, "confirmed": True}
        result = self.tools[step.tool].run(pending.step_input, data)
        if result.requires_confirmation:
            raise ConfirmationRequired(run, step, pending.step_input, data, result.confirmation_prompt)

        context.add_tool_result(ToolResult(step.tool, pending.step_input, result.success, result.output, result.data))
        self._finish(run, step, result)
        return self._drive(run, context)

    # ---- main loop ----

    def _drive(self, run: ThreadRun, context: SessionContext) -> Completed:
        while run.queue:
            step = run.queue.pop(0)
            tool = self.tools[step.tool]
            step_data = context.snapshot_data()
            step_input = step.input

            if step.depends_on:
                missing = [k for k in step.depends_on if k not in run.outputs]
                if missing:
                    result = ToolCallResult(False, f"Missing dependencies: {missing}")
                    context.add_tool_result(ToolResult(step.tool, step_input, False, result.output))
                    self._finish(run, step, result)
                    continue
                deps = [run.outputs[k] for k in step.depends_on]
                for dep in deps:
                    step_data.update(dep.data)
                step_input = self._build_input(step, tool, deps, run, step_data)

            log.info("-> %s: %r", step.tool, step_input)
            result = tool.run(step_input, step_data)
            log.info("   success=%s", result.success)

            if result.requires_confirmation:
                raise ConfirmationRequired(run, step, step_input, step_data, result.confirmation_prompt)

            context.add_tool_result(ToolResult(step.tool, step_input, result.success, result.output, result.data))

            action = "continue"
            decision: dict = {}
            if step.tool not in _NO_ADAPT:
                decision = self._decide(step, result, run, context)
                action = decision["action"]
                if action == "retry" and self._queue_retry(step, decision, run):
                    continue

            self._finish(run, step, result, abort=(action == "abort"))
            if action == "inject":
                self._queue_injected(decision, run)

        return run.completed

    def _finish(self, run: ThreadRun, step: PlanStep, result: ToolCallResult, abort: bool = False) -> None:
        """Record a result. A failure only ends the thread if a later step needs its output."""
        run.outputs[step.output_key] = result
        run.completed.append((step, result))
        # summarize depends on everything but copes with failures, so it doesn't count.
        needed = any(step.output_key in (s.depends_on or []) and s.tool != "summarize" for s in run.queue)
        if abort or (not result.success and needed):
            raise ExecutionError(step, result)

    # ---- input resolution ----

    def _build_input(self, step: PlanStep, tool: BaseTool, deps: list[ToolCallResult], run: ThreadRun, data: dict) -> str:
        combined = "\n\n---\n\n".join(f"[From {key}]\n{dep.output}" for key, dep in zip(step.depends_on, deps))

        if step.tool == "summarize":
            # Passed through verbatim: the resolver would squash it into a short instruction.
            return f"Thread goal: {run.goal}\n\nTool outputs from this thread (summarize all of them):\n\n{combined}"
        if step.tool == "terminal":
            return self._resolve_command(run.goal, combined, data)

        prompt = (
            f'Thread goal: "{run.goal}"\n'
            f'User\'s original request: "{run.query or step.input}"\n\n'
            f"Output(s) from previous tool(s) in this thread:\n{combined}\n\n"
            f"Next tool: {tool.name}\nWhat it expects: {tool.input_schema}\n\n"
        )
        if tool.name == "open_url" and data.get("attempted_urls"):
            prompt += f"URLs already attempted (do NOT pick these): {data['attempted_urls']}\n\n"
        prompt += f"Produce exactly the input string for {tool.name}."

        resolved = clean_reply(self.llm.ask(prompt, system=resolver_system(), temperature=0.0, max_tokens=256))
        log.info("resolved input for %s: %r", tool.name, resolved)
        return resolved

    def _resolve_command(self, goal: str, context_text: str, data: dict) -> str:
        facts = []
        if data.get("path"):
            facts.append(f"Known file path: {data['path']}")
        facts += [f"Available path: {p}" for p in data.get("all_paths", [])[:3] if p != data.get("path")]

        if len(context_text) > 3000:
            context_text = context_text[:2000] + "\n...\n" + context_text[-500:]

        prompt = (
            ("CONCRETE FACTS FROM EARLIER STEPS:\n" + "\n".join(facts) + "\n\n" if facts else "")
            + f'Thread goal: "{goal}"\n'
            + f"Context from previous steps in this thread:\n{context_text}\n\n"
            + "Output the single bash command to run."
        )
        reply = clean_reply(self.llm.ask(prompt, system=terminal_resolver_system(), temperature=0.0, max_tokens=128))
        command = next((line.strip() for line in reply.splitlines() if line.strip()), reply)
        log.info("resolved command: %r", command)
        return command

    # ---- adaptive control ----

    def _decide(self, step: PlanStep, result: ToolCallResult, run: ThreadRun, context: SessionContext) -> dict:
        """Ask the LLM whether to continue, retry, inject new steps, or abort. Always has an "action"."""
        finished = {**run.outputs, step.output_key: result}
        attempted = set(context.snapshot_data().get("attempted_urls", []))
        candidates = list(dict.fromkeys(
            hit["url"]
            for r in finished.values() if r.success
            for hit in r.data.get("results", [])
            if hit.get("url") and hit["url"] not in attempted
        ))
        search_keys = [k for k, r in finished.items() if r.success and r.data.get("results")]

        prompt = (
            f'Thread goal: "{run.goal}"\n'
            f'Original user request: "{run.query}"\n\n'
            f"Step that just ran:\n"
            f"  tool: {step.tool}\n  input: {step.input!r}\n  success: {result.success}\n"
            f"  output (first 600 chars): {result.output[:600]!r}\n\n"
            f"Retries used on this step: {run.retries.get(step.output_key, 0)} (max {MAX_RETRIES})\n"
            f"Remaining planned steps: {[s.tool for s in run.queue]}\n"
            f"Existing output keys (new keys must not clash): {list(finished)}\n"
            f"Available tools: {list(self.tools)}\n"
            f"Web search output keys (use as depends_on for injected open_url steps): {search_keys}\n"
            f"URL candidates not yet attempted: {candidates[:8]}\n\n"
            f"Decide: continue, retry, inject, or abort?"
        )
        reply = self.llm.ask(prompt, system=adapt_system(), temperature=0.0, max_tokens=512)
        decision = next(extract_json_objects(reply), {})
        if decision.get("action") not in _ACTIONS:
            decision = {"action": "continue", "reason": "no usable decision"}
        log.info("adaptive: %s (%s)", decision["action"], decision.get("reason", ""))
        return decision

    def _queue_retry(self, step: PlanStep, decision: dict, run: ThreadRun) -> bool:
        """Re-queue the step with a new input. Returns False if the retry isn't worth doing."""
        used = run.retries.get(step.output_key, 0)
        new_input = decision.get("retry_input")
        if used >= MAX_RETRIES or not new_input or new_input == step.input:
            return False
        run.retries[step.output_key] = used + 1
        run.queue.insert(0, replace(step, input=str(new_input)))
        log.info("retrying %s with %r", step.tool, new_input)
        return True

    def _queue_injected(self, decision: dict, run: ThreadRun) -> None:
        new_steps = parse_injected_steps(decision.get("new_steps"), self.tools, run.outputs)
        if new_steps:
            log.info("injecting %d step(s)", len(new_steps))
            run.queue[0:0] = new_steps
