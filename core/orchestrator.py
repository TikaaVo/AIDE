"""
The conversation loop.

  1. acknowledge the request while the planner builds a plan (threads of tool steps)
  2. run the threads in parallel
  3. if a tool needs a yes/no, pause the turn and ask; resume on the next message
  4. synthesize one final answer from the per-thread summaries
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from core.context import SessionContext
from core.executor import Completed, ConfirmationRequired, ExecutionError, Executor
from core.llm import LLM
from core.planner import Planner, Thread
from core.prompts import CONFIRM, personality
from tools.base import BaseTool

log = logging.getLogger("aide.orchestrator")

HISTORY_WINDOW = 8
MAX_UNCLEAR_REPLIES = 3


@dataclass
class ThreadResult:
    thread: Thread
    completed: Completed
    summary: str = ""
    error: str = ""      # set when the thread failed or the user cancelled it
    index: int = 0       # position in the plan


@dataclass
class _Pending:
    """A turn paused until the user answers a yes/no for one or more threads."""
    query: str
    voice_mode: bool
    done: dict[int, ThreadResult]
    waiting: list[tuple[int, Thread, ConfirmationRequired]]   # oldest first
    unclear_replies: int = 0


def verified_urls(results: list[ThreadResult]) -> list[str]:
    """URLs that open_url actually fetched. Only these are safe to give to the user."""
    urls: list[str] = []
    for r in results:
        for step, res in r.completed:
            url = res.data.get("url")
            if step.tool == "open_url" and res.success and url and url not in urls:
                urls.append(url)
    return urls


class Orchestrator:
    def __init__(self, llm: LLM, tools: dict[str, BaseTool], speaker=None):
        """`speaker` is anything with say(text, block=False); None keeps AIDE silent."""
        self.llm = llm
        self.tools = tools
        self.planner = Planner(tools, llm)
        self.executor = Executor(tools, llm)
        self.speaker = speaker
        self.context = SessionContext()
        self._pending: _Pending | None = None

    # ---- entry point ----

    def handle(
        self,
        query: str,
        update_callback: Callable[[str], None] | None = None,
        voice_mode: bool = False,
    ) -> str:
        """Process one user message and return the reply. Interim messages go to update_callback."""
        log.info("user: %r (voice=%s)", query, voice_mode)
        notify = update_callback or (lambda _msg: None)

        if self._pending:
            voice_mode = self._pending.voice_mode
            reply = self._answer_confirmation(query, notify)
        else:
            reply = self._run_turn(query, notify, voice_mode)

        self.context.add_turn("user", query)
        self.context.add_turn("assistant", reply)
        self._say(voice_mode, reply, block=True)
        return reply

    def reset(self) -> None:
        self.context = SessionContext()
        self._pending = None
        log.info("session reset")

    # ---- a normal turn ----

    def _run_turn(self, query: str, notify: Callable[[str], None], voice_mode: bool) -> str:
        with ThreadPoolExecutor(max_workers=2) as pool:
            ack_future = pool.submit(self._acknowledge, query, voice_mode)
            plan_future = pool.submit(self.planner.plan, query, self.context)
            ack = ack_future.result()
            notify(ack)
            self._say(voice_mode, ack)
            plan = plan_future.result()
        log.info("plan: %s", [[s.tool for s in t.steps] for t in plan.threads])

        if all(self.tools[s.tool].is_conversational for s in plan.steps):
            return self._chat(query, voice_mode)

        done, waiting = self._run_threads(list(enumerate(plan.threads)), query)
        return self._advance(_Pending(query, voice_mode, done, waiting))

    def _run_threads(
        self, items: list[tuple[int, Thread]], query: str
    ) -> tuple[dict[int, ThreadResult], list[tuple[int, Thread, ConfirmationRequired]]]:
        """Run threads in parallel. Returns finished results and the threads paused for confirmation."""

        def work(item: tuple[int, Thread]):
            index, thread = item
            try:
                completed = self.executor.execute_thread(thread, self.context, query)
            except ConfirmationRequired as paused:
                return index, thread, paused
            except ExecutionError as e:
                return index, thread, ThreadResult(thread, [], error=str(e), index=index)
            return index, thread, self._thread_result(thread, completed, index)

        done: dict[int, ThreadResult] = {}
        waiting: list[tuple[int, Thread, ConfirmationRequired]] = []
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            for index, thread, outcome in pool.map(work, items):
                if isinstance(outcome, ConfirmationRequired):
                    waiting.append((index, thread, outcome))
                else:
                    done[index] = outcome
        return done, waiting

    @staticmethod
    def _thread_result(thread: Thread, completed: Completed, index: int) -> ThreadResult:
        summary = next((r.output for s, r in completed if s.tool == "summarize" and r.success), "")
        return ThreadResult(thread, completed, summary, index=index)

    def _advance(self, pending: _Pending) -> str:
        """Ask the next pending confirmation, or finish the turn if nothing is waiting."""
        if pending.waiting:
            self._pending = pending
            prompt = pending.waiting[0][2].prompt
            return f"⚠️ Before I continue:\n\n{prompt}\n\nReply yes to proceed or no to cancel."
        self._pending = None
        results = [pending.done[i] for i in sorted(pending.done)]
        return self._synthesize(pending.query, results, pending.voice_mode)

    # ---- confirmation ----

    def _answer_confirmation(self, reply: str, notify: Callable[[str], None]) -> str:
        pending = self._pending
        verdict = self.classify_confirmation(reply)
        log.info("confirmation reply %r -> %s", reply, verdict)

        if verdict == "UNCLEAR":
            pending.unclear_replies += 1
            if pending.unclear_replies < MAX_UNCLEAR_REPLIES:
                return "I didn't quite catch that. Please reply with yes or no.\n\n" + pending.waiting[0][2].prompt
            verdict = "NO"   # out of patience: treat as a refusal

        pending.unclear_replies = 0
        index, thread, paused = pending.waiting.pop(0)

        if verdict == "NO":
            pending.done[index] = ThreadResult(
                thread, paused.run.completed, error="User cancelled the operation.", index=index
            )
        else:
            ack = self._llm(
                "The user just confirmed a pending action. Write a one-sentence acknowledgment that you're proceeding.",
                max_tokens=30,
                include_history=False,
            )
            notify(ack)
            self._say(pending.voice_mode, ack)
            try:
                completed = self.executor.resume(paused, self.context)
                pending.done[index] = self._thread_result(thread, completed, index)
            except ConfirmationRequired as again:
                pending.waiting.insert(0, (index, thread, again))
            except ExecutionError as e:
                pending.done[index] = ThreadResult(thread, [], error=str(e), index=index)

        return self._advance(pending)

    def classify_confirmation(self, reply: str) -> str:
        verdict = self.llm.ask(reply, system=CONFIRM, temperature=0.0, max_tokens=5).upper()
        return next((word for word in ("YES", "NO", "UNCLEAR") if word in verdict), "UNCLEAR")

    # ---- talking ----

    def _llm(self, prompt: str, *, max_tokens: int = 300, voice_mode: bool = False, include_history: bool = True) -> str:
        messages = [{"role": "system", "content": personality(voice_mode)}]
        if include_history:
            messages += self.context.history[-HISTORY_WINDOW:]
        messages.append({"role": "user", "content": prompt})
        return self.llm.chat(messages, temperature=0.7, max_tokens=max_tokens)

    def _acknowledge(self, query: str, voice_mode: bool) -> str:
        prompt = (
            f'The user just said: "{query}"\n\n'
            "Write a very short, engaging acknowledgment (one sentence, at most 10 words) showing you "
            "understood the task and are getting started. Output only the acknowledgment."
        )
        return self._llm(prompt, max_tokens=30, voice_mode=voice_mode, include_history=False)

    def _chat(self, query: str, voice_mode: bool) -> str:
        prior = self.context.tool_results_text()
        if prior:
            prompt = (
                f"Data already retrieved:\n{prior}\n\nUser: \"{query}\"\n\n"
                "Answer from the data above if it has what's needed, otherwise from your own knowledge."
            )
        else:
            prompt = f'User: "{query}"\nRespond naturally.'
        return self._llm(prompt, max_tokens=150 if voice_mode else 300, voice_mode=voice_mode)

    def _synthesize(self, query: str, results: list[ThreadResult], voice_mode: bool) -> str:
        """Write the final answer from every thread's summary (or raw output, or error)."""
        blocks = []
        for r in results:
            n = r.index + 1
            if r.error:
                blocks.append(f"[Thread {n} FAILED] {r.error}")
            elif r.summary:
                blocks.append(f"[Thread {n}: {r.thread.goal}]\n{r.summary}")
            else:
                raw = "\n\n".join(f"[{step.tool}] {res.output}" for step, res in r.completed)
                blocks.append(f"[Thread {n}]\n{raw}")

        prompt = f'User asked: "{query}"\n\nResults by sub-task:\n' + "\n\n".join(blocks)

        # Raw search results let the answer recover details a summary dropped (prices, temperatures).
        if web := self.context.web_search_context():
            prompt += f"\n\nRaw web search results (use if the summaries lack detail):\n{web}"

        if urls := verified_urls(results):
            listing = "\n".join(f"  - {u}" for u in urls)
            prompt += (
                f"\n\nVERIFIED URLS (fetched successfully, safe to give to the user):\n{listing}\n"
                "If the user asked for a URL, cite only these, never one that only appeared in a search snippet."
            )
        else:
            prompt += (
                "\n\nNO VERIFIED URLS: no page was fetched successfully this turn. If the user asked for "
                "a URL, say you couldn't verify one instead of guessing from search snippets."
            )

        prompt += (
            "\n\nWrite the final response covering EVERY sub-task above. State facts directly, be concise, "
            "use your personality, and don't drop any sub-task. 2-4 sentences unless more is needed."
        )
        return self._llm(prompt, max_tokens=1024 if voice_mode else 2048, voice_mode=voice_mode)

    def _say(self, voice_mode: bool, text: str, block: bool = False) -> None:
        if self.speaker and voice_mode and text.strip():
            self.speaker.say(text, block=block)
