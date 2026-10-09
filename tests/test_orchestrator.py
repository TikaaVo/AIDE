"""End-to-end turns with a scripted model and stub tools: no server, no network."""

import json

from core.orchestrator import Orchestrator
from conftest import EchoTool, FakeLLM
from tools.chat import ChatTool
from tools.summarize import SummarizeTool


def plan(*steps, goal="g"):
    return json.dumps({"threads": [{"goal": goal, "steps": list(steps)}]})


def routes(**overrides):
    base = {
        "Classify the user's reply": "YES",
        "You are the planner": plan({"tool": "chat", "input": "hi", "output_key": "t1_c"}),
        "very short, engaging acknowledgment": "On it.",
        "just confirmed a pending action": "Proceeding.",
        "adaptive controller": '{"action": "continue"}',
        "concise, factual summaries": "SUMMARY-TEXT",
        "Results by sub-task": "FINAL ANSWER",
        "Respond naturally": "CHAT ANSWER",
    }
    base.update(overrides)
    return base


def make(llm, **tools):
    registry = {"chat": ChatTool(), "summarize": SummarizeTool(llm), **tools}
    return Orchestrator(llm, registry)


def test_conversational_turn_skips_tools_and_updates_history():
    llm = FakeLLM(routes())
    bot = make(llm)
    updates = []

    assert bot.handle("hello", update_callback=updates.append) == "CHAT ANSWER"
    assert updates == ["On it."]
    assert [m["role"] for m in bot.context.history] == ["user", "assistant"]


def test_tool_turn_summarizes_each_thread_and_synthesizes():
    search = EchoTool("web_search", data={"results": [{"title": "T", "snippet": "S", "url": "https://u.dev"}], "query": "q"})
    llm = FakeLLM(routes(**{
        "You are the planner": plan({"tool": "web_search", "input": "q", "output_key": "t1_ws"}, goal="look it up"),
    }))
    bot = make(llm, web_search=search)

    assert bot.handle("look it up") == "FINAL ANSWER"
    synth = llm.prompts_containing("Results by sub-task")[0]
    assert "SUMMARY-TEXT" in synth and "look it up" in synth
    assert "NO VERIFIED URLS" in synth                      # a search snippet alone isn't verified
    assert "https://u.dev" in synth                         # but raw results are available as a fallback


def test_only_fetched_urls_are_marked_verified():
    search = EchoTool("web_search", data={"results": [{"url": "https://snippet.dev"}], "query": "q"})
    fetch = EchoTool("open_url", data={"url": "https://fetched.dev"})
    llm = FakeLLM(routes(**{
        "You are the planner": plan(
            {"tool": "web_search", "input": "q", "output_key": "ws"},
            {"tool": "open_url", "input": "top", "output_key": "ou", "depends_on": "ws"},
        ),
        "You turn the output": "https://fetched.dev",
    }))
    bot = make(llm, web_search=search, open_url=fetch)
    bot.handle("find the docs")
    synth = llm.prompts_containing("Results by sub-task")[0]
    assert "VERIFIED URLS (fetched" in synth and "  - https://fetched.dev" in synth


def danger_bot(**overrides):
    danger = EchoTool("danger", confirm=True)
    llm = FakeLLM(routes(**{
        "You are the planner": plan({"tool": "danger", "input": "launch", "output_key": "t1_d"}),
        **overrides,
    }))
    return make(llm, danger=danger), danger, llm


def test_confirmation_yes_resumes_and_finishes():
    bot, danger, llm = danger_bot()
    first = bot.handle("do the risky thing")
    assert "Before I continue" in first and "run launch?" in first
    assert bot._pending is not None and not danger.calls[-1][1].get("confirmed")

    assert bot.handle("yes please") == "FINAL ANSWER"
    assert danger.calls[-1][1]["confirmed"] is True
    assert bot._pending is None


def test_confirmation_no_cancels_without_running_anything():
    bot, danger, llm = danger_bot(**{"Classify the user's reply": "NO"})
    bot.handle("do the risky thing")
    assert bot.handle("nope") == "FINAL ANSWER"
    assert len(danger.calls) == 1                                       # only the original, unconfirmed call
    assert "User cancelled" in llm.prompts_containing("Results by sub-task")[0]


def test_unclear_replies_are_reasked_then_treated_as_no():
    bot, danger, llm = danger_bot(**{"Classify the user's reply": "UNCLEAR"})
    bot.handle("do the risky thing")
    assert "yes or no" in bot.handle("banana")
    assert "yes or no" in bot.handle("what?")
    assert bot.handle("hmm") == "FINAL ANSWER"                          # third strike cancels
    assert len(danger.calls) == 1


def test_reset_clears_session_and_pending_confirmation():
    bot, _, _ = danger_bot()
    bot.handle("do the risky thing")
    bot.reset()
    assert bot._pending is None and bot.context.history == []


class RecordingSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text, block=False):
        self.said.append(text)


def test_speaker_is_only_used_in_voice_mode():
    llm = FakeLLM(routes())
    speaker = RecordingSpeaker()
    bot = Orchestrator(llm, {"chat": ChatTool(), "summarize": SummarizeTool(llm)}, speaker)

    bot.handle("typed message")
    assert speaker.said == []
    bot.handle("spoken message", voice_mode=True)
    assert speaker.said == ["On it.", "CHAT ANSWER"]
