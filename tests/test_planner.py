import json

from core.context import SessionContext
from core.planner import (Plan, PlanStep, Planner, Thread, add_summary_step, dedup_steps,
                          enforce_terminal_prereqs)
from conftest import EchoTool, FakeLLM

TOOLS = {n: EchoTool(n) for n in ("web_search", "deep_think", "terminal", "file_search", "read_file",
                                  "open_url", "chat", "summarize")}


def plan_for(raw_reply: str) -> Plan:
    llm = FakeLLM({"planner": raw_reply})
    return Planner(TOOLS, llm).plan("do the thing", SessionContext())


def step(tool, key, inp="x", deps=None):
    return PlanStep(tool, inp, key, deps)


def test_parses_fenced_json_and_merges_sibling_objects():
    one = {"threads": [{"goal": "a", "steps": [{"tool": "web_search", "input": "q1", "output_key": "t1_a"}]}]}
    two = {"threads": [{"goal": "b", "steps": [{"tool": "web_search", "input": "q2", "output_key": "t2_a"}]}]}
    plan = plan_for(f"```json\n{json.dumps(one)}\n{json.dumps(two)}\n```")
    assert [t.goal for t in plan.threads] == ["a", "b"]


def test_unknown_tools_are_dropped_and_summary_is_appended():
    plan = plan_for(json.dumps({"threads": [{"goal": "g", "steps": [
        {"tool": "teleport", "input": "x", "output_key": "k0"},
        {"tool": "web_search", "input": "q", "output_key": "k1", "depends_on": None},
    ]}]}))
    tools = [s.tool for s in plan.threads[0].steps]
    assert tools == ["web_search", "summarize"]
    assert plan.threads[0].steps[-1].depends_on == ["k1"]


def test_flat_steps_format_is_upgraded_to_a_thread():
    plan = plan_for(json.dumps({"steps": [{"tool": "web_search", "input": "q", "output_key": "k"}]}))
    assert len(plan.threads) == 1 and plan.threads[0].goal == "do the thing"


def test_nested_input_object_is_flattened():
    s = PlanStep.from_dict({"tool": "web_search", "input": {"query": "cats"}, "output_key": "k"}, "d")
    assert s.input == "cats"


def test_two_bad_replies_fall_back_to_a_chat_plan():
    llm = FakeLLM({"planner": ["not json at all", '{"threads": []}']})
    plan = Planner(TOOLS, llm).plan("hello", SessionContext())
    assert [s.tool for s in plan.steps] == ["chat"]
    assert len(llm.calls) == 2


def test_chat_only_thread_gets_no_summary_step():
    thread = Thread("g", [step("chat", "c")])
    add_summary_step(thread)
    assert [s.tool for s in thread.steps] == ["chat"]


def test_dedup_rewires_dependents_but_keeps_open_url():
    steps = [
        step("file_search", "a", "Blackjack  files"),
        step("file_search", "b", "blackjack files"),
        step("open_file", "c", deps=["b"]),
        step("open_url", "u1", "fetch top result"),
        step("open_url", "u2", "fetch top result"),
    ]
    out = dedup_steps(steps)
    assert [s.output_key for s in out] == ["a", "c", "u1", "u2"]
    assert out[1].depends_on == ["a"]


def test_terminal_gets_search_and_think_injected():
    out = enforce_terminal_prereqs([step("terminal", "t", "install ffmpeg")], "install ffmpeg", "install ffmpeg")
    assert [s.tool for s in out] == ["web_search", "deep_think", "terminal"]
    assert set(out[-1].depends_on) == {out[0].output_key, out[1].output_key}


def test_terminal_reuses_existing_search_and_think():
    steps = [step("web_search", "ws"), step("deep_think", "dt", deps=["ws"]), step("terminal", "t")]
    out = enforce_terminal_prereqs(steps, "g", "q")
    assert [s.tool for s in out] == ["web_search", "deep_think", "terminal"]
    assert out[-1].depends_on == ["ws", "dt"]


def test_running_a_located_file_skips_the_prereqs():
    steps = [step("file_search", "fs"), step("terminal", "t", "run blackjack script", deps=["fs"])]
    out = enforce_terminal_prereqs(steps, "Find and run the blackjack script", "run it")
    assert [s.tool for s in out] == ["file_search", "terminal"]


def test_installing_after_a_file_search_still_gets_prereqs():
    steps = [step("file_search", "fs"), step("terminal", "t", "pip install numpy", deps=["fs"])]
    out = enforce_terminal_prereqs(steps, "install numpy for my script", "q")
    assert "web_search" in [s.tool for s in out]


def test_think_flag_is_passed_to_the_model():
    llm = FakeLLM({"planner": "{}"})
    planner = Planner(TOOLS, llm)
    planner.ask("q", SessionContext())
    planner.ask("q", SessionContext(), think=False)
    assert [c["think"] for c in llm.calls] == [True, False]
