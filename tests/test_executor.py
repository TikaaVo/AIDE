import pytest

from core.context import SessionContext
from core.executor import ConfirmationRequired, ExecutionError, Executor
from core.planner import PlanStep, Thread
from conftest import EchoTool, FakeLLM
from tools.summarize import SummarizeTool

CONTINUE = '{"action": "continue", "reason": "fine"}'


def run(thread, tools, llm, ctx=None):
    ctx = ctx or SessionContext()
    return Executor(tools, llm).execute_thread(thread, ctx, "user query"), ctx


def test_dependent_step_receives_the_resolved_input():
    search = EchoTool("web_search")
    fetch = EchoTool("open_url")
    llm = FakeLLM({"adaptive controller": CONTINUE, "You turn the output": "https://example.com/doc"})
    thread = Thread("read the docs", [
        PlanStep("web_search", "docs", "a"),
        PlanStep("open_url", "fetch top result", "b", ["a"]),
    ])
    done, _ = run(thread, {"web_search": search, "open_url": fetch}, llm)

    assert fetch.calls[0][0] == "https://example.com/doc"
    assert [s.output_key for s, _ in done] == ["a", "b"]
    resolver_prompt = llm.prompts_containing("You turn the output")[0]
    assert "web_search says docs" in resolver_prompt   # the resolver saw the dependency's output


def test_summarize_gets_raw_outputs_without_a_resolver_call():
    llm = FakeLLM({"adaptive controller": CONTINUE, "concise, factual summaries": "SUMMARY"})
    tools = {"web_search": EchoTool("web_search"), "summarize": SummarizeTool(llm)}
    thread = Thread("g", [
        PlanStep("web_search", "weather", "a"),
        PlanStep("summarize", "", "a_summary", ["a"]),
    ])
    done, _ = run(thread, tools, llm)

    assert done[-1][1].output == "SUMMARY"
    assert not llm.prompts_containing("You turn the output")
    assert "web_search says weather" in llm.prompts_containing("concise, factual summaries")[0]


def test_retry_runs_the_step_again_with_the_new_input():
    search = EchoTool("web_search")
    llm = FakeLLM({"adaptive controller": [
        '{"action": "retry", "retry_input": "better query"}',
        CONTINUE,
    ]})
    done, _ = run(Thread("g", [PlanStep("web_search", "bad query", "a")]), {"web_search": search}, llm)
    assert [c[0] for c in search.calls] == ["bad query", "better query"]
    assert len(done) == 1


def test_retry_with_the_same_input_is_ignored():
    search = EchoTool("web_search")
    llm = FakeLLM({"adaptive controller": '{"action": "retry", "retry_input": "same"}'})
    run(Thread("g", [PlanStep("web_search", "same", "a")]), {"web_search": search}, llm)
    assert len(search.calls) == 1


def test_garbage_from_the_controller_means_continue():
    llm = FakeLLM({"adaptive controller": "I think we should carry on, friend."})
    done, _ = run(Thread("g", [PlanStep("web_search", "q", "a")]), {"web_search": EchoTool("web_search")}, llm)
    assert len(done) == 1


def test_inject_adds_steps_that_run_next():
    search = EchoTool("web_search", data={"results": [{"url": "https://x.dev/a"}]})
    fetch = EchoTool("open_url")
    llm = FakeLLM({
        "adaptive controller": ['{"action": "inject", "new_steps": [{"tool": "open_url", '
                                '"input": "https://x.dev/a", "output_key": "extra", "depends_on": ["web_search"]}]}',
                                CONTINUE],
        "You turn the output": "https://x.dev/a",
    })
    done, _ = run(Thread("g", [PlanStep("web_search", "q", "a")]), {"web_search": search, "open_url": fetch}, llm)
    assert [s.output_key for s, _ in done] == ["a", "extra"]
    assert fetch.calls  # the injected step really ran


def test_failure_only_halts_the_thread_when_a_later_step_needs_it():
    bad = EchoTool("web_search", success=False)
    llm = FakeLLM({"adaptive controller": CONTINUE})

    # nothing depends on it: the thread carries on
    done, _ = run(Thread("g", [PlanStep("web_search", "q", "a")]), {"web_search": bad}, llm)
    assert done[0][1].success is False

    # something depends on it: the thread stops
    thread = Thread("g", [PlanStep("web_search", "q", "a"), PlanStep("open_url", "x", "b", ["a"])])
    with pytest.raises(ExecutionError):
        run(thread, {"web_search": bad, "open_url": EchoTool("open_url")}, llm)


def test_confirmation_pauses_then_resumes_where_it_left_off():
    danger = EchoTool("danger", confirm=True)
    after = EchoTool("after")
    llm = FakeLLM({"adaptive controller": CONTINUE})
    ex = Executor({"danger": danger, "after": after}, llm)
    ctx = SessionContext()
    thread = Thread("g", [PlanStep("danger", "rm stuff", "a"), PlanStep("after", "next", "b")])

    with pytest.raises(ConfirmationRequired) as paused:
        ex.execute_thread(thread, ctx, "q")
    assert paused.value.prompt == "run rm stuff?"
    assert not after.calls   # nothing after the gate ran

    done = ex.resume(paused.value, ctx)
    assert danger.calls[-1][1]["confirmed"] is True
    assert [s.output_key for s, _ in done] == ["a", "b"]
