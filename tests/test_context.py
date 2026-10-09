from core.context import SessionContext, ToolResult


def test_file_search_hits_are_promoted():
    ctx = SessionContext()
    hits = [{"path": "/a.py", "name": "a.py"}, {"path": "/b.py", "name": "b.py"}]
    ctx.add_tool_result(ToolResult("file_search", "q", True, "out", {"hits": hits}))
    assert ctx.data["path"] == "/a.py"
    assert ctx.data["all_paths"] == ["/a.py", "/b.py"]


def test_rendering_skips_scaffolding_and_keeps_recent_three_per_tool():
    ctx = SessionContext()
    for i in range(5):
        ctx.add_tool_result(ToolResult("web_search", f"q{i}", True, f"result {i}"))
    ctx.add_tool_result(ToolResult("deep_think", "x", True, "private reasoning"))

    text = ctx.tool_results_text()
    assert "private reasoning" not in text
    assert "result 0" not in text and "result 1" not in text
    assert "result 2" in text and "result 4" in text
    assert text.index("result 2") < text.index("result 4")   # chronological


def test_web_search_context_and_history():
    ctx = SessionContext()
    ctx.add_tool_result(ToolResult(
        "web_search", "weather", True, "...",
        {"query": "weather", "results": [{"title": "T", "snippet": "S", "url": "http://u"}]},
    ))
    assert "http://u" in ctx.web_search_context()

    ctx.add_turn("user", "hi")
    ctx.add_turn("assistant", "hello")
    assert ctx.history_text(1) == "ASSISTANT: hello"
