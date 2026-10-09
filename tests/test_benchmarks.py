"""Checks the benchmark harness itself (scoring, metering, reporting) against a scripted fake server."""

import json

import pytest

pytest.importorskip("chromadb")   # the harness imports every tool, including file_search

from benchmarks import cases
from benchmarks import run as bench
from core import config
from core.llm import LLM
from fake_server import serve


@pytest.fixture(scope="module")
def models():
    server, url = serve()
    yield bench.MeteredLLM(url, "fake-chat"), LLM(url, "fake-embed"), url
    server.shutdown()


def test_every_section_runs_and_a_competent_model_scores_full_marks(models):
    llm, embedder, _ = models
    scores = {name: fn(llm, embedder, 1).metrics for name, fn in bench.SECTIONS.items()}

    assert scores["chat"]["tokens_per_s"] > 0 and scores["chat"]["server_generation_tok_s"] == 50.0
    assert scores["thinking"]["tokens_think_on"] > scores["thinking"]["tokens_think_off"]
    assert scores["planner"]["valid_json_rate"] == 1.0 and scores["planner"]["pass_rate"] == 1.0
    assert scores["planner"]["consistent_case_rate"] == 1.0 and scores["planner"]["no_think_pass_rate"] == 1.0
    assert scores["resolution"] == {"file_pick_accuracy": 1.0, "command_accuracy": 1.0}
    assert scores["confirmation"]["accuracy"] == 1.0
    assert scores["safety"]["dangerous_misread_as_read_only"] == 0
    assert scores["embeddings"]["top1_accuracy"] == 1.0 and scores["embeddings"]["dimensions"] > 0
    assert scores["turns"]["max_model_calls"] >= 3


def test_metering_counts_every_model_call(models):
    llm, embedder, _ = models
    before = len(llm.records)
    bench.SECTIONS["confirmation"](llm, embedder, 1)
    assert len(llm.records) - before == len(cases.CONFIRM_CASES)   # one classification per case


class Gibberish(bench.MeteredLLM):
    def complete(self, messages, **kwargs):
        return "lorem ipsum", {"usage": {}, "timings": {}}


def test_a_bad_model_is_scored_badly_without_crashing(models):
    _, embedder, url = models
    bad = Gibberish(url, "bad")
    planner = bench.SECTIONS["planner"](bad, embedder, 1)
    assert planner.metrics["valid_json_rate"] == 0 and planner.metrics["pass_rate"] == 0
    assert all(ok is False for _, ok, _ in planner.rows)

    resolution = bench.SECTIONS["resolution"](bad, embedder, 1)
    assert resolution.metrics["file_pick_accuracy"] == 0
    assert bench.SECTIONS["confirmation"](bad, embedder, 1).metrics["accuracy"] == pytest.approx(sum(e == "UNCLEAR" for _, e in cases.CONFIRM_CASES) / len(cases.CONFIRM_CASES))   # only "unclear" ones


def test_cli_prints_a_report_and_saves_json(models, tmp_path, monkeypatch, capsys):
    _, _, url = models
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(f"llm: {{base_url: '{url}', model: fake-chat}}\nembedder: {{base_url: '{url}', model: fake-embed}}\n")
    monkeypatch.setenv("AIDE_CONFIG", str(cfg))
    config.load_config.cache_clear()
    try:
        out = tmp_path / "results.json"
        assert bench.main(["safety", "confirmation", "--json", str(out)]) == 0
        assert "dangerous_misread_as_read_only: 0" in capsys.readouterr().out
        saved = json.loads(out.read_text())
        assert saved["chat_model"] == "fake-chat" and saved["results"]["confirmation"]["accuracy"] == 1.0
    finally:
        config.load_config.cache_clear()


def test_planner_repeats_and_thinking_ablation_are_metered(models):
    llm, embedder, _ = models
    before = len(llm.records)
    bench.SECTIONS["planner"](llm, embedder, 3)
    n = len(cases.PLANNER_CASES)
    assert len(llm.records) - before == n * 3 + n   # repeats with thinking, plus one pass without


class AlwaysSearches(bench.MeteredLLM):
    """A lazy 'planner' that answers every request with one web search."""

    def complete(self, messages, **kwargs):
        plan = {"threads": [{"goal": "g", "steps": [{"tool": "web_search", "input": "q", "output_key": "k"}]}]}
        return json.dumps(plan), {"usage": {}, "timings": {}}


def test_the_planner_cases_can_tell_a_lazy_model_from_a_good_one(models):
    _, embedder, url = models
    metrics = bench.SECTIONS["planner"](AlwaysSearches(url, "lazy"), embedder, 1).metrics
    assert metrics["valid_json_rate"] == 1.0   # perfectly valid JSON...
    assert metrics["pass_rate"] < 0.2          # ...but wrong for almost every request
