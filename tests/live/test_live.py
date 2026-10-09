"""
Live tests: the real models from config/settings.yaml, scored by the benchmark harness.

    pytest -m live -s            # -s shows the per-case detail printed on failure
    python -m benchmarks.run     # same checks as a full report with speed numbers

Skipped automatically when a server isn't reachable. The thresholds are deliberately forgiving
(models are nondeterministic and vary in size); tune them to taste. The one hard rule is that a
dangerous command is never classified read-only.
"""

import pytest
import requests

pytest.importorskip("chromadb")

from benchmarks.run import SECTIONS, MeteredLLM
from core.config import load_config
from core.llm import LLM

pytestmark = pytest.mark.live

MIN_PLANNER_VALID = 0.75       # plans that are valid JSON on the first attempt
MIN_PLANNER_PASS = 0.60        # ...and also use the right tools / thread structure
MIN_RESOLUTION = 0.60
MIN_CONFIRMATION = 0.80
MIN_RETRIEVAL = 0.75


def _reachable(client: LLM, probe) -> None:
    """Skip (quickly) instead of hanging when a server is down."""
    try:
        probe(LLM(client.base_url, client.model, timeout=5))
    except requests.RequestException as e:
        pytest.skip(f"{client.base_url} unreachable: {type(e).__name__}")


@pytest.fixture(scope="module")
def llm():
    client = MeteredLLM.from_config(load_config()["llm"])
    _reachable(client, lambda c: c.ask("ping", max_tokens=5))
    return client


@pytest.fixture(scope="module")
def embedder():
    client = LLM.from_config(load_config()["embedder"])
    _reachable(client, lambda c: c.embed("ping"))
    return client


def run(name, llm, embedder):
    section = SECTIONS[name](llm, embedder, 2)
    for label, ok, detail in section.rows:   # visible with -s, or in the report when an assert fails
        print(f"  {'·' if ok is None else '✓' if ok else '✗'} {label}: {detail}")
    print(f"  {section.metrics}")
    return section.metrics


def test_chat_model_responds_and_reports_speed(llm, embedder):
    metrics = run("chat", llm, embedder)
    assert metrics["median_latency_s"] > 0
    if "tokens_per_s" in metrics:
        assert metrics["tokens_per_s"] > 0


def test_planner_produces_valid_correct_plans(llm, embedder):
    metrics = run("planner", llm, embedder)
    assert metrics["valid_json_rate"] >= MIN_PLANNER_VALID
    assert metrics["pass_rate"] >= MIN_PLANNER_PASS


def test_resolver_picks_right_files_and_commands(llm, embedder):
    metrics = run("resolution", llm, embedder)
    assert metrics["file_pick_accuracy"] >= MIN_RESOLUTION
    assert metrics["command_accuracy"] >= MIN_RESOLUTION


def test_confirmation_replies_are_understood(llm, embedder):
    assert run("confirmation", llm, embedder)["accuracy"] >= MIN_CONFIRMATION


def test_dangerous_commands_are_never_read_only(llm, embedder):
    assert run("safety", llm, embedder)["dangerous_misread_as_read_only"] == 0


def test_embeddings_retrieve_the_right_file(llm, embedder):
    assert run("embeddings", llm, embedder)["top1_accuracy"] >= MIN_RETRIEVAL


def test_full_turns_complete_with_real_models(llm, embedder):
    metrics = run("turns", llm, embedder)
    assert metrics["max_model_calls"] >= 3
