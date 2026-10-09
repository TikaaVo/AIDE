"""Indexing and ranking against a tiny on-disk Chroma DB with a fake embedder (no model needed)."""

import hashlib

import pytest

pytest.importorskip("chromadb")

from core.llm import LLM
from tools.file_search import FileSearchTool

VOCAB = ["blackjack", "recipe", "invoice", "python", "pasta"]


class BagOfWordsEmbedder(LLM):
    """Deterministic embeddings: one dimension per vocabulary word."""

    def __init__(self):
        super().__init__("http://fake", "fake")
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        lowered = text.lower()
        return [1.0 + lowered.count(w) * 5 for w in VOCAB]


@pytest.fixture
def tool(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "game.py").write_text("python blackjack q-learning agent")
    (docs / "dinner.md").write_text("pasta recipe with tomatoes")
    (docs / "taxes.txt").write_text("invoice for march")
    monkeypatch.setattr("tools.file_search.load_config", lambda: {"paths": {"file_search_roots": [str(docs)]}})
    return FileSearchTool(BagOfWordsEmbedder(), db_path=tmp_path / "chroma")


def test_search_ranks_the_relevant_file_first(tool):
    tool.index()
    result = tool.run("blackjack python", {})
    assert result.success
    assert result.data["hits"][0]["name"] == "game.py"
    assert result.data["hits"][0]["path"].endswith("docs/game.py")


def test_unchanged_files_are_not_re_embedded(tool):
    tool.index()
    after_first = tool.embedder.calls
    tool.index()
    assert tool.embedder.calls == after_first


def test_empty_index_reports_gracefully(tool):
    assert not tool.run("anything", {}).success


def test_modified_files_are_re_embedded(tool, tmp_path):
    import os

    tool.index()
    before = tool.embedder.calls
    changed = tmp_path / "docs" / "taxes.txt"
    changed.write_text("invoice for april")
    os.utime(changed, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    tool.index()
    assert tool.embedder.calls == before + 1
