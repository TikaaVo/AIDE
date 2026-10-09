import ollama
import numpy as np
from tools.base import BaseTool


class Embedder:
    """
    Wraps the Ollama embedding model.
    Used by the planner to score tool relevance as a fast hint —
    not to make routing decisions.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            response = ollama.embeddings(model=self.model, prompt=text)
            self._cache[text] = np.array(response["embedding"], dtype=np.float32)
        return self._cache[text]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def score_tools(
        self,
        query: str,
        tools: dict[str, BaseTool]
    ) -> dict[str, float]:
        """
        Returns a score in [0, 1] for each tool based on the max cosine
        similarity between the query embedding and any of the tool's example
        phrase embeddings.

        This is a cheap relevance signal passed to the planner as a hint.
        It does NOT make routing decisions.
        """
        query_emb = self.embed(query)
        scores: dict[str, float] = {}

        for tool_name, tool in tools.items():
            phrase_embs = [self.embed(p) for p in tool.example_phrases]
            scores[tool_name] = max(
                self.cosine_similarity(query_emb, p) for p in phrase_embs
            )

        return scores

    def precompute(self, tools: dict[str, BaseTool]) -> None:
        """Pre-embed all example phrases at startup to warm the cache."""
        for tool in tools.values():
            for phrase in tool.example_phrases:
                self.embed(phrase)