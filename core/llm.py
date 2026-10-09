"""Thin client for a llama.cpp server (OpenAI-compatible API) plus JSON helpers."""

import json
import re
from collections.abc import Iterator

import requests

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```\w*\n?|\n?```$")


class LLM:
    """One chat/embedding endpoint. Everything in AIDE talks to the model through this."""

    def __init__(self, base_url: str, model: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_config(cls, section: dict) -> "LLM":
        return cls(section["base_url"], section["model"], section.get("timeout", 300))

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Return the reply text. See complete() for the options."""
        return self.complete(messages, **kwargs)[0]

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        think: bool = False,
    ) -> tuple[str, dict]:
        """
        Return (reply text, server stats). `think` toggles the chat template's reasoning mode.
        Stats are whatever the server reports: {"usage": {...}, "timings": {...}} (llama.cpp does both).
        """
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": think},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"]["content"] or ""
        stats = {"usage": body.get("usage", {}), "timings": body.get("timings", {})}
        return _THINK_BLOCK.sub("", text).strip(), stats

    def ask(self, prompt: str, system: str = "", **kwargs) -> str:
        """Single-turn convenience wrapper around chat()."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def clean_reply(text: str) -> str:
    """Strip a surrounding ```code fence``` and one pair of wrapping quotes from a model reply."""
    text = _FENCE.sub("", text.strip()).strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"`":
        text = text[1:-1].strip()
    return text


def extract_json_objects(text: str) -> Iterator[dict]:
    """
    Yield every top-level JSON object embedded in `text`.

    Models wrap JSON in fences, add chatter, or emit several objects back to
    back. Scanning for '{' and using raw_decode copes with all three.
    """
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        yield obj
        i = text.find("{", end)
