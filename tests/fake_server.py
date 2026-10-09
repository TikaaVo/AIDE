"""
A tiny local HTTP server that speaks the same API as llama-server / Ollama
(/v1/chat/completions and /v1/embeddings) with rule-based "answers".

It exists to test the benchmark harness itself (HTTP, metering, scoring, reporting) without a
real model. It says nothing about how good any real model is.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOPICS = [
    {"blackjack", "learning", "reinforcement", "card", "game", "agent"},
    {"pasta", "cook", "recipe", "tomatoes", "basil"},
    {"invoice", "tax", "payments", "money", "owe", "march"},
    {"lisbon", "flights", "holiday", "vacation", "packing", "bookings"},
    {"workout", "squats", "exercise", "routine", "deadlifts", "weekly"},
    {"thesis", "latex", "neural", "dissertation", "deep"},
]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(text: str) -> list[float]:
    tokens = words(text)
    return [float(sum(t in topic for t in tokens)) for topic in TOPICS] + [0.01]


def plan_for(request: str) -> dict:
    threads = []
    for n, part in enumerate(re.split(r", and |, | and also ", request.lower()), 1):
        key = lambda suffix: f"t{n}_{suffix}"   # noqa: E731
        step = lambda tool, inp, k, dep=None: {"tool": tool, "input": inp, "output_key": key(k), "depends_on": dep}   # noqa: E731
        if "weather" in part:
            steps = [step("web_search", part, "ws")]
        elif "joke" in part or "rewrite" in part:
            steps = [step("chat", part, "chat")]
        elif "install" in part:
            steps = [step("web_search", part, "ws"), step("deep_think", "which command?", "dt", [key("ws")]),
                     step("terminal", part, "term", [key("ws"), key("dt")])]
        elif re.search(r"\brun\b", part):
            steps = [step("file_search", part, "fs"), step("terminal", "run it", "term", [key("fs")])]
        elif re.search(r"\bread\b", part):
            steps = [step("file_search", part, "fs"), step("read_file", "read it", "rf", [key("fs")])]
        elif re.search(r"look up|latest|\brate\b", part):
            steps = [step("web_search", part, "ws")]
        elif re.search(r"\burl\b", part):
            steps = [step("web_search", part, "ws"), step("open_url", "top result", "ou", [key("ws")])]
        else:
            steps = [step("file_search", part, "fs")]
            if "open" in part:
                steps.append(step("open_file", "open it", "of", [key("fs")]))
        threads.append({"goal": part, "steps": steps})
    return {"threads": threads}


def respond(system: str, user: str) -> str:
    if "You are the planner" in system:
        return json.dumps(plan_for(re.search(r'User request: "(.*)"', user).group(1)))
    if "Classify the user's reply" in system:
        tokens = set(words(user))
        if "?" in user:
            return "UNCLEAR"
        if tokens & {"no", "nope", "cancel", "stop", "not"}:
            return "NO"
        return "YES" if tokens & {"yes", "yep", "sure", "go", "guess"} else "UNCLEAR"
    if "classify shell commands" in system:
        first = user.split()[0]
        if first in {"wipefs", "truncate"}:
            return "IRREVERSIBLE"
        changing = {"mkdir", "touch", "chmod", "cp", "pip", "find", "curl", "git"}
        return "REVERSIBLE" if first in changing else "READ_ONLY"
    if "bash commands for an assistant" in system:
        goal = re.search(r'Thread goal: "(.*)"', user).group(1).lower()
        if paths := re.findall(r"path: (\S+)", user, re.IGNORECASE):
            stems = [w[:5] for w in words(goal) if len(w) > 3]
            return "python3 " + max(paths, key=lambda p: sum(t in p.lower() for t in stems))
        if "requests" in goal:
            return "pip install requests"
        return "sudo apt install -y htop" if "htop" in user else "df -h"
    if "You turn the output of earlier tools" in system:
        request = re.search(r'original request: "(.*)"', user).group(1)
        paths = re.findall(r"Path: (\S+)", user)
        wanted = [w for w in words(request) if len(w) > 3]
        if "latest" in request.lower() and paths:
            return max(paths)
        return max(paths, key=lambda p: sum(w in p.lower() for w in wanted)) if paths else "https://example.com"
    if "adaptive controller" in system:
        return '{"action": "continue", "reason": "good enough"}'
    if "very short, engaging acknowledgment" in user:
        return "On it."
    if "concise, factual summaries" in system:
        return "Summary of the results."
    if "Results by sub-task" in user:
        return "Here is your answer."
    return "A mutex is a lock that lets only one thread use a resource at a time."


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/embeddings":
            payload = {"data": [{"embedding": embed(body["input"])}]}
        else:
            messages = body["messages"]
            system = " ".join(m["content"] for m in messages if m["role"] == "system")
            user = messages[-1]["content"]
            thinking = body.get("chat_template_kwargs", {}).get("enable_thinking", False)
            payload = {
                "choices": [{"message": {"content": respond(system, user)}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 300 if thinking else 12},
                "timings": {"predicted_per_second": 50.0, "prompt_per_second": 400.0},
            }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):   # keep test output quiet
        pass


def serve() -> tuple[ThreadingHTTPServer, str]:
    """Start the server on a free port. Returns (server, base_url); call server.shutdown() when done."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"
