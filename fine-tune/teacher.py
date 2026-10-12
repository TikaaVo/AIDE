"""Sample the teacher planner on data/pool.jsonl -> data/teacher.jsonl.

Cascade: 2 samples per request, a 3rd only if they disagree (or are invalid). Resumable: re-run to continue.
Requests go in a fixed random order, so a pilot is representative of the whole pool.
Works with any OpenAI-compatible server (llama.cpp server, vLLM, ...).

    python teacher.py 100     # pilot: check output + token lengths before the full run
    python teacher.py         # everything not yet settled

Does the tool glossary (prompts/tools.txt) help? It is OFF by default; GLOSSARY=1 enables it. Score the teacher on gold with it:
    POOL=data/gold/gold.jsonl OUT=data/teacher_gold2.jsonl MAX_SAMPLES=1 TEMP=0 python teacher.py
    python evaluate.py teacher --file data/teacher_gold2.jsonl --name teacher_tools
Re-sample only the requests that had no usable label, into a separate file build_train.py merges automatically:
    ONLY=data/needs_resample.txt OUT=data/teacher2.jsonl python teacher.py 600

Teacher ceiling on the gold set (do this while the server is still up):
    POOL=data/gold/gold.jsonl OUT=data/teacher_gold.jsonl MAX_SAMPLES=1 TEMP=0 python teacher.py

Put AIDE's real planner system prompt in prompts/planner.txt. The user message is the raw request,
so adapt ask() if AIDE wraps it (date, context, ...).
Env: TEACHER_URL, TEACHER_MODEL, TEMP (default 0.7), POOL, OUT, WORKERS (match llama.cpp's -np), MAX_TOKENS, MAX_SAMPLES (default 3; rerun with 5 later to
retry requests where samples never agreed), THINKING=1 to leave the model's thinking ON
(default off: thinking burns hundreds of tokens per sample, and an unfinished thought comes back as an EMPTY reply).
Match whatever AIDE itself uses for its planner.
"""
import os, random, sys, threading
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

from common import DATA, append_jsonl, canon_of_raw, read_jsonl

URL = os.environ.get("TEACHER_URL", "http://localhost:8080/v1/chat/completions")
MODEL = os.environ.get("TEACHER_MODEL", "local")
THINKING = os.environ.get("THINKING", "0") == "1"
WORKERS, TEMP = int(os.environ.get("WORKERS", 4)), float(os.environ.get("TEMP", 0.7))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 3000 if THINKING else 1024))  # plans with 5+ threads can exceed 1024
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 3))  # cap per request; raise later to retry unresolved ones
SYSTEM = open("prompts/planner.txt", encoding="utf-8").read()
TOOLS_FILE = Path(os.environ.get("TOOLS_FILE", "prompts/tools.txt"))  # optional glossary: what each tool really does
GLOSSARY = TOOLS_FILE.exists() and os.environ.get("GLOSSARY", "0") == "1"  # opt-in: in our gold test it made the teacher output more invalid plans
if GLOSSARY:
    SYSTEM += "\n\n" + TOOLS_FILE.read_text(encoding="utf-8").strip() + "\n"
OUT = Path(os.environ.get("OUT", DATA / "teacher.jsonl"))
POOL = os.environ.get("POOL", DATA / "pool.jsonl")  # any jsonl with id + text
ONLY = os.environ.get("ONLY")  # file with one request id per line: only these are sampled
lock = threading.Lock()


def ask(text):
    r = requests.post(URL, timeout=600, json=dict(
        model=MODEL, temperature=TEMP, max_tokens=MAX_TOKENS, cache_prompt=True,
        chat_template_kwargs={"enable_thinking": THINKING},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])).json()
    ch = r["choices"][0]
    return ch["message"].get("content") or "", r["usage"]["completion_tokens"], ch.get("finish_reason")


def sample(req, k):
    raw, n, finish = ask(req["text"])
    c = canon_of_raw(raw)
    rec = dict(id=req["id"], sample=k, raw=raw, valid=c is not None, canon=c, tokens=n, finish=finish)
    with lock:
        append_jsonl(OUT, rec)  # keep raw output even when invalid: failures are informative
    return rec


def settled(recs):  # done once two valid samples agree, or the sample cap is reached
    c = Counter(r["canon"] for r in recs if r["valid"])
    return bool(c) and c.most_common(1)[0][1] >= 2 or len(recs) >= MAX_SAMPLES


def work(job):
    req, recs = job
    try:
        while not settled(recs):
            recs.append(sample(req, len(recs)))
    except Exception as e:  # server hiccup: skip, a re-run picks it up
        print(f"{req['id']}: {e!r}", flush=True)


if __name__ == "__main__":
    done = defaultdict(list)
    for r in read_jsonl(OUT):
        r["canon"] = canon_of_raw(r["raw"])  # recomputed, so canonicalisation fixes apply to old samples too
        r["valid"] = r["canon"] is not None
        done[r["id"]].append(r)
    only = set(open(ONLY).read().split()) if ONLY else None
    todo = [(r, list(done[r["id"]])) for r in read_jsonl(POOL)
            if (only is None or r["id"] in only) and not settled(done[r["id"]])]
    random.Random(0).shuffle(todo)
    if len(sys.argv) > 1:
        todo = todo[:int(sys.argv[1])]
    print(f"{len(todo)} requests to go (thinking {'ON' if THINKING else 'off'}, {WORKERS} workers, tool glossary {'ON' if GLOSSARY else 'off'})", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(work, todo), 1):
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", flush=True)