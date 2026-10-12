"""Shared helpers: JSONL io, text cleaning, similarity, and plan validation / canonicalisation / compact form.

Plan format (AIDE):  {"threads": [{"goal": str, "steps": [{"tool", "input", "output_key", "depends_on"}]}]}
Assumption: depends_on points to EARLIER steps in the SAME thread. Loosen validate() if AIDE allows more.

Gold record format (data/gold/gold.jsonl; generated from gold.csv by gold.py, don't hand-edit the jsonl):
  {"id": "g001", "text": "<request>", "must_use": ["file_search"], "must_not_use": ["terminal"],
   "min_threads": 1, "max_threads": 3, "ordering": [["file_search", "open_file"]],
   "must_use_any": [["chat", "web_search"]], "tags": ["typo"], "note": ""}
  must_use_any: each inner list = at least one of these tools must appear.
"""
import hashlib, html, json, re, unicodedata, zlib
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("data")
TOOLS = {"file_search", "open_file", "read_file", "deep_think", "terminal",
         "web_search", "chat", "summarize", "open_url"}
STUDENT_PROMPT = "Plan the tool calls for this request.\n\nRequest: {}\n\nPlan:\n"


# ---------- io ----------
def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- text ----------
def clean(t):  # unicode-normalise, unescape HTML, drop zero-width chars, collapse whitespace
    t = html.unescape(unicodedata.normalize("NFKC", t))
    return re.sub(r"\s+", " ", re.sub(r"[\u200b\ufeff]", "", t)).strip()


def norm_key(t):  # for exact dedup: case/punctuation-insensitive
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def bucket(id_, n=100):  # deterministic 0..n-1 from an id: re-runs never move examples between splits
    return int(hashlib.sha1(id_.encode()).hexdigest(), 16) % n


def vectors(texts, dim=2048):
    """Unit vectors + a similarity threshold for 'near duplicate'.
    Uses sentence-transformers if installed (better, multilingual), else a hashed bag of words (no deps)."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return m.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype("float32"), 0.90
    except ImportError:
        pass
    X = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        w = re.findall(r"\w+", t.lower())
        for g in w + [" ".join(p) for p in zip(w, w[1:])]:
            X[i, zlib.crc32(g.encode()) % dim] += 1
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)
    return X, 0.85


# ---------- plans ----------
def parse_plan(raw):  # first JSON object in a model output (tolerates fences, <think> blocks, chatter)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    i = raw.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(raw[i:])[0]
    except json.JSONDecodeError:
        return None


def deps(step):
    d = step.get("depends_on")
    return [] if d is None else [d] if isinstance(d, str) else list(d)


def validate(plan):
    """List of problems; empty list = valid."""
    if not isinstance(plan, dict) or not isinstance(plan.get("threads"), list) or not plan["threads"]:
        return ["no threads"]
    errs, seen = [], set()
    for ti, th in enumerate(plan["threads"]):
        steps = th.get("steps") if isinstance(th, dict) else None
        if not steps or not isinstance(steps, list):
            errs.append(f"thread {ti}: no steps")
            continue
        local = set()
        for s in steps:
            if not isinstance(s, dict) or s.get("tool") not in TOOLS or not isinstance(s.get("input"), str):
                errs.append(f"thread {ti}: bad step {str(s)[:60]}")
                continue
            k, d = s.get("output_key"), s.get("depends_on")
            if not isinstance(k, str) or k in seen:
                errs.append(f"bad or duplicate output_key {k!r}")
            if not (d is None or isinstance(d, str) or (isinstance(d, list) and all(isinstance(x, str) for x in d))):
                errs.append(f"{k}: bad depends_on")
                continue
            if any(x not in local for x in deps(s)):
                errs.append(f"{k}: depends on unknown / later / other-thread step")
            seen.add(k)
            local.add(k)
    return errs


def reduced(steps):
    """Per step: dependency indices after transitive reduction (drops edges implied by other edges)."""
    idx = {s["output_key"]: i for i, s in enumerate(steps)}
    d = [sorted(idx[x] for x in deps(s)) for s in steps]
    anc = []
    for i in range(len(steps)):  # validated plans are topologically ordered
        a = set()
        for j in d[i]:
            a |= {j} | anc[j]
        anc.append(a)
    return [[j for j in d[i] if not any(j in anc[k] for k in d[i] if k != j)] for i in range(len(steps))]


def canon(plan):
    """Wording- and order-independent form of a VALID plan. Threads are kept as a multiset (1 vs 2 threads matters);
    inside a thread, steps are (tool, dep-tools) nodes as a SET, so 2 vs 3 parallel web_search steps count as the same plan."""
    out = []
    for th in plan["threads"]:
        st, r = th["steps"], reduced(th["steps"])
        out.append(sorted({(s["tool"], tuple(sorted({st[j]["tool"] for j in r[i]}))) for i, s in enumerate(st)}))
    return json.dumps(sorted(out))


def drop_summarize(plan):
    """AIDE appends `summarize` to every thread itself, so a planner must never emit it. Remove such steps; steps that
    depended on one inherit its dependencies. Malformed plans are returned unchanged (validate() rejects them)."""
    try:
        out = json.loads(json.dumps(plan))
        threads = []
        for th in out["threads"]:
            removed = {s.get("output_key"): deps(s) for s in th["steps"] if s.get("tool") == "summarize"}

            def expand(keys, seen=()):
                res = []
                for k in keys:
                    for x in (expand(removed[k], seen + (k,)) if k in removed and k not in seen else [k]):
                        if x not in res:
                            res.append(x)
                return res

            steps = []
            for s in th["steps"]:
                if s.get("tool") == "summarize":
                    continue
                if s.get("depends_on") is not None:
                    ds = expand(deps(s))
                    s["depends_on"] = ds[0] if len(ds) == 1 else ds or None
                steps.append(s)
            if steps:
                th["steps"] = steps
                threads.append(th)
        out["threads"] = threads
        return out
    except (TypeError, AttributeError, KeyError):
        return plan


def canon_of_raw(raw):  # canonical form of a raw model output (summarize ignored), or None if it isn't a valid plan
    p = parse_plan(raw or "")
    p = drop_summarize(p) if p is not None else None
    return canon(p) if p is not None and not validate(p) else None


def tools_of(plan):  # per-thread presence: in how many threads each tool appears
    return Counter(t for th in plan["threads"] for t in {s["tool"] for s in th["steps"]})


def edges_of(plan):  # (dep_tool, tool) edges after reduction, each counted once per thread
    c = Counter()
    for th in plan["threads"]:
        st, r = th["steps"], reduced(th["steps"])
        c.update({(st[j]["tool"], s["tool"]) for i, s in enumerate(st) for j in r[i]})
    return c


def f1(a, b):  # multiset F1; two empty sets count as a perfect match
    if not a and not b:
        return 1.0
    tp = sum((a & b).values())
    p, r = tp / max(sum(a.values()), 1), tp / max(sum(b.values()), 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def compare(pred, ref):
    """Score a predicted plan against a reference plan (None pred = unparseable)."""
    if pred is None or validate(pred):
        return dict(valid=0, exact=0, tool_f1=0.0, edge_f1=0.0, threads_ok=0)
    return dict(valid=1, exact=int(canon(pred) == canon(ref)),
                tool_f1=f1(tools_of(pred), tools_of(ref)), edge_f1=f1(edges_of(pred), edges_of(ref)),
                threads_ok=int(len(pred["threads"]) == len(ref["threads"])))


def explain_case(plan, case):
    """Violated gold constraints as readable strings (empty list = the plan passes)."""
    if plan is None:
        return ["unparseable output"]
    errs = validate(plan)
    if errs:
        return [f"invalid plan: {errs[0]}"]
    used, n, bad = set(tools_of(plan)), len(plan["threads"]), []
    if set(case.get("must_use", [])) - used:
        bad.append(f"missing {sorted(set(case['must_use']) - used)}")
    if used & set(case.get("must_not_use", [])):
        bad.append(f"forbidden {sorted(used & set(case['must_not_use']))} used")
    bad += [f"none of {g} used" for g in case.get("must_use_any", []) if not used & set(g)]
    lo, hi = case.get("min_threads", 1), case.get("max_threads", 99)
    if not lo <= n <= hi:
        bad.append(f"{n} threads (allowed {lo}-{hi})")
    for a, b in case.get("ordering", []):  # in every thread holding both tools, a comes first
        for th in plan["threads"]:
            seq = [s["tool"] for s in th["steps"]]
            if a in seq and b in seq and seq.index(a) > seq.index(b):
                bad.append(f"{a} after {b}")
                break
    return bad


def check_case(plan, case):
    """Gold-set constraints (PlannerCase-style + ordering + any-of): True if the plan satisfies all of them."""
    return not explain_case(plan, case)


# ---------- teacher-label hygiene (from the planner prompt's own rules) ----------
def _ancestors(st):
    idx, anc = {s["output_key"]: i for i, s in enumerate(st)}, []
    for s in st:
        a = set()
        for k in deps(s):
            a |= {idx[k]} | anc[idx[k]]
        anc.append(a)
    return anc


def lint(plan):
    """The terminal tool's rule: ALWAYS plan web_search and deep_think before it, unless the step just runs a local file
    that file_search already found (a file_search ancestor is our proxy for that). Returns violated rule names."""
    bad = set()
    for th in plan["threads"]:
        st, anc = th["steps"], _ancestors(th["steps"])
        for i, s in enumerate(st):
            if s["tool"] != "terminal":
                continue
            before = {st[j]["tool"] for j in anc[i]}
            if "file_search" in before:
                continue
            if "web_search" not in before:
                bad.add("terminal_without_web_search")
            if "deep_think" not in before:
                bad.add("terminal_without_deep_think")
    return sorted(bad)


def repair_terminal(plan):
    """Insert the web_search and/or deep_think that the terminal rule requires (see lint) instead of dropping the example.
    Returns (new plan, number of threads repaired)."""
    out, repaired = json.loads(json.dumps(plan)), 0
    fmt = lambda ks: None if not ks else ks[0] if len(ks) == 1 else ks
    for th in out["threads"]:
        changed = False
        while True:
            st, anc = th["steps"], _ancestors(th["steps"])
            bad = next((s for i, s in enumerate(st) if s["tool"] == "terminal"
                        and "file_search" not in {st[j]["tool"] for j in anc[i]}
                        and not {"web_search", "deep_think"} <= {st[j]["tool"] for j in anc[i]}), None)
            if bad is None:
                break
            before = {st[j]["tool"] for j in anc[st.index(bad)]}
            keys = {s["output_key"] for h in out["threads"] for s in h["steps"]}

            def fresh(base):
                while base in keys:
                    base += "x"
                keys.add(base)
                return base

            extra = []
            if "web_search" not in before:
                k = fresh(bad["output_key"] + "_ws")
                st.insert(0, {"tool": "web_search", "input": f"how to {bad['input']} command line", "output_key": k,
                              "depends_on": None})
                extra.append(k)
            if "deep_think" not in before:
                k = fresh(bad["output_key"] + "_dt")
                st.insert(st.index(bad), {"tool": "deep_think", "input": f"What is the exact command to do this: {bad['input']}",
                                          "output_key": k, "depends_on": fmt(deps(bad) + extra)})
                extra.append(k)
            bad["depends_on"] = fmt(deps(bad) + extra)
            changed = True
        repaired += changed
    return out, repaired


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_YEAR = re.compile(rf"\s*(?:\b(?:as of|in|for|of|during|by|from)\s+)?(?:(?:{_MONTHS})\s+)?(?<![/\w.-])(202[3-9]|2030)\b", re.I)


def strip_years(plan, request):
    """The teacher had no date and wrote guessed years ('weather today 2025'). Remove recent years it invented from
    goals and step inputs; years the user wrote themselves are kept. Returns a new plan."""
    keep = set(re.findall(r"\b20\d\d\b", request))
    fix = lambda s: re.sub(r"\s+", " ", _YEAR.sub(lambda m: m.group(0) if m.group(1) in keep else "", s)).strip()
    out = json.loads(json.dumps(plan))
    for th in out["threads"]:
        th["goal"] = fix(th.get("goal", ""))
        for s in th["steps"]:
            s["input"] = fix(s["input"])
    return out


def length_batches(lengths, bs, rng=None):
    """Batches of indices with similar lengths (less padding). With rng: shuffled training order; without: sorted."""
    idx = list(range(len(lengths)))
    if rng is None:
        idx.sort(key=lambda i: lengths[i])
        return [idx[i:i + bs] for i in range(0, len(idx), bs)]
    rng.shuffle(idx)
    chunks = [sorted(idx[i:i + bs * 32], key=lambda j: lengths[j]) for i in range(0, len(idx), bs * 32)]
    out = [c[i:i + bs] for c in chunks for i in range(0, len(c), bs)]
    rng.shuffle(out)
    return out


# ---------- compact target (what the student learns to emit) ----------
def to_compact(plan):
    """THREAD <goal> / n|tool|input|dep,dep   (keys are re-derived on render)."""
    lines = []
    for th in plan["threads"]:
        lines.append("THREAD " + re.sub(r"\s+", " ", th.get("goal", "")).strip())
        idx = {s["output_key"]: i for i, s in enumerate(th["steps"], 1)}
        for i, s in enumerate(th["steps"], 1):
            inp = re.sub(r"\s+", " ", s["input"]).strip()
            lines.append(f"{i}|{s['tool']}|{inp}|{','.join(str(idx[x]) for x in deps(s))}")
    return "\n".join(lines)


def from_compact(text):
    """Inverse of to_compact; returns a full plan dict or None if malformed."""
    threads = []
    for line in text.strip().splitlines():
        if line.startswith("THREAD"):
            threads.append({"goal": line[6:].strip(), "steps": []})
            continue
        if not threads:
            return None
        try:
            n, tool, rest = line.split("|", 2)
            inp, d = rest.rsplit("|", 1)
        except ValueError:
            return None
        ds = [f"t{len(threads)}_{x}" for x in d.split(",") if x]
        threads[-1]["steps"].append({"tool": tool, "input": inp, "output_key": f"t{len(threads)}_{n}",
                                     "depends_on": None if not ds else ds[0] if len(ds) == 1 else ds})
    return {"threads": threads} if threads else None