"""teacher.jsonl (+ teacher2.jsonl) + pool.jsonl -> data/train.jsonl, data/val.jsonl.

Which label a request gets:
  * public requests: at least 2 valid teacher samples must share the same canonical plan (agreement).
  * synth requests (we know the intents they were generated from): the agreed plan must ALSO satisfy INTENT_SPEC below.
    If it doesn't, or the samples never agreed, any valid sample that satisfies the stricter spec is used instead
    ("rescued"). Agreement filters noise; the spec catches mistakes the teacher makes consistently.
Hygiene on the chosen plan: years the teacher invented are stripped, a missing deep_think before terminal is inserted.
Requests with no usable label go to data/needs_resample.txt (feed it to teacher.py via ONLY=..., see teacher.py).
Stats: data/build_stats.json. Rejected plans to read: data/rejected.jsonl, data/disagreements.jsonl.
"""
import json, os, re
from collections import Counter, defaultdict

from common import (DATA, STUDENT_PROMPT, TOOLS, bucket, canon, canon_of_raw, drop_summarize, from_compact, lint,
                    parse_plan, read_jsonl, repair_terminal, strip_years, to_compact, tools_of, validate, write_jsonl)

MAX_CHAT = 0.20  # max share of chat-only requests in the final set
LINT = os.environ.get("LINT", "repair")                   # terminal without deep_think: repair | drop | off
STRIP_YEARS = os.environ.get("STRIP_YEARS", "1") == "1"   # remove years the teacher invented (it had no date)
SPEC = os.environ.get("SPEC", "1") == "1"                 # verify synth plans against INTENT_SPEC

# intent -> (required tools, extra tools that may appear). Edit freely: it is the planner policy in one place.
# From AIDE's tool descriptions: file_search finds local files; open_file launches a file for the USER (xdg-open, executes
# nothing); read_file lets AIDE read a local file; open_url fetches a page's text (optional: web_search results are often
# enough, so it is only allowed for the open_url intent); terminal ALWAYS needs web_search + deep_think before it unless it
# runs a file file_search found; deep_think is for complex/risky steps, never simple lookups; summarize is added by AIDE.
INTENT_SPEC = {
    "find_open": ({"file_search", "open_file"}, {"read_file"}),
    "find_run": ({"file_search", "terminal"}, {"read_file", "deep_think", "web_search"}),
    "find_read": ({"file_search", "read_file"}, {"deep_think", "web_search"}),
    "install": ({"web_search", "deep_think", "terminal"}, set()),
    "terminal": ({"web_search", "deep_think", "terminal"}, {"file_search"}),
    "weather": ({"web_search"}, set()),
    "lookup": ({"web_search"}, set()),
    "open_url": ({"open_url"}, {"web_search"}),
    "chat": (set(), {"chat", "web_search"}),
}


def spec_problems(plan, meta, strict):
    """Why a plan does not match the intents its request was generated from (empty list = fine).
    strict (used for rescues): also require exactly one thread per intent. Dependency requests may legitimately use
    extra tools and share threads; negation requests ("don't search...") may legitimately omit required tools."""
    req, allowed = set(), set()
    for i in meta["intents"]:
        r, a = INTENT_SPEC.get(i, (set(), set(TOOLS)))
        req |= r
        allowed |= r | a
    used, n, tw, bad = set(tools_of(plan)), len(plan["threads"]), meta.get("twist"), []
    if tw != "negation":
        bad += [f"missing {t}" for t in sorted(req - used)]
    if tw != "dependency":
        bad += [f"unexpected {t}" for t in sorted(used - allowed)]
        if n < len(meta["intents"]):
            bad.append("fewer threads than intents")
        if strict and n != len(meta["intents"]):
            bad.append("threads != intents")
    return bad


def clean(plan, text):
    """Hygiene on a valid plan -> (plan, years_stripped, repaired, dropped_by_lint)."""
    plan = drop_summarize(plan)  # AIDE appends summarize itself
    changed = False
    if STRIP_YEARS:
        new = strip_years(plan, text)
        changed, plan = new != plan, new
    viol = LINT != "off" and bool(lint(plan))
    if viol and LINT == "repair":
        plan, _ = repair_terminal(plan)
        return plan, changed, True, False
    return plan, changed, False, viol


def load_teacher(path):
    by = defaultdict(list)
    for r in read_jsonl(path):
        r["canon"] = canon_of_raw(r["raw"])  # recomputed: canonicalisation changes apply without re-sampling
        r["valid"] = r["canon"] is not None
        by[r["id"]].append(r)
    return by


pool = {r["id"]: r for r in read_jsonl(DATA / "pool.jsonl")}
by_id, t2 = load_teacher(DATA / "teacher.jsonl"), load_teacher(DATA / "teacher2.jsonl")
by_id.update(t2)  # re-sampled requests (teacher2.jsonl, e.g. with the tool glossary) replace their old samples

stats = Counter(requests=sum(i in pool for i in by_id), ignored_not_in_pool=sum(i not in pool for i in by_id),
                resampled_requests=sum(i in pool for i in t2))
kept, toks, dis, rejected, resample = [], [], [], [], []
tot, ok, use, reasons = Counter(), Counter(), Counter(), Counter()
for id_, recs in by_id.items():
    if id_ not in pool:
        continue
    req = pool[id_]
    meta = req.get("meta") or {}
    text, tw = req["text"], str(meta.get("twist"))
    has_spec = SPEC and bool(meta.get("intents"))
    tot[tw] += 1
    valid = [r for r in recs if r["valid"]]
    if not valid:
        stats["no_valid_plan"] += 1
        resample.append(id_)
        continue
    counts = Counter(r["canon"] for r in valid)
    c, n = counts.most_common(1)[0]
    order = sorted(valid, key=lambda r: -counts[r["canon"]])  # most common plan first (stable)

    def first_ok(cands, strict):  # first candidate that survives hygiene and (for synth) the spec
        first_why = None
        for r in cands:
            plan, yrs, rep, drop = clean(parse_plan(r["raw"]), text)
            bad = ["dropped by lint"] if drop else (spec_problems(plan, meta, strict) if has_spec else [])
            first_why = bad if first_why is None else first_why
            if not bad:
                return (r, plan, yrs, rep), []
        return None, first_why or []

    hit, how = None, None
    if n >= 2:
        stats["agreed"] += 1
        ok[tw] += 1
        hit, why = first_ok([r for r in order if r["canon"] == c], strict=False)
        how = "agreed"
        if hit is None and why:
            stats["majority_plan_rejected"] += 1
            reasons.update(why)
            rejected.append(dict(id=id_, text=text, intents=meta.get("intents"), twist=tw, reasons=why,
                                 plan=to_compact(drop_summarize(parse_plan(order[0]["raw"])))))
    if hit is None and has_spec:
        hit, _ = first_ok(order, strict=True)
        how = "rescued"
    if hit is None:
        if n < 2:
            stats["disagree"] += 1
            variants = {r["canon"]: drop_summarize(parse_plan(r["raw"])) for r in valid}
            kinds = set()
            for p, q in [(p, q) for i, p in enumerate(variants.values()) for q in list(variants.values())[i + 1:]]:
                kinds.add("thread_count" if len(p["threads"]) != len(q["threads"])
                          else "tool_set" if set(tools_of(p)) != set(tools_of(q)) else "structure_only")
            kind = next((k for k in ("thread_count", "tool_set", "structure_only") if k in kinds), "mostly_invalid")
            dis.append(dict(id=id_, text=text, twist=tw, kind=kind, variants=[to_compact(p) for p in variants.values()]))
        resample.append(id_)
        continue

    rec, plan, yrs, rep = hit
    compact = to_compact(plan)
    back = from_compact(compact)
    if back is None or validate(back) or canon(back) != canon(plan):
        stats["compact_roundtrip_fail"] += 1
        continue
    stats["years_stripped"] += yrs
    stats["lint_repaired"] += rep
    stats["rescued"] += how == "rescued"
    use[tw] += 1
    kept.append(dict(id=id_, split=req["split"], source=req["source"], text=text,
                     prompt=STUDENT_PROMPT.format(text), completion=compact, plan=plan, lang=req.get("lang", "en"),
                     n_threads=len(plan["threads"]), meta=meta, label=how,
                     chat_only=all(s["tool"] == "chat" for t in plan["threads"] for s in t["steps"])))
    toks.append(rec["tokens"])

chat = sorted((r for r in kept if r["chat_only"]), key=lambda r: bucket(r["id"], 10**6))
rest = [r for r in kept if not r["chat_only"]]
max_chat = int(MAX_CHAT / (1 - MAX_CHAT) * len(rest))
stats["chat_dropped_for_balance"] = max(len(chat) - max_chat, 0)
final = rest + chat[:max_chat]

stats["kept"] = len(final)
write_jsonl(DATA / "disagreements.jsonl", dis)
write_jsonl(DATA / "rejected.jsonl", rejected)
open(DATA / "needs_resample.txt", "w").write("\n".join(resample) + ("\n" if resample else ""))
stats["needs_resample"] = len(resample)
stats["spec_reject_reasons"] = dict(reasons.most_common(10))
stats["disagree_by_difference"] = dict(Counter(d["kind"] for d in dis))  # what the disagreeing samples differ in
stats["years_in_plans"] = dict(Counter(y for r in final for y in re.findall(r"\b20\d\d\b", r["completion"])))
stats["agreement_by_twist"] = {k: round(ok[k] / tot[k], 2) for k in tot}
stats["usable_by_twist"] = {k: round(use[k] / tot[k], 2) for k in tot}  # agreement + rescue
stats["agreement_rate"] = round(stats["agreed"] / max(stats["requests"], 1), 3)
stats["threads_dist"] = dict(sorted(Counter(r["n_threads"] for r in final).items()))
stats["labels"] = dict(Counter(r["label"] for r in final))
stats["by_source"] = dict(Counter(r["source"] for r in final))
stats["kept_tokens_mean_max"] = [round(sum(toks) / max(len(toks), 1)), max(toks, default=0)]  # kept plans only
allrecs = [r for i, recs in by_id.items() if i in pool for r in recs]  # only requests still in the pool
stats["samples"], stats["samples_per_request"] = len(allrecs), round(len(allrecs) / max(stats["requests"], 1), 2)
stats["empty_outputs"] = sum(not r["raw"].strip() for r in allrecs)  # many = thinking ate the token budget
stats["tokens_all_mean_max"] = [round(sum(r["tokens"] for r in allrecs) / max(len(allrecs), 1)),
                                max((r["tokens"] for r in allrecs), default=0)]
stats["finish_reasons"] = dict(Counter(r.get("finish", "?") for r in allrecs))

for split in ("train", "val"):
    write_jsonl(DATA / f"{split}.jsonl", [r for r in final if r["split"] == split])
    stats[split] = sum(r["split"] == split for r in final)
json.dump(stats, open(DATA / "build_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))