"""data/raw/*.jsonl -> data/pool.jsonl: clean, length-filter, exact dedup, decontaminate vs gold, near dedup, split.

Gold texts are read from data/gold/*.jsonl ("text" field). If cases.py (your PlannerCase benchmark) is importable,
its queries are treated as gold too. Stats go to data/pool_stats.json (reportable curation numbers).
"""
import json, random, re
from collections import Counter

import numpy as np

from common import DATA, bucket, clean, norm_key, read_jsonl, vectors, write_jsonl

MIN_W, MAX_W = 3, 120   # words
VAL_PCT = 10            # % of pool held out as validation (by id hash)
PUBLIC_CAP = 500        # per public source (not synth/mine): tool-like requests first, then a random fill of the rest
TOOLISH = re.compile(r"\b(files?|folders?|director(y|ies)|open|run|install|script|search|look ?up|find|weather|forecast|"
                     r"web ?site|web ?page|url|link|latest|current|today|tomorrow|news|price|download|terminal|command|my|"
                     r"cartella|apri|installa|meteo|cerca|trova|sito|mio|mia|miei)\b", re.I)
GREETING_STRIP = 85     # % of synth requests whose leading "hey/hi/yo" gets stripped (by id hash)
GREETING = re.compile(r"^(hey|hi|hello|yo|ciao)\b(\s+(there|fam|dude|man|assistant))?\s*[,!.:-]*\s+", re.I)
LEAK = re.compile(r"not a (web|local|website)|\(a (local|website)", re.I)  # synth.py's intent hints, copied verbatim

rows = [r for p in sorted((DATA / "raw").glob("*.jsonl")) for r in read_jsonl(p)]
stats = {"raw": len(rows)}

for r in rows:
    r["text"] = clean(r["text"])
stats["leak_dropped"] = stats["greeting_stripped"] = 0
kept_rows = []
for r in rows:  # synth-only fixes: generator artifacts (hint leaks, ~50% of requests opening with "Hey")
    if r["source"] == "synth":
        if LEAK.search(r["text"]):
            stats["leak_dropped"] += 1
            continue
        m = GREETING.match(r["text"])
        if m and len(r["text"]) > m.end() and bucket(r["id"]) < GREETING_STRIP:
            rest = r["text"][m.end():]
            r["text"] = rest[0].upper() + rest[1:] if r["text"][0].isupper() else rest
            stats["greeting_stripped"] += 1
    kept_rows.append(r)
rows = [r for r in kept_rows if MIN_W <= len(r["text"].split()) <= MAX_W]
stats["length_filtered"] = len(rows)

seen, out = set(), []
for r in rows:  # exact dedup
    k = norm_key(r["text"])
    if k and k not in seen:
        seen.add(k)
        out.append(r)
rows = out
stats["exact_deduped"] = len(rows)

refs = [clean(g["text"]) for p in (DATA / "gold").glob("*.jsonl") for g in read_jsonl(p)]
try:
    import cases
    refs += [c.query for c in cases.PLANNER_CASES] + list(getattr(cases, "TURN_CASES", []))
except ImportError:
    print("note: cases.py not importable, decontaminating against data/gold only")
stats["gold_refs"] = len(refs)

X, thr = vectors([r["text"] for r in rows] + refs)
P, R = X[:len(rows)], X[len(rows):]
dup_thr = round(thr + 0.05, 2)  # near-dup removal is stricter than decontamination: "find and READ x" vs "find and OPEN x" must both survive
ex_decon, ex_dup = [], []
if len(R):  # decontaminate: drop anything close to a gold/benchmark request
    S = P @ R.T
    hit = S.max(1) >= thr
    ex_decon = [(rows[i]["text"], refs[int(S[i].argmax())]) for i in np.where(hit)[0][:6]]
    rows = [r for r, h in zip(rows, hit) if not h]
    P = P[~hit]
stats["decontaminated"] = len(rows)

langs = [r.get("lang", "en") for r in rows]  # near-dups are removed within one language only (an Italian request is not a dup of its English twin)
buf = {g: [np.empty((langs.count(g), P.shape[1]), P.dtype), []] for g in set(langs)}
keep = []
for i, g in enumerate(langs):  # greedy, O(n^2) per language; fine up to ~10k
    K, idx = buf[g]
    sims = K[:len(idx)] @ P[i]
    if not idx or sims.max() < dup_thr:
        K[len(idx)] = P[i]
        idx.append(i)
        keep.append(i)
    elif len(ex_dup) < 400:
        ex_dup.append((rows[i]["text"], rows[idx[int(sims.argmax())]]["text"]))
rows = [rows[i] for i in keep]
stats["near_deduped"] = len(rows)
stats["sim_threshold"], stats["dedup_threshold"] = thr, dup_thr

by_src = {}
for r in rows:
    by_src.setdefault(r["source"], []).append(r)
rows, stats["public_toolish_before_cap"] = [], {}
for src, rs in by_src.items():
    if src in ("synth", "mine"):  # your own data is never capped
        rows += rs
        continue
    rs.sort(key=lambda r: (not TOOLISH.search(r["text"]), bucket(r["id"], 10**6)))
    stats["public_toolish_before_cap"][src] = f"{sum(bool(TOOLISH.search(r['text'])) for r in rs)}/{len(rs)}"
    rows += rs[:PUBLIC_CAP]
stats["after_public_cap"] = len(rows)
stats["by_lang"] = dict(Counter(r.get("lang", "en") for r in rows))

def split_key(r):  # synth: same entities+intents in different styles must land in the same split, or val leaks into train
    m = r.get("meta") or {}
    return "|".join(sorted(m["slots"]) + sorted(m["intents"])) if m.get("slots") else r["id"]


for r in rows:
    r["split"] = "val" if bucket(split_key(r)) < VAL_PCT else "train"
stats["by_source"] = dict(Counter(r["source"] for r in rows))
stats["by_split"] = dict(Counter(r["split"] for r in rows))

write_jsonl(DATA / "pool.jsonl", rows)
json.dump(stats, open(DATA / "pool_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))

print("\n--- dropped as too close to GOLD (dropped | gold) ---")
for a, b in ex_decon:
    print(f"  {a}\n    | {b}")
print("\n--- random near-duplicates dropped (dropped | kept) ---")
for a, b in random.Random(0).sample(ex_dup, min(8, len(ex_dup))):
    print(f"  {a}\n    | {b}")