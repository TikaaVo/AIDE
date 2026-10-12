"""Gold set helper: fill in a spreadsheet, get a validated data/gold/gold.jsonl.

    python gold.py init requests.txt   # add requests (one per line) to data/gold/gold.csv; existing rows are kept
    python gold.py build               # validate gold.csv -> gold.jsonl + coverage report

gold.csv columns (open in LibreOffice/Excel, save as "CSV UTF-8", keep the header):
  id            g001, g002, ... (auto)
  text          the request
  must_use      tools that MUST appear, space-separated        file_search read_file
  must_not_use  tools that must NOT appear                      terminal open_url
  min_threads   blank = 1
  max_threads   blank = 99
  ordering      a>b pairs: in every thread containing both, a comes first      file_search>read_file
  any_of        "/" inside a group = at least one of them; space-separate groups    chat/web_search
  tags          free labels, used later for per-category scores    negation typo italian open_url
  note          anything (why you chose these constraints, doubts)
Only constrain what you're sure of. A row with no constraints at all is reported as TODO.
"""
import csv, re, sys
from collections import Counter

from common import DATA, TOOLS, write_jsonl

CSV, JSONL = DATA / "gold" / "gold.csv", DATA / "gold" / "gold.jsonl"
COLS = ["id", "text", "must_use", "must_not_use", "min_threads", "max_threads", "ordering", "any_of", "tags", "note"]
words = lambda s: [w for w in re.split(r"[,\s]+", (s or "").strip()) if w]


def load():
    if not CSV.exists():
        return []
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("text") or "").strip()]


def save(rows):
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel shows accents correctly
        w = csv.DictWriter(f, COLS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def init(src):
    rows = load()
    have = {r["text"].strip() for r in rows}
    n = max([int(r["id"][1:]) for r in rows if re.fullmatch(r"g\d+", r.get("id") or "")] + [0])
    added = 0
    for t in (l.strip() for l in open(src, encoding="utf-8")):
        if t and t not in have:
            n += 1
            added += 1
            have.add(t)
            rows.append({"id": f"g{n:03d}", "text": t})
    save(rows)
    print(f"added {added} rows ({len(rows)} total) -> {CSV}")


def build():
    out, errs, todo, seen = [], [], [], Counter()
    for r in load():
        i, text = r.get("id") or "?", r["text"].strip()
        seen[text] += 1
        must, mnot = words(r.get("must_use")), words(r.get("must_not_use"))
        order = [p.split(">") for p in words(r.get("ordering"))]
        anyof = [g.split("/") for g in words(r.get("any_of"))]
        for t in must + mnot + [x for p in order for x in p] + [x for g in anyof for x in g]:
            if t not in TOOLS:
                errs.append(f"{i}: unknown tool {t!r} (valid: {', '.join(sorted(TOOLS))})")
        if any(len(p) != 2 for p in order):
            errs.append(f"{i}: ordering needs a>b pairs, got {r.get('ordering')!r}")
        if set(must) & set(mnot):
            errs.append(f"{i}: {sorted(set(must) & set(mnot))} is both must_use and must_not_use")
        try:
            lo, hi = int(r.get("min_threads") or 1), int(r.get("max_threads") or 99)
        except ValueError:
            errs.append(f"{i}: min/max_threads must be whole numbers")
            continue
        if lo > hi:
            errs.append(f"{i}: min_threads {lo} > max_threads {hi}")
        if not (must or mnot or order or anyof or lo > 1 or hi < 99):
            todo.append(i)
        out.append(dict(id=i, text=text, must_use=must, must_not_use=mnot, min_threads=lo, max_threads=hi,
                        ordering=order, must_use_any=anyof, tags=words(r.get("tags")), note=(r.get("note") or "").strip()))
    errs += [f"duplicate text x{n}: {t[:60]}" for t, n in seen.items() if n > 1]
    if errs:
        sys.exit("NOT written, fix these first:\n  " + "\n  ".join(errs))
    write_jsonl(JSONL, out)
    uses = Counter(t for r in out for t in r["must_use"] + [x for g in r["must_use_any"] for x in g])
    print(f"wrote {len(out)} rows -> {JSONL}")
    print("tools required (must_use / any_of):", {t: uses.get(t, 0) for t in sorted(TOOLS)})
    print("rows with min_threads>1:", sum(r["min_threads"] > 1 for r in out),
          "| with ordering:", sum(bool(r["ordering"]) for r in out),
          "| with negative constraints:", sum(bool(r["must_not_use"]) for r in out))
    print("tags:", dict(Counter(t for r in out for t in r["tags"])) or "none yet")
    if todo:
        print(f"TODO: {len(todo)} rows have no constraints yet (they would pass any plan): {', '.join(todo[:15])}"
              + (" ..." if len(todo) > 15 else ""))


if __name__ == "__main__":
    {"init": lambda: init(sys.argv[2]), "build": build}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()