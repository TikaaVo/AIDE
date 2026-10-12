"""Pull request sources into data/raw/<name>.jsonl as {id, text, source, lang, license}.

    python fetch.py dolly oasst mine

mine: put your own requests in data/raw/mine.txt (one per line); written to data/raw/mine.jsonl.
Add a source = add a function returning a list of dicts with id/text/lang/license.
Check each dataset's license yourself before relying on it.
"""
import hashlib, random, sys

from common import DATA, write_jsonl

CAP = 3000  # per source; public sets are mostly off-topic for a desktop agent anyway
rng = random.Random(0)


def dolly():
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    return [dict(id=f"dolly_{i}", text=r["instruction"], lang="en", license="cc-by-sa-3.0")
            for i, r in enumerate(ds) if not r["context"]]  # no-context prompts only


def oasst():
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    return [dict(id=f"oasst_{r['message_id']}", text=r["text"], lang=r["lang"], license="apache-2.0")
            for r in ds if r["parent_id"] is None and r["lang"] in ("en", "it")]  # first user turns only


def mine():
    lines = [l.strip() for l in open(DATA / "raw" / "mine.txt", encoding="utf-8") if l.strip()]
    return [dict(id="mine_" + hashlib.sha1(t.encode()).hexdigest()[:8], text=t, lang="en", license="own")
            for t in lines]


SOURCES = dict(dolly=dolly, oasst=oasst, mine=mine)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        rows = SOURCES[name]()
        rng.shuffle(rows)
        rows = [dict(r, source=name, meta={}) for r in rows[:CAP]]
        write_jsonl(DATA / "raw" / f"{name}.jsonl", rows)
        print(f"{name}: wrote {len(rows)}")