"""Compose multi-intent examples from single-intent ones: data/train.jsonl -> data/train_aug.jsonl.

Joining independent requests gives the union of their threads, so the labels are exact by construction (the compact
format numbers steps per thread, so plans simply concatenate). This targets the planner's main weakness, merging
several sub-tasks into one thread. Uses TRAIN rows only: never val, never gold.

    python augment.py          # 1200 composed examples
    N=2000 python augment.py
Then train with:  AUG=1 python train.py
"""
import os
import random
import re
from collections import Counter

from common import DATA, STUDENT_PROMPT, norm_key, read_jsonl, write_jsonl

N = int(os.environ.get("N", 1200))
rng = random.Random(0)
EN = ["Also, ", "And ", "Oh and ", "Then ", "Plus, ", "While you're at it, ", "Another thing: ", "And also, ", ""]
IT = ["Inoltre, ", "E poi ", "Poi ", "Già che ci sei, ", "E anche ", "Ah, e ", ""]
GREETING = re.compile(r"^(hey|hi|hello|yo|ciao)\b[\s,!.:-]*", re.I)


def lang(r):
    return r.get("lang") or ("it" if (r.get("meta") or {}).get("style") == "italian" else "en")


def lower_first(s):  # "Find my notes" -> "find my notes", but keep "I ..." and acronyms
    return s if re.match(r"(I\b|I'|[A-Z]{2,})", s) else s[:1].lower() + s[1:]


def join(texts):
    out = texts[0].strip()
    for t in texts[1:]:
        t = GREETING.sub("", t.strip()) or t.strip()
        c = rng.choice(EN if lang_now == "en" else IT)
        if not out.endswith((".", "?", "!")):
            out += "."
        out += " " + (c + lower_first(t) if c else t[:1].upper() + t[1:])
    return out


# single-thread, tool-using, non-dependency examples from the TRAIN split
rows = [r for r in read_jsonl(DATA / "train.jsonl")
        if r["split"] == "train" and r["n_threads"] == 1 and not r["chat_only"]
        and (r.get("meta") or {}).get("twist") in (None, "irrelevant_detail", "vague")]
by_lang = {lg: [r for r in rows if lang(r) == lg] for lg in ("en", "it")}
print(f"{len(rows)} usable single-thread examples ({len(by_lang['en'])} en, {len(by_lang['it'])} it)")

out, seen, tries = [], set(), 0
while len(out) < N and tries < N * 30:
    tries += 1
    lang_now = "it" if rng.random() < 0.1 and len(by_lang["it"]) >= 8 else "en"
    k = rng.choices([2, 3, 4], [45, 40, 15])[0]
    picks = rng.sample(by_lang[lang_now], k)
    ints = [set((p.get("meta") or {}).get("intents", [])) for p in picks]
    if any(a & b for i, a in enumerate(ints) for b in ints[i + 1:]):  # never two requests of the same kind
        continue
    text = join([p["text"] for p in picks])
    if norm_key(text) in seen:
        continue
    seen.add(norm_key(text))
    out.append(dict(id=f"comp_{len(out):05d}", split="train", source="composed", lang=lang_now, text=text,
                    prompt=STUDENT_PROMPT.format(text), completion="\n".join(p["completion"] for p in picks),
                    n_threads=k, chat_only=False, meta=dict(composed_from=[p["id"] for p in picks])))

write_jsonl(DATA / "train_aug.jsonl", out)
print(f"wrote {len(out)} composed examples -> {DATA / 'train_aug.jsonl'}")
print("threads:", dict(sorted(Counter(r["n_threads"] for r in out).items())), "| italian:", sum(r["lang"] == "it" for r in out))
for r in rng.sample(out, min(4, len(out))):
    print("\n" + r["text"] + "\n  -> " + r["completion"].replace("\n", " // ")[:300])