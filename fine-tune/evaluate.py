"""Score planners on the gold set (hand-written constraints) and the validation split (agreement with the teacher).

    python evaluate.py teacher                        # the teacher's score on gold, from data/teacher_gold.jsonl (no GPU)
    python evaluate.py model                          # the BASE model with the same prompt: the "before" number
    python evaluate.py model --adapter out/planner    # the fine-tuned model: the "after" number

Writes results/<name>.json and results/<name>_gold_failures.jsonl (every failed request, and why).
"gold pass" = the plan is valid AND satisfies every constraint of that gold row.
"""
import argparse, json, os, time
from collections import defaultdict
from pathlib import Path

from common import DATA, STUDENT_PROMPT, compare, drop_summarize, explain_case, from_compact, parse_plan, read_jsonl, validate

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["teacher", "model"])
ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B"))
ap.add_argument("--adapter", default=None)
ap.add_argument("--file", default=str(DATA / "teacher_gold.jsonl"), help="teacher mode: which teacher output to score")
ap.add_argument("--name", default=None)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--no-val", action="store_true")
ap.add_argument("--no-gold", action="store_true", help="validation only (use this to compare checkpoints; keep gold for the end)")
args = ap.parse_args()
name = args.name or ("teacher" if args.mode == "teacher" else Path(args.adapter).name if args.adapter else "base")
Path("results").mkdir(exist_ok=True)
gold = read_jsonl(DATA / "gold" / "gold.jsonl")
res = {}


def score_gold(plans, outputs):
    by_tag, rows = defaultdict(lambda: [0, 0]), []
    for g, plan, out in zip(gold, plans, outputs):
        why = explain_case(plan, g)
        rows.append(dict(id=g["id"], text=g["text"], why=why, output=out))
        for t in g.get("tags", []) + ["ALL"]:
            by_tag[t][0] += not why
            by_tag[t][1] += 1
    passed, valid = by_tag["ALL"][0], sum(p is not None and not validate(p) for p in plans)
    print(f"\n=== GOLD ({len(gold)} requests) ===")
    print(f"pass {passed}/{len(gold)} = {100 * passed / len(gold):.1f}%   |   valid plans {valid}/{len(gold)}")
    print("by tag: " + "  ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(by_tag.items(), key=lambda kv: -kv[1][1]) if t != "ALL"))
    fails = [r for r in rows if r["why"]]
    with open(f"results/{name}_gold_failures.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in fails)
    print(f"\nfirst failures (all {len(fails)} in results/{name}_gold_failures.jsonl):")
    for r in fails[:12]:
        print(f"  {r['id']}  {r['text'][:70]!r}\n        -> {'; '.join(r['why'])}")
    res["gold_pass"], res["gold_valid"] = passed / len(gold), valid / len(gold)
    res["gold_by_tag"] = {t: f"{a}/{b}" for t, (a, b) in by_tag.items()}


def run_teacher():
    recs = {}
    for r in read_jsonl(args.file):
        recs.setdefault(r["id"], r)
    missing = [g["id"] for g in gold if g["id"] not in recs]
    if missing:
        print(f"warning: no teacher output for {len(missing)} gold rows (e.g. {missing[:5]}); they count as failures")
    outs = [recs[g["id"]]["raw"] if g["id"] in recs else "" for g in gold]
    score_gold([drop_summarize(p) if p else p for p in map(parse_plan, outs)], outs)
    toks = [r["tokens"] for r in recs.values()]
    print(f"\nteacher output: mean {sum(toks) / max(len(toks), 1):.0f} tokens per plan (full JSON format)")


def run_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    @torch.no_grad()
    def generate(prompts, bs):
        tok.padding_side = "left"
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs = [""] * len(prompts)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, max_new_tokens=1024, do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
            for i, o in zip(idx, out[:, enc.input_ids.shape[1]:]):
                outs[i] = tok.decode(o, skip_special_tokens=True)
            print(f"  generated {min(s + bs, len(order))}/{len(order)}", end="\r", flush=True)
        print()
        return outs

    if not args.no_gold:
        outs = generate([STUDENT_PROMPT.format(g["text"]) for g in gold], args.bs)
        score_gold([from_compact(o) for o in outs], outs)

    if not args.no_val:
        val = read_jsonl(DATA / "val.jsonl")
        vouts = generate([r["prompt"] for r in val], args.bs)
        m, by_n = defaultdict(float), defaultdict(lambda: [0, 0])
        for r, o in zip(val, vouts):
            cmp = compare(from_compact(o), r["plan"])
            for k, v in cmp.items():
                m[k] += v / len(val)
            n = by_n["1" if r["n_threads"] == 1 else "2" if r["n_threads"] == 2 else "3+"]
            n[0] += cmp["exact"]
            n[1] += 1
        print(f"\n=== VALIDATION ({len(val)} requests, reference = teacher plan) ===")
        print(f"valid {100 * m['valid']:.1f}%  |  exact structure {100 * m['exact']:.1f}%  |  thread count {100 * m['threads_ok']:.1f}%  "
              f"|  tool F1 {m['tool_f1']:.3f}  |  edge F1 {m['edge_f1']:.3f}")
        print("exact structure by thread count: " + "  ".join(f"{k}: {a}/{b}" for k, (a, b) in sorted(by_n.items())))
        res["val"], res["val_exact_by_threads"] = dict(m), {k: f"{a}/{b}" for k, (a, b) in by_n.items()}

    if args.no_gold:
        return
    prompts = [STUDENT_PROMPT.format(g["text"]) for g in gold[:10]]  # speed with batch size 1, like a real request
    t0, n_tok = time.time(), 0
    for p in prompts:
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1024, do_sample=False, pad_token_id=tok.pad_token_id)
        n_tok += out.shape[1] - enc.input_ids.shape[1]
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\nspeed (batch size 1, {dev}): {dt / len(prompts):.2f} s per plan, {n_tok / dt:.0f} tokens/s")
    res["seconds_per_plan"], res["tokens_per_second"] = dt / len(prompts), n_tok / dt


run_teacher() if args.mode == "teacher" else run_model()
json.dump(res, open(f"results/{name}.json", "w"), indent=2)
print(f"\nsaved results/{name}.json")