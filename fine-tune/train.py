"""LoRA fine-tune of ONE small causal LM on data/train.jsonl (request -> compact plan). Loss on the plan tokens only.

    SMOKE=1 python train.py     # tiny run (32 examples) to check that everything works; also runs on CPU
    python train.py             # Qwen2.5-0.5B, 3 epochs: minutes on a 12 GB GPU

Env: MODEL (default Qwen/Qwen2.5-0.5B; Qwen/Qwen2.5-1.5B also fits in 12 GB), OUT (default out/planner), EPOCHS, LR,
AUG=1 (also train on data/train_aug.jsonl from augment.py), SAVE_ALL=1 (keep every epoch in OUT/epoch<N>), RANK, DROPOUT,
BS (default 2) and ACCUM (default 8). Out of memory? Lower BS and raise ACCUM so that BS * ACCUM stays 16.
The adapter with the best validation loss is saved to OUT. Stop llama-server first: it is holding your VRAM.
"""
import math, os, random, time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, length_batches, read_jsonl

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
OUT = os.environ.get("OUT", "out/planner")
SMOKE = os.environ.get("SMOKE") == "1"
AUG, SAVE_ALL = os.environ.get("AUG") == "1", os.environ.get("SAVE_ALL") == "1"
RANK, DROPOUT = int(os.environ.get("RANK", 64)), float(os.environ.get("DROPOUT", 0.05))
EPOCHS = 1 if SMOKE else int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-4))
BS, ACCUM, MAXLEN = int(os.environ.get("BS", 2)), int(os.environ.get("ACCUM", 8)), 1024
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = random.Random(0)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token


def encode(r):  # (input ids, labels): loss only on the plan (+ end-of-text), never on the prompt
    p = tok(r["prompt"], add_special_tokens=False).input_ids
    c = tok(r["completion"], add_special_tokens=False).input_ids + [tok.eos_token_id]
    return p + c, [-100] * len(p) + c


def load(name, limit=None):
    enc = [encode(r) for r in read_jsonl(DATA / f"{name}.jsonl")[:limit]]
    keep = [e for e in enc if len(e[0]) <= MAXLEN]
    print(f"{name}: {len(keep)} examples ({len(enc) - len(keep)} dropped for length > {MAXLEN}), "
          f"mean {sum(len(e[0]) for e in keep) / max(len(keep), 1):.0f} tokens", flush=True)
    return keep


train, val = load("train", 32 if SMOKE else None), load("val", 16 if SMOKE else None)
if AUG:
    aug = load("train_aug", 32 if SMOKE else None)
    assert aug, "AUG=1 but data/train_aug.jsonl is missing or empty: run python augment.py first"
    train += aug


def collate(data, b):
    L = max(len(data[j][0]) for j in b)
    ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for k, j in enumerate(b):
        x, y = data[j]
        ids[k, :len(x)] = torch.tensor(x)
        lab[k, :len(y)] = torch.tensor(y)
        att[k, :len(x)] = 1
    return ids.to(dev), att.to(dev), lab.to(dev)


model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=DROPOUT, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def loss_of(ids, att, lab):  # summed loss over plan tokens, and how many plan tokens there were
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        logits = model(input_ids=ids, attention_mask=att).logits
    tgt = lab[:, 1:]
    sel = tgt != -100
    return F.cross_entropy(logits[:, :-1][sel].float(), tgt[sel], reduction="sum"), sel.sum()


@torch.no_grad()
def val_loss():
    model.eval()
    tot = n = 0
    for b in length_batches([len(e[0]) for e in val], BS):
        s, k = loss_of(*collate(val, b))
        tot, n = tot + s.item(), n + k.item()
    model.train()
    return tot / max(n, 1)


params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
total = EPOCHS * math.ceil(math.ceil(len(train) / BS) / ACCUM)  # optimizer steps
warm = max(1, int(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

best, t0 = float("inf"), time.time()
model.train()
for ep in range(1, EPOCHS + 1):
    run = seen = 0
    batches = length_batches([len(e[0]) for e in train], BS, rng)
    for i, b in enumerate(batches, 1):
        s, k = loss_of(*collate(train, b))
        (s / k / ACCUM).backward()
        run, seen = run + s.item(), seen + k.item()
        if i % ACCUM == 0 or i == len(batches):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if i % (ACCUM * 20) == 0:
            print(f"  epoch {ep} batch {i}/{len(batches)}  train loss {run / seen:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    v = val_loss()
    print(f"epoch {ep}: train loss {run / seen:.4f} | val loss {v:.4f} | {time.time() - t0:.0f}s", flush=True)
    if SAVE_ALL:
        model.save_pretrained(f"{OUT}/epoch{ep}")
        tok.save_pretrained(f"{OUT}/epoch{ep}")
    if v < best:
        best = v
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
        print(f"  saved adapter -> {OUT}", flush=True)
print(f"done. best val loss {best:.4f}. Next: python evaluate.py model --adapter {OUT}")