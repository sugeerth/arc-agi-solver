"""
Build a single self-contained ARC-AGI notebook for Colab H100.

Output: arc_kiss.ipynb

Pipeline: download data -> SFT QLoRA on Qwen3-8B -> eval full ARC-AGI public eval
          -> write submission.json + bundle. End to end ~5-6h on H100 80GB.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "arc_kiss.ipynb"


def md(text): return ("md", text)
def code(text): return ("code", text)


def nb(cells):
    out_cells = []
    for kind, source in cells:
        src = source if isinstance(source, list) else [source]
        normalized = []
        for i, line in enumerate(src):
            if not line.endswith("\n") and i != len(src) - 1:
                line = line + "\n"
            normalized.append(line)
        if kind == "md":
            out_cells.append({"cell_type": "markdown", "metadata": {}, "source": normalized})
        else:
            out_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                              "outputs": [], "source": normalized})
    return {"cells": out_cells,
            "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                         "language_info": {"name":"python","version":"3.10"}},
            "nbformat": 4, "nbformat_minor": 5}


CELLS = [
    md("""# ARC-AGI End-to-End on a single H100

This is the only notebook you need. It:

1. Downloads ARC-AGI-1 + RE-ARC + (best-effort) BARC, with strict shape filtering so bad rows can't crash training.
2. QLoRA-fine-tunes **Qwen3-8B** on the augmented pool (D4 × color-perm).
3. Evaluates on the full 400-task ARC-AGI-1 public eval — both **greedy** and **D4 self-consistency vote** (the +5–10 pt inference trick).
4. Writes `submission.json` in the official Kaggle format.
5. Bundles every artifact into `<RUN_ID>.zip` and offers a Colab download link.

**Hardware:** H100 80 GB on Colab Pro+ (will auto-fallback to SDPA on A100/L4 if FlashAttention isn't there).
**Wall clock:** ~5–6 hours total (SFT ~2–3 h, greedy eval ~30 min, D4 vote ~2.5 h).
**Expected score:** 15–35% on ARC-AGI-1 public eval — paper-comparable for a single 8B + LoRA + no TTT.

Just hit `Runtime → Run all` and walk away.
"""),
    code("""# 1 — Hardware + install
import torch, time, json, random, os, copy
from pathlib import Path
assert torch.cuda.is_available(), "Set Runtime to a GPU (H100/A100/L4)."
GPU = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU} ({VRAM_GB:.0f} GB) | torch {torch.__version__}")
"""),
    code("""!pip -q install -U \\
    "transformers>=4.51,<4.55" "trl>=0.15,<0.20" "peft>=0.14" \\
    "accelerate>=1.4" "bitsandbytes>=0.45" "datasets>=3.2" \\
    sentencepiece huggingface_hub
"""),
    code("""# 2 — Run config (the only knobs to touch)
SEED            = 0
MODEL_ID        = "Qwen/Qwen3-8B"
BARC_CAP        = 5000        # how many BARC rows to use; 5k -> ~3h SFT
N_PER_TASK      = 4           # D4*color-perm augmentation factor
SFT_EPOCHS      = 1
MAX_LEN         = 3500
RUN_D4_VOTE     = True        # +5-10 pts but ~2.5h slower; set False for greedy-only
LIMIT_EVAL      = None        # None = all 400; set 50 for a quick smoke test

random.seed(SEED); torch.manual_seed(SEED)

ROOT    = Path("/content") if Path("/content").exists() else Path(".")
RUN_ID  = f"arc-{int(time.time())}"
RUN_DIR = ROOT / "runs" / RUN_ID; RUN_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "arc_data"; DATA_DIR.mkdir(exist_ok=True)
print("RUN_DIR:", RUN_DIR)
"""),
    code("""# 3 — Download ARC-AGI-1 (400 train + 400 public eval)
import urllib.request, zipfile, io
if not (DATA_DIR / "training").exists():
    print("Downloading ARC-AGI-1 ...")
    with urllib.request.urlopen("https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip") as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    src = DATA_DIR / "ARC-AGI-master" / "data"
    for kind in ["training", "evaluation"]:
        (DATA_DIR / kind).mkdir(exist_ok=True)
        for f in (src / kind).glob("*.json"):
            (DATA_DIR / kind / f.name).write_bytes(f.read_bytes())

arc_train = [json.loads(p.read_text()) for p in sorted((DATA_DIR/"training").glob("*.json"))]
arc_eval  = [(p.stem, json.loads(p.read_text())) for p in sorted((DATA_DIR/"evaluation").glob("*.json"))]
print(f"ARC: {len(arc_train)} train, {len(arc_eval)} eval")
"""),
    code("""# 4 — Download RE-ARC + BARC (best-effort; OK if either skips)
re_arc_dir = DATA_DIR / "re_arc"; re_arc_tasks = []
if not re_arc_dir.exists():
    try:
        with urllib.request.urlopen("https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip") as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(re_arc_dir)
    except Exception as e: print("RE-ARC skip:", e)
if re_arc_dir.exists():
    for fp in sorted(re_arc_dir.rglob("*.json"))[:5000]:
        try: re_arc_tasks.append(json.loads(fp.read_text()))
        except: pass
print("RE-ARC:", len(re_arc_tasks))

try:
    from datasets import load_dataset
    barc = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                        split="train")
    print("BARC:", len(barc))
except Exception as e:
    barc = None; print("BARC skip:", e)
"""),
    code("""# 5 — Defensive normalize + filter (this is the bulletproof data step)
def is_well_formed(task):
    \"\"\"Strict shape: {train:[{input:2D-list, output:2D-list}, ...], test:[...]}.\"\"\"
    try:
        for kind in ("train", "test"):
            xs = task.get(kind)
            if not isinstance(xs, list) or not xs: return False
            for e in xs:
                if not (isinstance(e, dict) and "input" in e and "output" in e): return False
                for k in ("input", "output"):
                    g = e[k]
                    if not (isinstance(g, list) and g and isinstance(g[0], list)): return False
        return True
    except Exception:
        return False

def coerce_examples(exs):
    \"\"\"Coerce BARC pair encodings (dict, list[2], tuple[2]) into [{'input', 'output'}].\"\"\"
    if not isinstance(exs, list): return None
    out = []
    for e in exs:
        if isinstance(e, dict) and "input" in e and "output" in e:
            inp, outp = e["input"], e["output"]
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            inp, outp = e[0], e[1]
        else:
            return None
        if not (isinstance(inp, list) and inp and isinstance(inp[0], list)): return None
        if not (isinstance(outp, list) and outp and isinstance(outp[0], list)): return None
        out.append({"input": inp, "output": outp})
    return out

def normalize_barc(row):
    train, test = None, None
    if "train" in row and "test" in row:
        train = coerce_examples(row["train"]); test = coerce_examples(row["test"])
    elif "examples" in row and "test" in row:
        train = coerce_examples(row["examples"]); test = coerce_examples(row["test"])
    elif "examples" in row:
        ex = coerce_examples(row["examples"])
        if ex and len(ex) >= 2: train, test = ex[:-1], [ex[-1]]
    return {"train": train, "test": test} if (train and test) else None

base_pool = [t for t in (list(arc_train) + list(re_arc_tasks)) if is_well_formed(t)]
print(f"After ARC + RE-ARC filter: {len(base_pool)}")

if barc is not None:
    added = 0
    for r in barc.select(range(min(BARC_CAP, len(barc)))):
        n = normalize_barc(r)
        if n is not None and is_well_formed(n):
            base_pool.append(n); added += 1
    print(f"BARC added (after coerce+filter): {added}")

print(f"FINAL base pool: {len(base_pool)}")
"""),
    code("""# 6 — Augmentation + format helpers
def rot90(g):  return [list(r) for r in zip(*g[::-1])]
def fliplr(g): return [r[::-1] for r in g]
D4 = [
    lambda g: g, rot90,
    lambda g: rot90(rot90(g)),
    lambda g: rot90(rot90(rot90(g))),
    fliplr,
    lambda g: rot90(fliplr(g)),
    lambda g: rot90(rot90(fliplr(g))),
    lambda g: rot90(rot90(rot90(fliplr(g)))),
]
INV = [0, 3, 2, 1, 4, 5, 6, 7]

def apply_d4(task, k):
    f = D4[k]
    return {"train":[{"input":f(e["input"]),"output":f(e["output"])} for e in task["train"]],
            "test": [{"input":f(e["input"]),"output":f(e["output"])} for e in task["test"]]}

def apply_color_perm(task, perm):
    def remap(g): return [[perm[c] for c in row] for row in g]
    return {"train":[{"input":remap(e["input"]),"output":remap(e["output"])} for e in task["train"]],
            "test": [{"input":remap(e["input"]),"output":remap(e["output"])} for e in task["test"]]}

SYSTEM = ("You are an expert ARC puzzle solver. Output only the test output grid as digits "
          "separated by spaces, one row per line. No explanations.")

def grid_to_str(g): return "\\n".join(" ".join(str(c) for c in row) for row in g)

def parse_grid(s):
    rows = []
    for line in s.strip().splitlines():
        line = line.strip()
        if not line: continue
        try: rows.append([int(c) for c in line.split()])
        except: return None
    if not rows or any(len(r) != len(rows[0]) for r in rows): return None
    return rows

def format_user(task):
    parts = ["You are solving an ARC puzzle. Each grid is a 2D array of digits 0-9 (colors).",
             "Find the transformation rule from the input/output examples, then apply it to the test input.", ""]
    for i, ex in enumerate(task["train"], 1):
        parts.append(f"Example {i} Input:");  parts.append(grid_to_str(ex["input"]))
        parts.append(f"Example {i} Output:"); parts.append(grid_to_str(ex["output"]))
        parts.append("")
    parts.append("Test Input:"); parts.append(grid_to_str(task["test"][0]["input"]))
    parts.append(""); parts.append("Test Output:")
    return "\\n".join(parts)

def format_assistant(task):
    return grid_to_str(task["test"][0]["output"])
"""),
    code("""# 7 — Build augmented training set
rng = random.Random(SEED)
augmented = []
for t in base_pool:
    for _ in range(N_PER_TASK):
        x = apply_d4(t, rng.randrange(8))
        perm = list(range(10)); rng.shuffle(perm)
        x = apply_color_perm(x, perm)
        augmented.append(x)
print(f"Augmented samples (pre-tokenize): {len(augmented)}")
"""),
    code("""# 8 — Load Qwen3-8B in 4-bit + LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
QWEN3_KW = {"enable_thinking": False} if "Qwen3" in MODEL_ID else {}

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True)
attn_impl = "flash_attention_2" if ("H100" in GPU or "A100" in GPU) else "sdpa"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map={"": 0},
    torch_dtype=torch.bfloat16, attn_implementation=attn_impl, trust_remote_code=True)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                 task_type="CAUSAL_LM",
                 target_modules=["q_proj","k_proj","v_proj","o_proj",
                                 "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
"""),
    code("""# 9 — Tokenize + length-filter SFT dataset
def to_chat_full(t):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(t)},
            {"role":"assistant","content":format_assistant(t)}]
    return tok.apply_chat_template(msgs, tokenize=False, **QWEN3_KW)

def to_chat_prompt(t):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(t)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **QWEN3_KW)

texts = []
for t in augmented:
    s = to_chat_full(t)
    if len(tok(s, add_special_tokens=False).input_ids) <= MAX_LEN:
        texts.append(s)
print(f"SFT samples after length filter: {len(texts)}")
assert len(texts) > 100, "Almost no samples survived. Check MAX_LEN or data."

from datasets import Dataset
sft_ds = Dataset.from_dict({"text": texts}).shuffle(seed=SEED)
"""),
    code("""# 10 — Train (QLoRA SFT, 1 epoch, paged 8-bit AdamW + grad-ckpt + bf16)
from trl import SFTTrainer, SFTConfig

cfg = SFTConfig(
    output_dir=str(RUN_DIR / "sft_ckpt"),
    num_train_epochs=SFT_EPOCHS,
    per_device_train_batch_size=2 if VRAM_GB >= 70 else 1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=20,
    save_steps=10_000, save_total_limit=1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_seq_length=MAX_LEN,
    dataset_text_field="text",
    packing=False, report_to="none", seed=SEED,
)
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=sft_ds, args=cfg)
trainer.train()
trainer.model.save_pretrained(str(RUN_DIR / "adapter"))
print("SFT done. Adapter:", RUN_DIR / "adapter")
"""),
    code("""# 11 — Greedy eval on the full ARC-AGI-1 public eval
from tqdm.auto import tqdm

@torch.inference_mode()
def predict_greedy(task, max_new=2048):
    text = to_chat_prompt(task)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return parse_grid(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))

eval_subset = arc_eval[:LIMIT_EVAL] if LIMIT_EVAL else arc_eval
greedy_preds = {}
g_correct = 0
for name, task in tqdm(eval_subset, desc="greedy eval"):
    pred = predict_greedy(task)
    greedy_preds[name] = pred
    if pred == task["test"][0]["output"]:
        g_correct += 1

n = len(eval_subset)
print(f"\\nGREEDY: {g_correct}/{n} = {g_correct/n:.1%}")
"""),
    code("""# 12 — D4 self-consistency vote (the +5-10pt inference trick)
def predict_d4_vote(task):
    counts = {}
    for k in range(8):
        try:
            t_aug = apply_d4(task, k)
            pred = predict_greedy(t_aug)
            if pred is None: continue
            inv = D4[INV[k]](pred)
            key = tuple(tuple(r) for r in inv)
            counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue
    if not counts: return None
    return [list(r) for r in max(counts, key=counts.get)]

d4_preds = {}; d_correct = 0
if RUN_D4_VOTE:
    for name, task in tqdm(eval_subset, desc="d4 vote"):
        pred = predict_d4_vote(task)
        d4_preds[name] = pred
        if pred == task["test"][0]["output"]:
            d_correct += 1
    print(f"\\nD4VOTE: {d_correct}/{n} = {d_correct/n:.1%}")
else:
    d4_preds = dict(greedy_preds)
    d_correct = g_correct
    print("D4 vote skipped (RUN_D4_VOTE=False).")
"""),
    code("""# 13 — Submission.json + score summary
submission = {}
for name, task in eval_subset:
    a1 = d4_preds.get(name) or [[0]]
    a2 = greedy_preds.get(name) or [[0]]
    submission[name] = [{"attempt_1": a1, "attempt_2": a2}]
(RUN_DIR / "submission.json").write_text(json.dumps(submission))

pass2 = sum(1 for name, t in eval_subset
            if (greedy_preds.get(name) == t["test"][0]["output"])
               or (d4_preds.get(name) == t["test"][0]["output"]))

scores = {
    "model": MODEL_ID,
    "run_id": RUN_ID,
    "n": n,
    "greedy_correct": g_correct, "greedy_acc": g_correct/n,
    "d4vote_correct": d_correct, "d4vote_acc": d_correct/n,
    "pass2_correct": pass2, "pass2_acc": pass2/n,
}
(RUN_DIR / "scores.json").write_text(json.dumps(scores, indent=2))

print("\\n" + "="*60)
print(f"  MODEL : {MODEL_ID}")
print(f"  RUN   : {RUN_ID}")
print(f"  TASKS : {n}")
print(f"  GREEDY: {g_correct}/{n} = {g_correct/n:.1%}")
print(f"  D4    : {d_correct}/{n} = {d_correct/n:.1%}")
print(f"  pass@2: {pass2}/{n} = {pass2/n:.1%}")
print("="*60)
"""),
    code("""# 14 — Bundle and download
import shutil
zpath = ROOT / f"{RUN_ID}.zip"
shutil.make_archive(str(zpath).removesuffix(".zip"), "zip", RUN_DIR)
print(f"\\nBundle: {zpath} ({zpath.stat().st_size/1e6:.1f} MB)")
print("Contents:")
print("  - adapter/        (LoRA adapter weights)")
print("  - submission.json (Kaggle-format predictions)")
print("  - scores.json     (greedy/D4/pass@2 numbers)")
print("  - sft_ckpt/       (training checkpoint, can delete)")

try:
    from google.colab import files
    files.download(str(zpath))
except Exception:
    print(f"\\nNot on Colab. Bundle is at: {zpath}")
"""),
    md("""## What to send back for analysis

Once it's done you'll have a zip file. Either:
- The `scores.json` printed above (one screenshot of the final summary box is enough)
- Or the full zip if you want me to dig into checkpoints / submission structure / etc.

## Common gotchas

- **OOM during SFT:** lower `BARC_CAP` to 2000, or `N_PER_TASK` to 2.
- **"No samples survived" assert hits in cell 9:** lower `MAX_LEN` to 2500.
- **Want a faster smoke test:** set `LIMIT_EVAL=20` in cell 2 — full pipeline finishes in ~3 hours.
- **Want greedy-only (skip D4 vote):** set `RUN_D4_VOTE=False` — saves ~2.5 hours.
- **BARC fails to download:** the cell skips it cleanly. You'll just train on ARC + RE-ARC, slightly weaker but still works.
"""),
]


if __name__ == "__main__":
    notebook = nb(CELLS)
    OUT.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {OUT}  ({len(CELLS)} cells)")
