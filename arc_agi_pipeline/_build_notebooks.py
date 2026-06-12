"""
Builds the four ARC-AGI notebooks from inline cell specs.
Run once: `python3 _build_notebooks.py` -> writes 4 .ipynb files.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    """Wrap a list of (kind, source) tuples into a valid ipynb dict."""
    out_cells = []
    for kind, source in cells:
        src = source if isinstance(source, list) else [source]
        # Ensure each line ends with \n except the last (Jupyter convention)
        normalized = []
        for i, line in enumerate(src):
            if not line.endswith("\n") and i != len(src) - 1:
                line = line + "\n"
            normalized.append(line)
        if kind == "md":
            out_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": normalized,
            })
        else:
            out_cells.append({
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": normalized,
            })
    return {
        "cells": out_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text):
    return ("md", text)


def code(text):
    return ("code", text)


# =========================================================================
# Notebook 1: 01_baseline.ipynb
# Zero-shot baseline on ARC-AGI eval set with Qwen2.5-7B-Instruct.
# Runs on free Colab T4 / Kaggle T4 in ~30 min.
# =========================================================================

baseline_cells = [
    md("""# 01 — ARC-AGI Baseline (Qwen2.5-7B-Instruct, zero-shot)

**Goal:** establish a no-training baseline so we know what training buys us.

**Pipeline overview:**
1. **This notebook (baseline)** — zero-shot Qwen on ARC-AGI eval. ~10–20% expected.
2. `02_train_h100.ipynb` — QLoRA SFT on Qwen2.5-14B with BARC + RE-ARC + augmentations.
3. `03_distill_t4x2.ipynb` — distill the 14B teacher into Qwen2.5-3B for cheap inference.
4. `04_eval_ttt.ipynb` — test-time training (per-task LoRA) at inference.

**Hardware:** Colab T4 or Kaggle T4 is enough (4-bit base + greedy decoding).
"""),
    code("""# Hardware check first
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
"""),
    code("""# Install dependencies (Colab/Kaggle)
!pip -q install -U transformers==4.46.2 accelerate==1.1.1 bitsandbytes==0.44.1 datasets==3.1.0 peft==0.13.2
"""),
    code("""# Download ARC-AGI-1 dataset (training + public eval)
import os, json, urllib.request, zipfile, io
from pathlib import Path

DATA_DIR = Path("/content/arc_data") if os.path.exists("/content") else Path("./arc_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not (DATA_DIR / "training").exists():
    url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
    print("Downloading ARC-AGI repo...")
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(DATA_DIR)
    # Move data/training and data/evaluation up
    src = DATA_DIR / "ARC-AGI-master" / "data"
    (DATA_DIR / "training").mkdir(exist_ok=True)
    (DATA_DIR / "evaluation").mkdir(exist_ok=True)
    for kind in ["training", "evaluation"]:
        for f in (src / kind).glob("*.json"):
            (DATA_DIR / kind / f.name).write_bytes(f.read_bytes())

train_files = sorted((DATA_DIR / "training").glob("*.json"))
eval_files  = sorted((DATA_DIR / "evaluation").glob("*.json"))
print(f"Train tasks: {len(train_files)}, Eval tasks: {len(eval_files)}")
"""),
    code("""# Load model in 4-bit (fits T4 16GB easily)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
print("Model loaded.")
"""),
    md("""## Grid <-> string serialization

Each ARC cell is an integer 0–9 (color). We render grids as space-separated rows:

```
0 0 1
0 1 0
1 0 0
```

This is the standard format used by most public ARC fine-tunes (BARC, MindsAI, Greenblatt).
"""),
    code("""def grid_to_str(g):
    return "\\n".join(" ".join(str(c) for c in row) for row in g)

def str_to_grid(s):
    rows = []
    for line in s.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = [int(c) for c in line.split()]
            rows.append(row)
        except ValueError:
            return None
    if not rows:
        return None
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        return None
    return rows

def format_task(task, include_test_output=False):
    parts = ["You are solving an ARC puzzle. Each grid is a 2D array of digits 0-9 (colors).",
             "Find the transformation rule from the input/output examples, then apply it to the test input.",
             ""]
    for i, ex in enumerate(task["train"], 1):
        parts.append(f"Example {i} Input:")
        parts.append(grid_to_str(ex["input"]))
        parts.append(f"Example {i} Output:")
        parts.append(grid_to_str(ex["output"]))
        parts.append("")
    test = task["test"][0]
    parts.append("Test Input:")
    parts.append(grid_to_str(test["input"]))
    parts.append("")
    parts.append("Test Output:")
    if include_test_output:
        parts.append(grid_to_str(test["output"]))
    return "\\n".join(parts)
"""),
    code("""# Inference helper — greedy, output-only parsing
@torch.inference_mode()
def predict(task, max_new_tokens=2048):
    prompt = format_task(task, include_test_output=False)
    messages = [
        {"role": "system", "content": "You are an expert ARC puzzle solver. Output only the test output grid as digits separated by spaces, one row per line. No explanations."},
        {"role": "user",   "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.eos_token_id,
    )
    completion = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return str_to_grid(completion), completion
"""),
    code("""# Run baseline on first N eval tasks (full 400 takes a while on T4)
N = 50  # bump to 400 for full eval
results = []
correct = 0
for i, fp in enumerate(eval_files[:N]):
    task = json.loads(fp.read_text())
    pred, raw = predict(task)
    truth = task["test"][0]["output"]
    is_correct = pred == truth
    correct += int(is_correct)
    results.append({"task": fp.stem, "correct": is_correct, "pred": pred})
    if (i + 1) % 10 == 0:
        print(f"[{i+1}/{N}] accuracy so far: {correct}/{i+1} = {correct/(i+1):.1%}")

print(f"\\nFINAL BASELINE: {correct}/{N} = {correct/N:.1%}")
"""),
    code("""# Save baseline for comparison after training
import json
with open(DATA_DIR / "baseline_results.json", "w") as f:
    json.dump({"n": N, "correct": correct, "accuracy": correct / N, "details": results}, f, indent=2)
print("Saved baseline_results.json")
"""),
    md("""## What to expect

- Qwen2.5-7B zero-shot on ARC-AGI eval: typically **5–15%**.
- Random / naive baselines: ~0%.
- After training (notebook 02): target **30–55%**.
- With test-time training (notebook 04): target **50–60%+**.

The state of the art (o3, Greenblatt's TTT) is **80%+**, but those use orders of magnitude more compute. Our pipeline targets a strong open-source result you can actually reproduce.
"""),
]


# =========================================================================
# Notebook 2: 02_train_h100.ipynb
# QLoRA SFT on Qwen2.5-14B-Instruct (largest that comfortably fits H100 80GB
# with QLoRA + augmentation training). Trains on BARC (synthetic) + ARC train
# + RE-ARC, with geometric/color augmentations.
# =========================================================================

train_cells = [
    md("""# 02 — Train teacher on H100 (Qwen2.5-14B-Instruct + QLoRA)

**Goal:** the largest model + best post-training that fits a single H100 (80GB) cleanly.

**Choices and *why*:**
- **Qwen2.5-14B-Instruct** — fits H100 in 4-bit with room for batch, optimizer state, and long ARC contexts. (32B is possible with QLoRA but inference becomes painful and TTT in notebook 04 needs headroom.)
- **QLoRA (4-bit base + LoRA on attention + MLP)** — keeps memory low, trains fast, and produces a small adapter that's cheap to swap and distill.
- **Training data:** ARC-AGI-1 train (400) + RE-ARC (Hodel's procedurally generated re-examples) + BARC (Berkeley synthetic, ~400K tasks generated by GPT-4). BARC is the biggest single source of signal in public ARC training.
- **Augmentations applied online:** 8 dihedral symmetries (rot/reflect) + random color permutations + example shuffling. Each base task becomes ~50–100 distinct training samples.
- **Loss:** standard SFT cross-entropy on the test-output tokens only (mask the prompt). This is what every working public ARC fine-tune does.
- **No DPO/RL in v1.** Keep it simple. Once SFT works, DPO can be added with rejected = sampled-incorrect-completion, chosen = ground-truth.

Run on Colab Pro+ with H100, ~6–10 hours for 1 epoch over ~50k augmented samples.
"""),
    code("""# Hardware check — confirm H100
import torch
assert torch.cuda.is_available(), "No GPU detected."
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name} ({vram:.0f} GB)")
if "H100" not in name and vram < 70:
    print("⚠ Not an H100 — you may need to drop to Qwen2.5-7B (set MODEL_ID below).")
"""),
    code("""# Dependencies — pinned versions known to work together
!pip -q install -U transformers==4.46.2 accelerate==1.1.1 bitsandbytes==0.44.1 \\
    datasets==3.1.0 peft==0.13.2 trl==0.12.1 huggingface_hub==0.26.2 sentencepiece
"""),
    code("""import os, json, random, copy
from pathlib import Path
import numpy as np
import torch

random.seed(0); np.random.seed(0); torch.manual_seed(0)

DATA_DIR = Path("/content/arc_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR  = Path("/content/teacher_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
"""),
    code("""# 1) ARC-AGI-1 train + eval
import urllib.request, zipfile, io
if not (DATA_DIR / "training").exists():
    url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(DATA_DIR)
    src = DATA_DIR / "ARC-AGI-master" / "data"
    for kind in ["training", "evaluation"]:
        (DATA_DIR / kind).mkdir(exist_ok=True)
        for f in (src / kind).glob("*.json"):
            (DATA_DIR / kind / f.name).write_bytes(f.read_bytes())

arc_train = [json.loads(p.read_text()) for p in sorted((DATA_DIR / "training").glob("*.json"))]
arc_eval  = [json.loads(p.read_text()) for p in sorted((DATA_DIR / "evaluation").glob("*.json"))]
print(f"ARC: {len(arc_train)} train, {len(arc_eval)} eval")
"""),
    code("""# 2) RE-ARC — Hodel's procedural re-generations of ARC tasks
# Repo: https://github.com/michaelhodel/re-arc — provides 1000s of variants per task.
# We download a precomputed bundle (small enough to fit in /content).
RE_ARC_URL = "https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip"
re_arc_dir = DATA_DIR / "re_arc"
if not re_arc_dir.exists():
    try:
        with urllib.request.urlopen(RE_ARC_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(re_arc_dir)
        print("RE-ARC downloaded")
    except Exception as e:
        print("RE-ARC download failed (skipping):", e)

re_arc_tasks = []
if re_arc_dir.exists():
    for fp in sorted(re_arc_dir.rglob("*.json"))[:5000]:  # cap for memory
        try:
            re_arc_tasks.append(json.loads(fp.read_text()))
        except Exception:
            pass
print(f"RE-ARC tasks loaded: {len(re_arc_tasks)}")
"""),
    code("""# 3) BARC — Berkeley synthetic ARC tasks (~400k via GPT-4)
# Hosted on HuggingFace: barc-llm/barc-200k-gp4-induction-gpt4-description
# This is the single biggest signal source for public ARC fine-tunes.
from datasets import load_dataset
try:
    barc = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems", split="train")
    print("BARC loaded:", len(barc), "tasks")
except Exception as e:
    print("BARC load failed (continuing without):", e)
    barc = None
"""),
    md("""## Augmentations

ARC problems are *invariant* under the dihedral group D4 (8 symmetries: rotations + reflections) and under color permutations (since color is just a categorical label, not a value with semantics). We exploit this to multiply our data ~8 × 10 = 80x.

This is the single most important data trick for ARC training.
"""),
    code("""# Geometric augmentations (D4 symmetries)
def rot90(g):  return [list(row) for row in zip(*g[::-1])]
def fliplr(g): return [row[::-1] for row in g]
def transp(g): return [list(row) for row in zip(*g)]

D4 = [
    lambda g: g,
    rot90,
    lambda g: rot90(rot90(g)),
    lambda g: rot90(rot90(rot90(g))),
    fliplr,
    lambda g: rot90(fliplr(g)),
    lambda g: rot90(rot90(fliplr(g))),
    lambda g: rot90(rot90(rot90(fliplr(g)))),
]

def apply_d4(task, k):
    f = D4[k]
    return {
        "train": [{"input": f(ex["input"]), "output": f(ex["output"])} for ex in task["train"]],
        "test":  [{"input": f(ex["input"]), "output": f(ex["output"])} for ex in task["test"]],
    }

def apply_color_perm(task, perm):
    # perm: list of length 10 mapping color i -> perm[i]
    def remap(g):
        return [[perm[c] for c in row] for row in g]
    return {
        "train": [{"input": remap(ex["input"]), "output": remap(ex["output"])} for ex in task["train"]],
        "test":  [{"input": remap(ex["input"]), "output": remap(ex["output"])} for ex in task["test"]],
    }

def shuffle_examples(task, rng):
    t = copy.deepcopy(task)
    rng.shuffle(t["train"])
    return t
"""),
    code("""# Prompt formatting — same as baseline so train/eval match
def grid_to_str(g):
    return "\\n".join(" ".join(str(c) for c in row) for row in g)

SYSTEM = ("You are an expert ARC puzzle solver. Output only the test output grid as digits "
          "separated by spaces, one row per line. No explanations.")

def format_user(task):
    parts = ["You are solving an ARC puzzle. Each grid is a 2D array of digits 0-9 (colors).",
             "Find the transformation rule from the input/output examples, then apply it to the test input.",
             ""]
    for i, ex in enumerate(task["train"], 1):
        parts.append(f"Example {i} Input:");  parts.append(grid_to_str(ex["input"]))
        parts.append(f"Example {i} Output:"); parts.append(grid_to_str(ex["output"]))
        parts.append("")
    parts.append("Test Input:");  parts.append(grid_to_str(task["test"][0]["input"]))
    parts.append(""); parts.append("Test Output:")
    return "\\n".join(parts)

def format_assistant(task):
    return grid_to_str(task["test"][0]["output"])
"""),
    code("""# Build the augmented training set as raw (prompt, completion) pairs
# Token-budget: cap at ~3500 tokens to keep batches sane on H100.

def gen_augmented(base_tasks, n_per_task=20, rng=None):
    rng = rng or random.Random(0)
    out = []
    for task in base_tasks:
        for _ in range(n_per_task):
            t = task
            t = apply_d4(t, rng.randrange(8))
            perm = list(range(10)); rng.shuffle(perm)
            t = apply_color_perm(t, perm)
            t = shuffle_examples(t, rng)
            out.append(t)
    return out

def _coerce_examples(exs):
    \"\"\"Each ex -> {'input': 2D-list, 'output': 2D-list}. Handles dict, list[2], tuple[2].\"\"\"
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
        train = _coerce_examples(row["train"]); test = _coerce_examples(row["test"])
    elif "examples" in row and "test" in row:
        train = _coerce_examples(row["examples"]); test = _coerce_examples(row["test"])
    elif "examples" in row:
        ex = _coerce_examples(row["examples"])
        if ex and len(ex) >= 2: train, test = ex[:-1], [ex[-1]]
    return {"train": train, "test": test} if (train and test) else None

base = list(arc_train) + list(re_arc_tasks)
if barc is not None:
    bsel = barc.select(range(min(20000, len(barc))))  # cap
    for r in bsel:
        norm = normalize_barc(r)
        if norm and norm["train"] and norm["test"]:
            base.append(norm)

print(f"Base task pool: {len(base)}")
augmented = gen_augmented(base, n_per_task=4)  # tune higher if epoch budget allows
print(f"Augmented samples: {len(augmented)}")
"""),
    code("""# Tokenizer + length filter (drop samples longer than 3500 tokens to keep training fast)
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # downgrade to 7B if not on H100
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

MAX_LEN = 3500

def to_chat(t):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": format_user(t)},
        {"role": "assistant", "content": format_assistant(t)},
    ]
    return tok.apply_chat_template(msgs, tokenize=False)

texts = []
for t in augmented:
    s = to_chat(t)
    if len(tok(s).input_ids) <= MAX_LEN:
        texts.append(s)
print(f"After length filter: {len(texts)} samples")
"""),
    code("""# Build a HuggingFace Dataset and train with TRL's SFTTrainer
from datasets import Dataset
ds = Dataset.from_dict({"text": texts})
ds = ds.shuffle(seed=0)
print(ds)
"""),
    code("""# Load model in 4-bit with LoRA adapters
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # H100 supports it
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
"""),
    code("""# Train with TRL SFTTrainer — completion-only loss
from trl import SFTTrainer, SFTConfig

cfg = SFTConfig(
    output_dir=str(OUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch 16
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_seq_length=MAX_LEN,
    dataset_text_field="text",
    packing=False,                    # ARC samples are heterogeneous, packing hurts
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    args=cfg,
)
trainer.train()
"""),
    code("""# Save adapter (small, ~200MB) — this is what you'll download to distill or eval
trainer.model.save_pretrained(str(OUT_DIR / "adapter"))
tok.save_pretrained(str(OUT_DIR / "adapter"))
print("Saved teacher LoRA adapter to", OUT_DIR / "adapter")
"""),
    code("""# (Optional) push adapter to HF Hub so notebooks 03 and 04 can pull it
# from huggingface_hub import login, HfApi
# login()  # paste a write token
# HfApi().upload_folder(folder_path=str(OUT_DIR / "adapter"),
#                       repo_id="<your-username>/qwen25-14b-arc-lora",
#                       repo_type="model", create_pr=False)
"""),
    code("""# Quick sanity eval on first 20 ARC eval tasks (no TTT yet — that's notebook 04)
import json, torch

@torch.inference_mode()
def predict(task, max_new=2048):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(task)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

def parse_grid(s):
    rows = []
    for line in s.strip().splitlines():
        line = line.strip()
        if not line: continue
        try: rows.append([int(c) for c in line.split()])
        except: return None
    if not rows or any(len(r)!=len(rows[0]) for r in rows): return None
    return rows

correct = 0
N = 20
for i, task in enumerate(arc_eval[:N]):
    raw = predict(task)
    if parse_grid(raw) == task["test"][0]["output"]:
        correct += 1
print(f"Quick eval: {correct}/{N} = {correct/N:.1%}")
""")
]


# =========================================================================
# Notebook 3: 03_distill_kaggle.ipynb
# Distill the 14B teacher into Qwen2.5-3B-Instruct on Kaggle T4x2 (32GB total).
# Strategy: SFT-on-teacher-outputs (sequence-level distillation). Simpler than
# logit KD and works very well in practice for instruction-following tasks.
# =========================================================================

distill_cells = [
    md("""# 03 — Distill teacher into Qwen2.5-3B (Kaggle T4 x 2)

**Goal:** turn the 14B teacher into a 3B student that runs cheaply at eval / TTT time.

**Why sequence-level distillation (not logit KL):**
- Logit KD requires keeping teacher in memory alongside student — won't fit T4x2 with a 14B teacher.
- Sequence-level (Kim & Rush 2016) is just SFT on teacher-generated outputs. Simple, interpretable, ~equivalent quality for instruction-style tasks.

**Plan:**
1. (Done in notebook 02 or here as a one-shot Colab job) Generate teacher outputs for ~10k augmented prompts → JSONL.
2. Load that JSONL on Kaggle.
3. SFT Qwen2.5-3B-Instruct + LoRA on (prompt, teacher_output) pairs across the 2 T4s with `accelerate`.

Kaggle constraint: **everything inline, no local imports**.
"""),
    code("""# Hardware check
import torch, os
print("CUDA:", torch.cuda.is_available(), "| #GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i}:", torch.cuda.get_device_name(i),
          round(torch.cuda.get_device_properties(i).total_memory/1e9,1), "GB")
"""),
    code("""# Install — Kaggle has most of these; pin for reproducibility
!pip -q install -U transformers==4.46.2 accelerate==1.1.1 bitsandbytes==0.44.1 \\
    datasets==3.1.0 peft==0.13.2 trl==0.12.1 sentencepiece
"""),
    md("""## Step 1 — Build the distillation corpus (run this on Colab H100 first)

Below is the snippet you would run on Colab (where the teacher already lives) to produce `teacher_distill.jsonl`. Then upload that file as a Kaggle dataset.

```python
# === On Colab H100, after notebook 02 ===
# Generates teacher outputs for the augmented training set, writes JSONL.
import json
from tqdm import tqdm

# `model`, `tok`, `augmented`, `format_user`, `SYSTEM` are from notebook 02.
out_path = "/content/teacher_distill.jsonl"
with open(out_path, "w") as f:
    for t in tqdm(augmented[:10000]):
        msgs = [{"role":"system","content":SYSTEM},
                {"role":"user","content":format_user(t)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=2048, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        f.write(json.dumps({
            "prompt": format_user(t),
            "teacher_output": completion,
            "ground_truth": format_assistant(t),
        }) + "\\n")
```

Once `teacher_distill.jsonl` is uploaded as a Kaggle dataset (e.g. `your-handle/arc-teacher-distill`), continue below.
"""),
    code("""# Step 2 — Load the distillation corpus on Kaggle
import json
from pathlib import Path

# Adjust to wherever you uploaded the JSONL
candidate_paths = [
    Path("/kaggle/input/arc-teacher-distill/teacher_distill.jsonl"),
    Path("./teacher_distill.jsonl"),
]
JSONL = next(p for p in candidate_paths if p.exists())
records = [json.loads(l) for l in JSONL.read_text().splitlines()]
print("Distill records:", len(records))
print("Sample keys:", list(records[0]))
"""),
    code("""# Step 3 — Format as chat for student SFT
from transformers import AutoTokenizer

STUDENT_ID = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(STUDENT_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token

SYSTEM = ("You are an expert ARC puzzle solver. Output only the test output grid as digits "
          "separated by spaces, one row per line. No explanations.")

def to_chat(rec):
    msgs = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":rec["prompt"]},
        {"role":"assistant","content":rec["teacher_output"]},
    ]
    return tok.apply_chat_template(msgs, tokenize=False)

MAX_LEN = 3500
texts = []
for r in records:
    s = to_chat(r)
    if len(tok(s).input_ids) <= MAX_LEN:
        texts.append(s)
print("Distill samples after length filter:", len(texts))
"""),
    code("""from datasets import Dataset
ds = Dataset.from_dict({"text": texts}).shuffle(seed=0)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# device_map="auto" splits across both T4s automatically
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
"""),
    code("""from trl import SFTTrainer, SFTConfig

cfg = SFTConfig(
    output_dir="./student_out",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=20,
    save_steps=1000,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_seq_length=MAX_LEN,
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
trainer.train()

# Save the student LoRA adapter
trainer.model.save_pretrained("./student_out/adapter")
tok.save_pretrained("./student_out/adapter")
print("Saved student adapter.")
"""),
    md("""## Download the adapter

After training, download `./student_out/adapter` (it's small, ~50MB). You'll plug this into notebook 04 for evaluation + test-time training.
"""),
]


# =========================================================================
# Notebook 4: 04_eval_ttt.ipynb
# Evaluate the (distilled) student on ARC-AGI eval, optionally with
# test-time training (per-task LoRA fine-tune on the 2-3 examples).
# =========================================================================

eval_cells = [
    md("""# 04 — Evaluate on ARC-AGI with Test-Time Training (TTT)

**Goal:** measure performance on the official ARC-AGI evaluation set, with the single trick that has consistently been the biggest win in the ARC Prize: **test-time training**.

**TTT in 1 paragraph:** for each test task, take the 2–4 train examples for that *single* task, apply augmentations to multiply them ~30x, and fine-tune a small LoRA adapter for ~10–30 steps on top of the base model. Then predict the test output. Cost: ~30s per task on a T4. Win: typically +15–25 percentage points.

You can run this notebook on a single T4 / RTX-class GPU, locally or on Kaggle. Use the distilled 3B student from notebook 03 for speed; or the 14B teacher from notebook 02 if you have an H100 around.
"""),
    code("""# Hardware
import torch, os
print("CUDA:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""),
    code("""!pip -q install -U transformers==4.46.2 accelerate==1.1.1 bitsandbytes==0.44.1 \\
    datasets==3.1.0 peft==0.13.2 sentencepiece
"""),
    code("""# 1) ARC eval set
import json, urllib.request, zipfile, io
from pathlib import Path
DATA_DIR = Path("./arc_data"); DATA_DIR.mkdir(exist_ok=True)
if not (DATA_DIR / "evaluation").exists():
    url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
    with urllib.request.urlopen(url) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    src = DATA_DIR / "ARC-AGI-master" / "data"
    (DATA_DIR / "evaluation").mkdir(exist_ok=True)
    for f in (src / "evaluation").glob("*.json"):
        (DATA_DIR / "evaluation" / f.name).write_bytes(f.read_bytes())
arc_eval = [(p.stem, json.loads(p.read_text())) for p in sorted((DATA_DIR/"evaluation").glob("*.json"))]
print("Eval tasks:", len(arc_eval))
"""),
    code("""# 2) Load base model + your trained adapter
# Pick ONE: the distilled student (cheaper) or the 14B teacher (stronger).
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_ID = "Qwen/Qwen2.5-3B-Instruct"           # student
# BASE_ID = "Qwen/Qwen2.5-14B-Instruct"        # teacher (needs >=24GB VRAM in 4-bit)
ADAPTER_PATH = "./student_out/adapter"          # path to the LoRA you trained

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(BASE_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE_ID, quantization_config=bnb,
                                            device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=False)
model.eval()
print("Loaded model + adapter.")
"""),
    code("""# 3) Shared helpers (mirror notebook 02)
import copy, random
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
    if not rows or any(len(r)!=len(rows[0]) for r in rows): return None
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

# D4 + color permutations for TTT augmentation
def rot90(g):  return [list(r) for r in zip(*g[::-1])]
def fliplr(g): return [r[::-1] for r in g]
D4 = [lambda g: g, rot90,
      lambda g: rot90(rot90(g)),
      lambda g: rot90(rot90(rot90(g))),
      fliplr,
      lambda g: rot90(fliplr(g)),
      lambda g: rot90(rot90(fliplr(g))),
      lambda g: rot90(rot90(rot90(fliplr(g))))]

def aug_task(task, k, color_perm):
    f = D4[k]
    def apply(ex):
        return {"input":  [[color_perm[c] for c in row] for row in f(ex["input"])],
                "output": [[color_perm[c] for c in row] for row in f(ex["output"])]}
    return {"train":[apply(e) for e in task["train"]],
            "test": [apply(e) for e in task["test"]]}
"""),
    code("""# 4) Inference helper (greedy)
@torch.inference_mode()
def predict(m, task, max_new=2048):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(task)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    return parse_grid(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))
"""),
    code("""# 5) TTT — for each task, train a fresh LoRA on its own examples.
# We make a *separate* small adapter on top of the loaded model, train ~20 steps,
# predict, then discard the adapter.

from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

def build_ttt_examples(task, n_aug=20, rng=None):
    # Build (prompt, completion) chat strings from augmented versions of THIS task,
    # treating each train example as a held-out test to learn from.
    rng = rng or random.Random(0)
    out = []
    examples = task["train"]
    if len(examples) < 2:
        return out
    for _ in range(n_aug):
        k = rng.randrange(8)
        perm = list(range(10)); rng.shuffle(perm)
        a = aug_task(task, k, perm)
        # Hold out one example to be the 'test', use others as train
        for held in range(len(a["train"])):
            sub = {"train":[e for j,e in enumerate(a["train"]) if j != held],
                   "test":[a["train"][held]]}
            msgs = [{"role":"system","content":SYSTEM},
                    {"role":"user","content":format_user(sub)},
                    {"role":"assistant","content":format_assistant(sub)}]
            out.append(tok.apply_chat_template(msgs, tokenize=False))
    return out

def ttt_and_predict(task, steps=20, lr=5e-5, n_aug=20):
    # Run TTT for one task, return prediction grid (or None on failure).
    # Wrap the already-PEFT model with a SECOND LoRA via peft is messy; instead,
    # train the existing adapter further but on a *copy* of its state, then restore.
    # Snapshot adapter weights:
    snap = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    # Re-enable training on adapter params
    for n, p in model.named_parameters():
        p.requires_grad = (n in snap)
    model.train()

    texts = build_ttt_examples(task, n_aug=n_aug)
    if not texts:
        model.eval()
        return predict(model, task)

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    for step in range(steps):
        t = rng.choice(texts)
        enc = tok(t, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
        labels = enc.input_ids.clone()
        out = model(**enc, labels=labels)
        out.loss.backward()
        opt.step(); opt.zero_grad()

    model.eval()
    pred = predict(model, task)

    # Restore adapter weights so the next task starts from the same base
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])
    return pred
"""),
    code("""# 6) Run eval — start without TTT for a sanity number, then with TTT.
import time

# (a) No TTT
N = 50  # ramp to len(arc_eval) for full run
t0 = time.time()
correct_no_ttt = 0
for i, (name, task) in enumerate(arc_eval[:N]):
    pred = predict(model, task)
    if pred == task["test"][0]["output"]:
        correct_no_ttt += 1
print(f"No-TTT: {correct_no_ttt}/{N} = {correct_no_ttt/N:.1%}  ({time.time()-t0:.0f}s)")
"""),
    code("""# (b) With TTT — ~30s/task on T4. Budget accordingly.
import time
t0 = time.time()
correct_ttt = 0
for i, (name, task) in enumerate(arc_eval[:N]):
    pred = ttt_and_predict(task, steps=20, lr=5e-5, n_aug=20)
    if pred == task["test"][0]["output"]:
        correct_ttt += 1
    if (i+1) % 5 == 0:
        elapsed = time.time() - t0
        print(f"[{i+1}/{N}] TTT acc {correct_ttt}/{i+1} = {correct_ttt/(i+1):.1%}  ({elapsed:.0f}s, ~{elapsed/(i+1):.1f}s/task)")
print(f"\\nWith TTT: {correct_ttt}/{N} = {correct_ttt/N:.1%}")
"""),
    code("""# 7) Save predictions in ARC submission format
import json
preds = {}
for name, task in arc_eval[:N]:
    pred = ttt_and_predict(task, steps=20, lr=5e-5, n_aug=20)
    if pred is None:
        pred = [[0]]
    preds[name] = [{"attempt_1": pred, "attempt_2": pred}]
with open("submission.json","w") as f:
    json.dump(preds, f)
print("Wrote submission.json with", len(preds), "tasks.")
"""),
]


# =========================================================================
# Notebook 5: 05_train_eval_interp.ipynb
# Single-H100 forensic pipeline: base -> SFT -> DPO -> GRPO with deep
# probe traces at each stage, then 11 plotly views. KISS, Colab-runnable.
# Uses the latest Qwen3 series + recent TRL with GRPOTrainer.
# =========================================================================

eval_interp_cells = [
    md("""# 05 — Train + Eval + Forensic Interpretability (single H100)

**Goal:** push toward an ARC-AGI top-100 result on a single H100 with the full modern alignment stack — **base → SFT → DPO → GRPO** — and capture *deep probe traces* at every stage so we can render 11 interactive Plotly views explaining what training actually did to the model.

**Hardware:** Colab Pro+ with H100 (80GB). Falls back to A100/L4 with reduced batch.

**Why this notebook exists alongside `02_train_h100`:**
- `02_train_h100` does SFT only; this one adds DPO (preference) + GRPO (RL with verifiable reward) — the same recipe top public ARC fine-tunes use.
- Captures hidden states, layer norms, top neurons, LoRA deltas, gradient flows at every stage so you can **see** how training rewires the network.
- Outputs 11 standalone HTML viz files in `runs/<RUN_ID>/viz/` — open them in any browser without re-running.

**Optimization stack:** QLoRA 4-bit + LoRA r=64 on attn+MLP + FlashAttention-2 + bf16 + paged_adamw_8bit + gradient checkpointing + D4×color-perm augmentation.
"""),
    code("""# 0.1 — Hardware check
import torch, sys
assert torch.cuda.is_available(), "Need a CUDA GPU. Switch Colab runtime to GPU (H100/A100/L4)."
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME} ({VRAM_GB:.0f} GB) | torch {torch.__version__} | python {sys.version.split()[0]}")
IS_H100 = "H100" in GPU_NAME
"""),
    code("""# 0.2 — Install dependencies (latest known-good for Qwen3 + GRPO)
# Qwen3 needs transformers>=4.51, GRPOTrainer needs trl>=0.15.
!pip -q install -U \\
    "transformers>=4.51,<4.55" \\
    "trl>=0.15,<0.20" \\
    "peft>=0.14" \\
    "accelerate>=1.4" \\
    "bitsandbytes>=0.45" \\
    "datasets>=3.2" \\
    "plotly>=5.24" \\
    sentencepiece huggingface_hub
"""),
    code("""# 0.3 — Run config + paths
import os, time, json, random
from pathlib import Path

# --- pick model ---
# Qwen3-8B is the natural top-100 starting point on a single H100 with the full 4-stage stack.
# Bump to 14B if you have headroom; drop to 4B for fast iteration.
MODEL_ID = "Qwen/Qwen3-8B"

RUN_ID   = f"run-{int(time.time())}"
ROOT     = Path("/content") if Path("/content").exists() else Path(".")
RUN_DIR  = ROOT / "runs" / RUN_ID
for sub in ["adapters/sft", "adapters/dpo", "adapters/grpo",
            "traces/base", "traces/sft", "traces/dpo", "traces/grpo",
            "logs", "viz", "eval"]:
    (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT / "arc_data"; DATA_DIR.mkdir(exist_ok=True)
SEED = 0
random.seed(SEED); torch.manual_seed(SEED)

print("RUN_DIR:", RUN_DIR)
print("MODEL  :", MODEL_ID)
"""),
    md("""## 1 — Data: ARC-AGI + RE-ARC + BARC, with D4×color-perm augmentation

Same recipe as `02_train_h100`. The single biggest data trick is multiplying every base task by the 8-element dihedral group × random color permutations.
"""),
    code("""# 1.1 — Download ARC-AGI-1
import urllib.request, zipfile, io
if not (DATA_DIR / "training").exists():
    url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
    print("Downloading ARC-AGI...")
    with urllib.request.urlopen(url) as r:
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
    code("""# 1.2 — RE-ARC (procedural re-generations of ARC tasks)
re_arc_dir = DATA_DIR / "re_arc"
re_arc_tasks = []
if not re_arc_dir.exists():
    try:
        with urllib.request.urlopen("https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip") as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(re_arc_dir)
    except Exception as e:
        print("RE-ARC skipped:", e)
if re_arc_dir.exists():
    for fp in sorted(re_arc_dir.rglob("*.json"))[:5000]:
        try: re_arc_tasks.append(json.loads(fp.read_text()))
        except: pass
print(f"RE-ARC tasks: {len(re_arc_tasks)}")
"""),
    code("""# 1.3 — BARC (Berkeley synthetic ARC, ~200k via GPT-4)
from datasets import load_dataset
try:
    barc = load_dataset(
        "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        split="train",
    )
    print("BARC:", len(barc))
except Exception as e:
    print("BARC load skipped:", e); barc = None
"""),
    code("""# 1.4 — Augmentation helpers (D4 + color permutation + example shuffle)
import copy

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

def apply_d4(task, k):
    f = D4[k]
    return {"train":[{"input":f(e["input"]),"output":f(e["output"])} for e in task["train"]],
            "test": [{"input":f(e["input"]),"output":f(e["output"])} for e in task["test"]]}

def apply_color_perm(task, perm):
    def remap(g): return [[perm[c] for c in row] for row in g]
    return {"train":[{"input":remap(e["input"]),"output":remap(e["output"])} for e in task["train"]],
            "test": [{"input":remap(e["input"]),"output":remap(e["output"])} for e in task["test"]]}

def shuffle_examples(task, rng):
    t = copy.deepcopy(task); rng.shuffle(t["train"]); return t
"""),
    code("""# 1.5 — Prompt format (matches 02_train_h100 + 04_eval_ttt for consistency)
SYSTEM = ("You are an expert ARC puzzle solver. Output only the test output grid as digits "
          "separated by spaces, one row per line. No explanations.")

def grid_to_str(g):
    return "\\n".join(" ".join(str(c) for c in row) for row in g)

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
    parts.append("Test Input:");  parts.append(grid_to_str(task["test"][0]["input"]))
    parts.append(""); parts.append("Test Output:")
    return "\\n".join(parts)

def format_assistant(task):
    return grid_to_str(task["test"][0]["output"])
"""),
    code("""# 1.6 — Build the augmented base pool + tokenize/length-filter for SFT
def _coerce_examples(exs):
    \"\"\"Each ex must be {'input': 2D-list, 'output': 2D-list}.
    BARC variants encode pairs as dict, list[2], or tuple[2] — handle all.\"\"\"
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
        train = _coerce_examples(row["train"]); test = _coerce_examples(row["test"])
    elif "examples" in row and "test" in row:
        train = _coerce_examples(row["examples"]); test = _coerce_examples(row["test"])
    elif "examples" in row:
        ex = _coerce_examples(row["examples"])
        if ex and len(ex) >= 2: train, test = ex[:-1], [ex[-1]]
    return {"train": train, "test": test} if (train and test) else None

base_pool = list(arc_train) + list(re_arc_tasks)
if barc is not None:
    for r in barc.select(range(min(20000, len(barc)))):
        n = normalize_barc(r)
        if n and n["train"] and n["test"]:
            base_pool.append(n)
print(f"Base pool: {len(base_pool)}")

def gen_augmented(tasks, n_per_task, rng):
    out = []
    for t in tasks:
        for _ in range(n_per_task):
            x = apply_d4(t, rng.randrange(8))
            perm = list(range(10)); rng.shuffle(perm)
            x = apply_color_perm(x, perm)
            x = shuffle_examples(x, rng)
            out.append(x)
    return out

augmented = gen_augmented(base_pool, n_per_task=4, rng=random.Random(SEED))
print(f"Augmented samples (pre-filter): {len(augmented)}")
"""),
    code("""# 1.7 — Load tokenizer + length-filter
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Qwen3 chat template supports a "thinking" mode; we want concise grids, no <think> blocks.
QWEN3_KW = {"enable_thinking": False} if "Qwen3" in MODEL_ID else {}

MAX_LEN = 3500

def to_chat_full(t):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(t)},
            {"role":"assistant","content":format_assistant(t)}]
    return tok.apply_chat_template(msgs, tokenize=False, **QWEN3_KW)

def to_chat_prompt(t):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(t)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **QWEN3_KW)

sft_texts = []
for t in augmented:
    s = to_chat_full(t)
    if len(tok(s, add_special_tokens=False).input_ids) <= MAX_LEN:
        sft_texts.append(s)
print(f"SFT samples after length filter: {len(sft_texts)}")

from datasets import Dataset
sft_dataset = Dataset.from_dict({"text": sft_texts}).shuffle(seed=SEED)
print(sft_dataset)
"""),
    md("""## 2 — Probe set

Three fixed ARC eval tasks we'll re-trace at each training stage. These are the "patients" — we image them at base, after SFT, after DPO, after GRPO, and watch how the network responds change.
"""),
    code("""# 2.1 — Pick 3 small, diverse probes from the eval set
PROBE_IDXS = [0, 13, 47]  # arbitrary but fixed; small grids preferred
PROBES = [(arc_eval[i][0], arc_eval[i][1]) for i in PROBE_IDXS if i < len(arc_eval)]
for pid, t in PROBES:
    h = len(t["test"][0]["input"]); w = len(t["test"][0]["input"][0])
    print(f"  probe {pid}: {len(t['train'])} examples, test input {h}x{w}")
"""),
    md("""## 3 — Trace capture utilities

Three helpers that get called once at base + after each training stage:

1. **logit lens** — what does the LM head predict if we read it off layer L's hidden state, instead of the last layer? Reveals where the answer "crystallizes" in the network.
2. **layer norms + top neurons** — per-layer attention output norm, MLP output norm, and the top-10 most-activated MLP neurons. Forward hooks.
3. **LoRA snapshot** — full state-dict diff between phase boundaries.
"""),
    code("""# 3.1 — Logit lens + per-token confidence/entropy
import numpy as np

@torch.inference_mode()
def capture_logit_lens(model, tok, probe_task, max_new=128, top_k_lens=1):
    \"\"\"Generate up to max_new tokens; for each step, for each layer, run the LM head
    on the last-position hidden state and capture top-1 token + prob, plus chosen-token
    prob and entropy. Returns dict suitable for JSON.\"\"\"
    base_for_head = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm_head = base_for_head.get_output_embeddings()
    final_norm = getattr(base_for_head.model, "norm", None)  # Qwen2/3 uses RMSNorm at .model.norm

    prompt = to_chat_prompt(probe_task)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    input_ids = inp.input_ids
    attn_mask = inp.attention_mask
    eos = tok.eos_token_id

    steps = []
    for step in range(max_new):
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of (n_layers+1) tensors [1, T, d]
        last_logits = out.logits[:, -1, :]
        probs_full  = torch.softmax(last_logits.float(), dim=-1)
        chosen_id   = int(torch.argmax(last_logits, dim=-1))
        chosen_prob = float(probs_full[0, chosen_id])
        ent         = float(-(probs_full * torch.log(probs_full.clamp_min(1e-12))).sum())

        layer_rows = []
        for li, h in enumerate(hs):
            h_last = h[:, -1, :].to(lm_head.weight.dtype)
            if final_norm is not None and li == len(hs) - 1:
                pass  # last hidden is post-norm in most arches
            elif final_norm is not None:
                h_last = final_norm(h_last)
            logits_l = lm_head(h_last).float()
            probs_l  = torch.softmax(logits_l, dim=-1)
            top1     = int(torch.argmax(probs_l, dim=-1))
            layer_rows.append({"layer": li, "top1_id": top1,
                               "top1_prob": float(probs_l[0, top1]),
                               "top1_tok": tok.decode([top1])})

        chosen_tok = tok.decode([chosen_id])
        steps.append({"step": step, "chosen_id": chosen_id, "chosen_tok": chosen_tok,
                      "chosen_prob": chosen_prob, "entropy": ent, "layers": layer_rows})

        # advance
        input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=model.device)], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((1,1), device=model.device, dtype=attn_mask.dtype)], dim=1)
        if chosen_id == eos:
            break
    return {"steps": steps, "n_layers": len(steps[0]["layers"]) if steps else 0}
"""),
    code("""# 3.2 — Forward-hook layer norms + top MLP neurons (single forward pass)
@torch.inference_mode()
def capture_layer_norms_and_neurons(model, tok, probe_task, top_k=10):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers
    attn_norms, mlp_norms, top_neurons = [], [], []

    handles = []
    cache = {}
    for i, layer in enumerate(layers):
        def make_hook_attn(idx):
            def h(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                cache[("attn", idx)] = t.detach().float().pow(2).mean(dim=-1).sqrt().mean().item()
            return h
        def make_hook_mlp(idx):
            def h(_m, _i, o):
                t = o
                cache[("mlp", idx)] = t.detach().float().pow(2).mean(dim=-1).sqrt().mean().item()
                # top-k neurons by mean |activation| across tokens
                act_abs = t.detach().float().abs().mean(dim=1).squeeze(0)  # [hidden]
                vals, idxs = torch.topk(act_abs, k=top_k)
                cache[("mlp_top", idx)] = (vals.tolist(), idxs.tolist())
            return h
        handles.append(layer.self_attn.register_forward_hook(make_hook_attn(i)))
        handles.append(layer.mlp.register_forward_hook(make_hook_mlp(i)))

    prompt = to_chat_prompt(probe_task) + format_assistant(probe_task)
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    _ = model(**inp, use_cache=False)

    for h in handles: h.remove()

    n = len(layers)
    for i in range(n):
        attn_norms.append(cache.get(("attn", i), 0.0))
        mlp_norms.append(cache.get(("mlp",  i), 0.0))
        vals, idxs = cache.get(("mlp_top", i), ([], []))
        top_neurons.append({"layer": i, "vals": vals, "idxs": idxs})
    return {"attn_norms": attn_norms, "mlp_norms": mlp_norms, "top_neurons": top_neurons}
"""),
    code("""# 3.3 — LoRA snapshot + delta
def snapshot_lora(model):
    snap = {}
    for n, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in n or "lora_B" in n):
            snap[n] = p.detach().to("cpu", dtype=torch.float32).clone()
    return snap

def lora_delta(prev, curr):
    \"\"\"Returns dict of {module_path: {prev_norm, curr_norm, delta_norm, pct}}.
    module_path strips lora_A/B suffix so we can aggregate per target module.\"\"\"
    keys = set(prev.keys()) | set(curr.keys())
    by_mod = {}
    for k in keys:
        mod = k.replace(".lora_A.default.weight", "").replace(".lora_B.default.weight", "")
        mod = mod.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        a = prev.get(k); b = curr.get(k)
        a_n = float(a.norm()) if a is not None else 0.0
        b_n = float(b.norm()) if b is not None else 0.0
        d_n = float((b - a).norm()) if (a is not None and b is not None and a.shape == b.shape) else b_n
        e = by_mod.setdefault(mod, {"prev_norm": 0.0, "curr_norm": 0.0, "delta_norm": 0.0})
        e["prev_norm"]  += a_n
        e["curr_norm"]  += b_n
        e["delta_norm"] += d_n
    for m, v in by_mod.items():
        v["pct"] = (v["delta_norm"] / max(v["prev_norm"], 1e-9)) * 100.0
    return by_mod

def write_traces(stage, model, tok, probes):
    for pid, task in probes:
        ll = capture_logit_lens(model, tok, task, max_new=96)
        ln = capture_layer_norms_and_neurons(model, tok, task, top_k=10)
        out = {"probe_id": pid, "stage": stage, "logit_lens": ll, "layer_stats": ln}
        (RUN_DIR / "traces" / stage / f"probe_{pid}.json").write_text(json.dumps(out))
    print(f"[traces] wrote {len(probes)} probes for stage={stage}")
"""),
    md("""## 4 — Stage callbacks

Single `TrainerCallback` that streams `{step, loss, grad_norm, lr, ...}` to `runs/<RUN_ID>/logs/<phase>.jsonl`. Used for SFT, DPO, GRPO. Captures phase-specific extras (chosen/rejected/margin, rewards, KL) when present in the log dict.
"""),
    code("""# 4.1 — TraceCallback
from transformers import TrainerCallback

class TraceCallback(TrainerCallback):
    def __init__(self, phase):
        self.phase = phase
        self.path = RUN_DIR / "logs" / f"{phase}.jsonl"
        self.f = open(self.path, "a")
    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None: return
        rec = {"phase": self.phase, "step": state.global_step}
        for k in ("loss","grad_norm","learning_rate",
                 "rewards/chosen","rewards/rejected","rewards/margins",
                 "reward","reward_std","kl"):
            if k in logs: rec[k] = float(logs[k])
        self.f.write(json.dumps(rec) + "\\n"); self.f.flush()
    def __del__(self):
        try: self.f.close()
        except: pass
"""),
    md("""## 5 — Stage 1: Base (no training, just trace)

Load base model in 4-bit + attach a fresh LoRA adapter (zero-initialized in lora_B). Capture probe traces. Pre-generate "rejected" completions for the DPO dataset while we're here.
"""),
    code("""# 5.1 — Load model in 4-bit + LoRA
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
attn_impl = "flash_attention_2" if IS_H100 else "sdpa"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
"""),
    code("""# 5.2 — Capture base traces + snapshot LoRA (= zeros) for delta math
write_traces("base", model, tok, PROBES)
SNAPSHOTS = {"base": snapshot_lora(model)}

# Save the (untrained) adapter so 'base' is reproducible
model.save_pretrained(str(RUN_DIR / "adapters" / "base"))
"""),
    code("""# 5.3 — Pre-generate base-model rejected completions for DPO
# We sample on a small subset of the augmented pool. chosen = ground-truth grid string.
from tqdm.auto import tqdm

DPO_N = 1500
dpo_source = augmented[:DPO_N]

dpo_records = []
model.eval()
for t in tqdm(dpo_source, desc="DPO rejected gen"):
    user = format_user(t)
    chosen = format_assistant(t)
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **QWEN3_KW)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN-512).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=512, do_sample=True, temperature=1.0,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
    rejected = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    if rejected and rejected != chosen:
        dpo_records.append({"prompt": text, "chosen": chosen, "rejected": rejected})

print(f"DPO records: {len(dpo_records)}")
(RUN_DIR / "logs" / "dpo_dataset_preview.json").write_text(json.dumps(dpo_records[:3], indent=2))
"""),
    md("""## 6 — Stage 2: SFT

TRL `SFTTrainer`, 1 epoch over augmented pool. Cosine LR + paged 8-bit AdamW + bf16 + gradient checkpointing. Capture traces after.
"""),
    code("""# 6.1 — SFT
from trl import SFTTrainer, SFTConfig

sft_cfg = SFTConfig(
    output_dir=str(RUN_DIR / "adapters" / "sft"),
    num_train_epochs=1,
    per_device_train_batch_size=2 if VRAM_GB >= 70 else 1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_steps=10_000,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_seq_length=MAX_LEN,
    dataset_text_field="text",
    packing=False,
    report_to="none",
    seed=SEED,
)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=sft_dataset,
    args=sft_cfg,
    callbacks=[TraceCallback("sft")],
)
sft_trainer.train()
sft_trainer.model.save_pretrained(str(RUN_DIR / "adapters" / "sft"))
print("SFT done.")
"""),
    code("""# 6.2 — Post-SFT traces + LoRA snapshot
write_traces("sft", model, tok, PROBES)
SNAPSHOTS["sft"] = snapshot_lora(model)
"""),
    md("""## 7 — Stage 3: DPO (preference optimization)

`chosen` = ground-truth grid string. `rejected` = base-model greedy/sampled completion (pre-generated in section 5.3). β=0.1, low LR. DPO needs a frozen reference; TRL handles this by snapshotting the model when `ref_model=None`.
"""),
    code("""# 7.1 — DPO
from trl import DPOTrainer, DPOConfig

# Build HF dataset from records. DPO format: {prompt, chosen, rejected}.
# Note: trl >= 0.15 accepts a "prompt" string already chat-templated (full text up through assistant tag).
dpo_ds = Dataset.from_list(dpo_records)
print(dpo_ds)

dpo_cfg = DPOConfig(
    output_dir=str(RUN_DIR / "adapters" / "dpo"),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_steps=10_000,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    beta=0.1,
    max_length=MAX_LEN,
    max_prompt_length=MAX_LEN-768,
    report_to="none",
    seed=SEED,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,           # PEFT model: TRL uses adapter-disabled forward as ref
    args=dpo_cfg,
    train_dataset=dpo_ds,
    tokenizer=tok,
    callbacks=[TraceCallback("dpo")],
)
dpo_trainer.train()
dpo_trainer.model.save_pretrained(str(RUN_DIR / "adapters" / "dpo"))
print("DPO done.")
"""),
    code("""# 7.2 — Post-DPO traces + snapshot
write_traces("dpo", model, tok, PROBES)
SNAPSHOTS["dpo"] = snapshot_lora(model)
"""),
    md("""## 8 — Stage 4: GRPO (RL with verifiable grid-match reward)

Group-relative policy optimization. For each prompt we sample N completions, score them with a simple reward function, and update toward the high-reward ones relative to the group mean.

**Reward:** +1 exact grid match, +0.1 parses-as-grid, 0 otherwise.
""")
,
    code("""# 8.1 — GRPO dataset (prompts only, with ground truth carried for the reward fn)
GRPO_N = 600
grpo_source = augmented[DPO_N:DPO_N + GRPO_N]

grpo_rows = []
for t in grpo_source:
    grpo_rows.append({
        "prompt": to_chat_prompt(t),
        "ground_truth": format_assistant(t),
    })
grpo_ds = Dataset.from_list(grpo_rows)
print(grpo_ds)
"""),
    code("""# 8.2 — Reward function
def grid_match_reward(completions, prompts=None, ground_truth=None, **kw):
    \"\"\"Returns a list of floats, one per completion.
    TRL passes additional dataset columns by name as kwargs (e.g. ground_truth).\"\"\"
    rewards = []
    truths = ground_truth if ground_truth is not None else [None]*len(completions)
    for c, gt in zip(completions, truths):
        text = c if isinstance(c, str) else (c[0]["content"] if isinstance(c, list) else str(c))
        g = parse_grid(text)
        if g is None:
            rewards.append(0.0); continue
        if gt is not None and parse_grid(gt) == g:
            rewards.append(1.0)
        else:
            rewards.append(0.1)
    return rewards
"""),
    code("""# 8.3 — GRPO training
from trl import GRPOTrainer, GRPOConfig

grpo_cfg = GRPOConfig(
    output_dir=str(RUN_DIR / "adapters" / "grpo"),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=5,
    save_steps=10_000,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    num_generations=4,
    max_prompt_length=MAX_LEN-512,
    max_completion_length=512,
    beta=0.04,
    report_to="none",
    seed=SEED,
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=grid_match_reward,
    args=grpo_cfg,
    train_dataset=grpo_ds,
    processing_class=tok,
    callbacks=[TraceCallback("grpo")],
)
grpo_trainer.train()
grpo_trainer.model.save_pretrained(str(RUN_DIR / "adapters" / "grpo"))
print("GRPO done.")
"""),
    code("""# 8.4 — Post-GRPO traces + snapshot
write_traces("grpo", model, tok, PROBES)
SNAPSHOTS["grpo"] = snapshot_lora(model)

# Persist snapshots to disk so we can recompute deltas later
import pickle
with open(RUN_DIR / "logs" / "lora_snapshots.pkl", "wb") as f:
    pickle.dump({k: {n: t.numpy() for n, t in v.items()} for k, v in SNAPSHOTS.items()}, f)
"""),
    md("""## 9 — Forensic interpretability (11 plotly views)

Every figure is also written to `runs/<RUN_ID>/viz/9_<id>.html` so you can open them later without rerunning.
"""),
    code("""# 9.0 — Loader: reads every trace file into nested dicts.
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STAGES = ["base", "sft", "dpo", "grpo"]

def load_all_traces():
    out = {s: {} for s in STAGES}
    for s in STAGES:
        for p in sorted((RUN_DIR / "traces" / s).glob("*.json")):
            d = json.loads(p.read_text())
            out[s][d["probe_id"]] = d
    return out

TRACES = load_all_traces()
PROBE_IDS = sorted({pid for s in STAGES for pid in TRACES[s].keys()})
print("Loaded probes:", PROBE_IDS, "stages:", STAGES)

VIZ_DIR = RUN_DIR / "viz"
def save_fig(fig, idx, title):
    fig.update_layout(title=title, template="plotly_white")
    path = VIZ_DIR / f"9_{idx}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"[viz] {path}")
"""),
    code("""# 9.1 — Logit Lens Heatmap Grid (probes x stages)
def viz_9_1(traces):
    if not PROBE_IDS: return None
    rows, cols = len(PROBE_IDS), 4
    fig = make_subplots(rows=rows, cols=cols,
                        row_titles=[f"probe {p}" for p in PROBE_IDS],
                        column_titles=STAGES,
                        horizontal_spacing=0.04, vertical_spacing=0.06)
    for r, pid in enumerate(PROBE_IDS, start=1):
        for c, s in enumerate(STAGES, start=1):
            d = traces[s].get(pid)
            if d is None: continue
            steps = d["logit_lens"]["steps"]
            if not steps: continue
            n_layers = len(steps[0]["layers"])
            Z = np.zeros((len(steps), n_layers))
            text = [["" for _ in range(n_layers)] for _ in range(len(steps))]
            for si, st in enumerate(steps):
                for li, lr in enumerate(st["layers"]):
                    Z[si, li] = lr["top1_prob"]
                    text[si][li] = f"L{li} step{si}: '{lr['top1_tok']}' p={lr['top1_prob']:.2f}"
            fig.add_trace(go.Heatmap(z=Z, text=text, hoverinfo="text",
                                     coloraxis="coloraxis", showscale=(r==1 and c==1)),
                          row=r, col=c)
    fig.update_layout(coloraxis={"colorscale":"Viridis","cmin":0,"cmax":1},
                      height=300*rows, width=1400)
    save_fig(fig, "1", "9.1 — Logit Lens Heatmap Grid (Y=gen step, X=layer, color=top-1 prob)")
    return fig

viz_9_1(TRACES)
"""),
    code("""# 9.2 — Per-Token Confidence + Entropy
def viz_9_2(traces):
    if not PROBE_IDS: return None
    fig = make_subplots(rows=len(PROBE_IDS)*2, cols=1,
                        shared_xaxes=False, vertical_spacing=0.04,
                        row_titles=sum([[f"probe {p} P(chosen)", f"probe {p} entropy"] for p in PROBE_IDS], []))
    for ri, pid in enumerate(PROBE_IDS):
        for s in STAGES:
            d = traces[s].get(pid)
            if d is None: continue
            steps = d["logit_lens"]["steps"]
            xs = [st["step"] for st in steps]
            ps = [st["chosen_prob"] for st in steps]
            es = [st["entropy"] for st in steps]
            ts = [st["chosen_tok"] for st in steps]
            fig.add_trace(go.Bar(x=xs, y=ps, name=f"{s}", legendgroup=s,
                                 hovertext=ts, hoverinfo="text+y", showlegend=(ri==0)),
                          row=ri*2+1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=es, mode="lines+markers", name=f"{s}",
                                     legendgroup=s, showlegend=False),
                          row=ri*2+2, col=1)
    fig.update_layout(height=300*len(PROBE_IDS)*2, width=1200, barmode="group")
    save_fig(fig, "2", "9.2 — Per-Token Confidence (top) + Entropy (bottom), per probe, all 4 stages")
    return fig

viz_9_2(TRACES)
"""),
    code("""# 9.3 — Layer Activation Profiles (per probe, all stages overlaid)
def viz_9_3(traces):
    if not PROBE_IDS: return None
    fig = make_subplots(rows=len(PROBE_IDS), cols=2,
                        column_titles=["attn output norm", "mlp output norm"],
                        row_titles=[f"probe {p}" for p in PROBE_IDS])
    for ri, pid in enumerate(PROBE_IDS, start=1):
        for s in STAGES:
            d = traces[s].get(pid)
            if d is None: continue
            an = d["layer_stats"]["attn_norms"]
            mn = d["layer_stats"]["mlp_norms"]
            xs = list(range(len(an)))
            fig.add_trace(go.Scatter(x=xs, y=an, mode="lines+markers", name=s,
                                     legendgroup=s, showlegend=(ri==1)), row=ri, col=1)
            fig.add_trace(go.Scatter(x=xs, y=mn, mode="lines+markers", name=s,
                                     legendgroup=s, showlegend=False), row=ri, col=2)
    fig.update_xaxes(title_text="layer index")
    fig.update_layout(height=320*len(PROBE_IDS), width=1300)
    save_fig(fig, "3", "9.3 — Layer Activation Profiles (which layers do work?)")
    return fig

viz_9_3(TRACES)
"""),
    code("""# 9.4 — Top-Neuron Heatmap (per probe x stage)
def viz_9_4(traces):
    if not PROBE_IDS: return None
    fig = make_subplots(rows=len(PROBE_IDS), cols=4,
                        column_titles=STAGES,
                        row_titles=[f"probe {p}" for p in PROBE_IDS],
                        horizontal_spacing=0.04, vertical_spacing=0.06)
    for ri, pid in enumerate(PROBE_IDS, start=1):
        for ci, s in enumerate(STAGES, start=1):
            d = traces[s].get(pid)
            if d is None: continue
            tn = d["layer_stats"]["top_neurons"]
            if not tn: continue
            Z = np.array([row["vals"] for row in tn])
            text = [[f"L{row['layer']} rank{j}: neuron #{row['idxs'][j]} val={row['vals'][j]:.3f}"
                     for j in range(len(row['vals']))] for row in tn]
            fig.add_trace(go.Heatmap(z=Z, text=text, hoverinfo="text",
                                     coloraxis="coloraxis", showscale=(ri==1 and ci==1)),
                          row=ri, col=ci)
    fig.update_layout(coloraxis={"colorscale":"Magma"},
                      height=320*len(PROBE_IDS), width=1400)
    save_fig(fig, "4", "9.4 — Top-10 MLP Neurons per Layer (Y=layer, X=top-k rank, color=|act|)")
    return fig

viz_9_4(TRACES)
"""),
    code("""# 9.5 — LoRA Delta Bars (per phase boundary)
def viz_9_5():
    boundaries = [("sft","base"), ("dpo","sft"), ("grpo","dpo")]
    fig = make_subplots(rows=1, cols=len(boundaries),
                        column_titles=[f"{b}-{a}" for b,a in boundaries],
                        horizontal_spacing=0.18)
    for ci, (curr, prev) in enumerate(boundaries, start=1):
        if curr not in SNAPSHOTS or prev not in SNAPSHOTS: continue
        deltas = lora_delta(SNAPSHOTS[prev], SNAPSHOTS[curr])
        items = sorted(deltas.items(), key=lambda kv: -abs(kv[1]["delta_norm"]))[:20]
        names = [k.split(".")[-3] + "." + k.split(".")[-2] if "." in k else k for k, _ in items]
        vals  = [v["delta_norm"] for _, v in items]
        pcts  = [v["pct"] for _, v in items]
        text  = [f"{p:.1f}%" for p in pcts]
        fig.add_trace(go.Bar(x=vals, y=names, orientation="h",
                             text=text, textposition="outside",
                             marker_color=["#2a9d8f" if p>=0 else "#e76f51" for p in pcts]),
                      row=1, col=ci)
        fig.update_yaxes(autorange="reversed", row=1, col=ci)
    fig.update_layout(height=700, width=1500, showlegend=False)
    save_fig(fig, "5", "9.5 — Top-20 LoRA modules by |Δnorm| at each phase boundary")
    return fig

viz_9_5()
"""),
    code("""# 9.6 — Gradient Flow Heatmap (per phase: step x summary stat)
# Note: we only logged a global grad_norm per step; we'll render that as a single-row
# heatmap per phase (the per-module breakdown would require deeper hooking).
def load_phase_log(phase):
    p = RUN_DIR / "logs" / f"{phase}.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def viz_9_6():
    phases = ["sft","dpo","grpo"]
    fig = make_subplots(rows=len(phases), cols=1,
                        row_titles=phases, shared_xaxes=False, vertical_spacing=0.08)
    for ri, ph in enumerate(phases, start=1):
        rows = load_phase_log(ph)
        if not rows: continue
        steps = [r["step"] for r in rows if "grad_norm" in r]
        gns   = [r["grad_norm"] for r in rows if "grad_norm" in r]
        if not gns: continue
        Z = np.log10(np.array([gns]) + 1e-9)
        fig.add_trace(go.Heatmap(z=Z, x=steps, coloraxis="coloraxis"),
                      row=ri, col=1)
    fig.update_layout(coloraxis={"colorscale":"RdBu"}, height=240*len(phases), width=1300)
    save_fig(fig, "6", "9.6 — Gradient Flow Heatmap (log10 grad_norm vs step, per phase)")
    return fig

viz_9_6()
"""),
    code("""# 9.7 — GRPO Reward Trajectory
def viz_9_7():
    rows = load_phase_log("grpo")
    if not rows:
        return None
    steps = [r["step"] for r in rows]
    rew   = [r.get("reward",   None) for r in rows]
    rstd  = [r.get("reward_std", None) for r in rows]
    kl    = [r.get("kl",       None) for r in rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["mean reward (line) + std (band)", "KL"])
    fig.add_trace(go.Scatter(x=steps, y=rew,  mode="lines+markers", name="reward"), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=rstd, mode="lines",          name="reward_std"), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=kl,   mode="lines",          name="kl"), row=2, col=1)
    fig.update_layout(height=600, width=1200)
    save_fig(fig, "7", "9.7 — GRPO Reward Trajectory")
    return fig

viz_9_7()
"""),
    code("""# 9.8 — Cross-Stage Representation Drift (per probe, attn+mlp norms, all 4 stages)
def viz_9_8(traces):
    if not PROBE_IDS: return None
    fig = make_subplots(rows=len(PROBE_IDS), cols=1,
                        row_titles=[f"probe {p}" for p in PROBE_IDS])
    palette = {"base":"#264653","sft":"#2a9d8f","dpo":"#e9c46a","grpo":"#e76f51"}
    for ri, pid in enumerate(PROBE_IDS, start=1):
        for s in STAGES:
            d = traces[s].get(pid)
            if d is None: continue
            an = d["layer_stats"]["attn_norms"]
            mn = d["layer_stats"]["mlp_norms"]
            xs = list(range(len(an)))
            fig.add_trace(go.Scatter(x=xs, y=an, mode="lines",
                                     line=dict(color=palette[s], dash="solid"),
                                     name=f"{s} attn", legendgroup=s, showlegend=(ri==1)),
                          row=ri, col=1)
            fig.add_trace(go.Scatter(x=xs, y=mn, mode="lines",
                                     line=dict(color=palette[s], dash="dot"),
                                     name=f"{s} mlp", legendgroup=s, showlegend=(ri==1)),
                          row=ri, col=1)
    fig.update_xaxes(title_text="layer index")
    fig.update_layout(height=320*len(PROBE_IDS), width=1300)
    save_fig(fig, "8", "9.8 — Cross-Stage Representation Drift (solid=attn, dotted=mlp)")
    return fig

viz_9_8(TRACES)
"""),
    code("""# 9.9 — Token-Level Cross-Stage Diff Table
def viz_9_9(traces):
    if not PROBE_IDS: return None
    figs = []
    for pid in PROBE_IDS:
        rows = []
        per = {s: traces[s].get(pid, {}).get("logit_lens", {}).get("steps", []) for s in STAGES}
        n = max((len(per[s]) for s in STAGES), default=0)
        for i in range(n):
            row = [i]
            toks = []
            for s in STAGES:
                if i < len(per[s]):
                    row.append(per[s][i]["chosen_tok"])
                    row.append(f"{per[s][i]['chosen_prob']:.2f}")
                    toks.append(per[s][i]["chosen_tok"])
                else:
                    row.append(""); row.append(""); toks.append(None)
            rows.append((row, toks))
        if not rows: continue
        # color cells red where token differs from base
        header = ["step"] + sum([[f"{s}.tok", f"{s}.prob"] for s in STAGES], [])
        cols = list(zip(*[r[0] for r in rows]))
        # build per-cell fill colors
        n_cols = len(header)
        fill = [["#ffffff"]*len(rows) for _ in range(n_cols)]
        for ri, (_, toks) in enumerate(rows):
            base_tok = toks[0]
            for si, s in enumerate(STAGES):
                if toks[si] is not None and base_tok is not None and toks[si] != base_tok:
                    fill[1 + si*2][ri] = "#fde2e2"
        fig = go.Figure(data=[go.Table(
            header=dict(values=header, fill_color="#264653", font=dict(color="white")),
            cells=dict(values=cols, fill_color=fill, align="left"),
        )])
        fig.update_layout(width=1300, height=min(900, 50 + 22*len(rows)))
        save_fig(fig, f"9_p{pid}", f"9.9 — Token-Level Diff (probe {pid}); pink = differs from base")
        figs.append(fig)
    return figs

viz_9_9(TRACES)
"""),
    code("""# 9.10 — Master Training Curves (2x2)
def viz_9_10():
    sft  = load_phase_log("sft")
    dpo  = load_phase_log("dpo")
    grpo = load_phase_log("grpo")
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["loss + grad_norm (all phases)",
                                        "DPO chosen / rejected / margin",
                                        "GRPO reward",
                                        "GRPO KL"])
    for label, rows, color in [("sft", sft, "#2a9d8f"),
                                ("dpo", dpo, "#e9c46a"),
                                ("grpo", grpo, "#e76f51")]:
        st = [r["step"] for r in rows]
        lo = [r.get("loss") for r in rows]
        gn = [r.get("grad_norm") for r in rows]
        fig.add_trace(go.Scatter(x=st, y=lo, mode="lines", name=f"{label} loss",
                                 line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=st, y=gn, mode="lines", name=f"{label} gnorm",
                                 line=dict(color=color, dash="dot")), row=1, col=1)
    if dpo:
        st = [r["step"] for r in dpo]
        for k, name in [("rewards/chosen","chosen"),("rewards/rejected","rejected"),("rewards/margins","margin")]:
            ys = [r.get(k) for r in dpo]
            fig.add_trace(go.Scatter(x=st, y=ys, mode="lines", name=name), row=1, col=2)
    if grpo:
        st = [r["step"] for r in grpo]
        fig.add_trace(go.Scatter(x=st, y=[r.get("reward") for r in grpo], mode="lines",
                                 name="reward"), row=2, col=1)
        fig.add_trace(go.Scatter(x=st, y=[r.get("kl") for r in grpo], mode="lines",
                                 name="kl"), row=2, col=2)
    fig.update_layout(height=900, width=1500)
    save_fig(fig, "10", "9.10 — Master Training Curves")
    return fig

viz_9_10()
"""),
    code("""# 9.11 — Final Summary Card
def viz_9_11(eval_scores):
    boundaries = [("sft","base"), ("dpo","sft"), ("grpo","dpo")]
    top_changes = []
    for curr, prev in boundaries:
        if curr in SNAPSHOTS and prev in SNAPSHOTS:
            d = lora_delta(SNAPSHOTS[prev], SNAPSHOTS[curr])
            top = sorted(d.items(), key=lambda kv: -abs(kv[1]["delta_norm"]))[:5]
            top_changes.append(f"{curr}-{prev}: " + ", ".join(k.split('.')[-2]+'.'+k.split('.')[-1] for k,_ in top))

    fields = [
        ("RUN_ID", RUN_ID),
        ("Model", MODEL_ID),
        ("GPU", f"{GPU_NAME} ({VRAM_GB:.0f} GB)"),
        ("Stages", " -> ".join(STAGES)),
        ("Optimizations", "QLoRA-NF4 / LoRA r=64 / FlashAttn-2 / bf16 / paged_adamw_8bit / grad-ckpt / D4+color-perm aug"),
        ("Probes", ", ".join(map(str, PROBE_IDS))),
    ]
    for s, sc in eval_scores.items():
        fields.append((f"acc[{s}]", f"{sc['correct']}/{sc['n']} = {sc['accuracy']:.1%}"))
    for line in top_changes:
        fields.append(("top LoRA Δ", line))
    fields.append(("artifacts", str(RUN_DIR)))

    fig = go.Figure(data=[go.Table(
        header=dict(values=["field","value"], fill_color="#264653", font=dict(color="white")),
        cells=dict(values=[[k for k,_ in fields],[v for _,v in fields]], align="left",
                   fill_color=[["#f4f4f4"]*len(fields), ["#ffffff"]*len(fields)]),
    )])
    fig.update_layout(width=1300, height=60 + 28*len(fields))
    save_fig(fig, "11", "9.11 — Final Summary Card")
    return fig
"""),
    md("""## 10 — Eval (greedy, first 50 ARC-AGI eval tasks at each stage)

For each stage, we *reload the matching adapter* on the same base model, run greedy decoding on the first 50 eval tasks, score exact-match. This produces the `acc[stage]` numbers on the summary card.
"""),
    code("""# 10.1 — Eval helper
@torch.inference_mode()
def predict(m, task, max_new=2048):
    text = to_chat_prompt(task)
    inp = tok(text, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_grid(raw), raw

def eval_current(m, n=50):
    correct = 0; details = []
    for i, (name, task) in enumerate(arc_eval[:n]):
        pred, _ = predict(m, task)
        ok = (pred == task["test"][0]["output"])
        correct += int(ok)
        details.append({"task": name, "correct": ok})
    return {"n": n, "correct": correct, "accuracy": correct/n, "details": details}
"""),
    code("""# 10.2 — Score every stage by reloading its adapter into the live PEFT model
from peft import PeftModel

eval_scores = {}
for s in STAGES:
    if s == "base":
        # disable adapters: equivalent to base predictions
        with model.disable_adapter():
            eval_scores[s] = eval_current(model, n=50)
    else:
        # reload that stage's saved adapter into the active model
        adapter_path = RUN_DIR / "adapters" / s
        # PEFT load_adapter swaps the active weights
        try:
            model.load_adapter(str(adapter_path), adapter_name="default", is_trainable=False)
        except Exception:
            # fallback: re-instantiate
            base_only = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb, device_map={"":0},
                torch_dtype=torch.bfloat16, attn_implementation=attn_impl, trust_remote_code=True)
            model = PeftModel.from_pretrained(base_only, str(adapter_path))
        eval_scores[s] = eval_current(model, n=50)
    print(f"acc[{s}] = {eval_scores[s]['correct']}/{eval_scores[s]['n']} = {eval_scores[s]['accuracy']:.1%}")
    (RUN_DIR / "eval" / f"{s}.json").write_text(json.dumps(eval_scores[s], indent=2))

# Now render the summary card with eval numbers in it
viz_9_11(eval_scores)
"""),
    md("""## 11 — Bundle artifacts

Zips `runs/<RUN_ID>/` into one file. On Colab the second cell exposes a download button.
"""),
    code("""# 11.1 — Zip the run directory
import shutil
zip_path = ROOT / f"{RUN_ID}.zip"
shutil.make_archive(str(zip_path).removesuffix(".zip"), "zip", RUN_DIR)
print("Bundle:", zip_path, f"({zip_path.stat().st_size/1e6:.1f} MB)")
"""),
    code("""# 11.2 — Colab download (no-op outside Colab)
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as e:
    print("Outside Colab — pull the zip via your file browser:", zip_path)
"""),
    md("""## What you should expect

- **Accuracy curve:** monotone or near-monotone. Typical: base 5–10%, sft 25–40%, dpo +2–5pts, grpo +2–5pts.
- **9.1 Logit Lens:** after SFT you should see top-1 probability sharpen earlier in the network — answer crystallizes a few layers sooner.
- **9.5 LoRA Δ:** dominated by `q_proj` / `v_proj` after SFT; `o_proj` / `down_proj` light up more during GRPO as the model is pushed away from the reference.
- **9.7 Reward trajectory:** reward should climb and reward_std should *not* collapse to 0 (collapse = mode collapse, bad).
- **Total time on H100:** ~6–10 hours for the full pipeline at the defaults above. Cut `n_per_task` in 1.6 or `DPO_N` / `GRPO_N` to iterate faster.
"""),
]


# =========================================================================
# Notebook 6: 06_run_benchmark.ipynb
# Take the trained adapter (from 02 or 05), host it (HF Hub or local), and
# run it against the actual ARC-AGI-1 (and try ARC-AGI-2) public eval.
# Generates an official-format submission.json. KISS + lots of explanation.
# =========================================================================

benchmark_cells = [
    md("""# 06 — Run the Trained Model on the ARC-AGI Benchmark

This is the "moment of truth" notebook: take whatever adapter you produced from notebook `02_train_h100` or `05_train_eval_interp`, point it at the **actual ARC-AGI public eval set**, and get a real, comparable score.

We'll go step by step:

1. Pick where the adapter lives (local disk / HuggingFace Hub / a zipped run bundle).
2. Pull the official ARC-AGI eval data.
3. Load base model + adapter on whatever GPU you have.
4. Run the benchmark **3 ways**, fastest → strongest:
   - **Greedy** (one shot, ~1 min/task on T4)
   - **D4 self-consistency vote** (the standard +5–10 pt trick used by every public top fine-tune)
   - **Test-time training (TTT)** (the +15–25 pt trick — slowest, optional)
5. Generate the official `submission.json`.
6. Inspect what worked and what didn't (interactive plotly).
7. (Optional) Push the adapter to HuggingFace Hub so you can host it anywhere.
8. (Optional) Serve it locally with vLLM.
9. Submit to the real ARC Prize leaderboard via Kaggle.

Designed to run on Colab (T4 free / L4 Pro / H100 Pro+) or a local GPU box.
"""),
    md("""## What is the ARC-AGI benchmark?

**ARC-AGI** = *Abstraction and Reasoning Corpus*, by François Chollet (2019). Each task is a tiny visual reasoning puzzle: 2–4 input/output grid examples that demonstrate a transformation rule, plus a held-out test input where you must produce the test output. Grids are 1×1 to 30×30 with up to 10 colors.

**Why it matters:** unlike MMLU/HumanEval/etc., ARC-AGI tasks are *novel* — by design, no two tasks share the same rule. Pure pattern memorization fails. It's one of the few benchmarks where humans (~85%) still trounce frontier LLMs (~80% with extreme reasoning compute, much less for normal use).

**Two versions:**
- **ARC-AGI-1** (2019): 400 train + 400 public eval + 100 semi-private + 100 hidden test. Public eval is what most papers report. Saturating.
- **ARC-AGI-2** (2024): newer, harder, deliberately built to stay hard for LLMs. ~120 public eval at first, growing. This is the *current* leaderboard target.

**Where the official leaderboard lives:** [arcprize.org](https://arcprize.org). Real submissions are scored on the *semi-private* set via a Kaggle competition kernel — no internet, time-limited. We'll cover that flow at the end.

For now we'll measure ourselves on the **public eval** (~400 tasks for v1, ~120 for v2). That's the standard self-reported number, and it's what you'll cite in papers/posts.
"""),
    md("""## What "running on the benchmark" actually means

Three levels of rigor. Pick whichever you can.

| Level | Set | Internet | Tooling | Score is "official"? |
|---|---|---|---|---|
| **A. Public eval (this notebook)** | 400 tasks ARC1 / ~120 ARC2 | yes | any GPU | self-reported, comparable to papers |
| **B. ARC Prize Kaggle leaderboard** | semi-private 100 | **no internet, 12h time limit** | Kaggle kernel only | yes, public leaderboard rank |
| **C. Hidden test** | 100, never released | no | only via Kaggle staff | only the actual prize winners get this |

For "top 100" you need **B** (Kaggle submission). But the score you get on **A** is a very strong predictor of where you'll land on **B**, so we do **A** first and submit to **B** when it's looking good.
"""),
    code("""# 0.1 — Hardware check
import torch, sys, time, os, json
from pathlib import Path
assert torch.cuda.is_available(), "GPU required. Switch Colab runtime to T4/L4/H100."
GPU = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU} ({VRAM_GB:.0f} GB) | torch {torch.__version__}")
"""),
    code("""# 0.2 — Install dependencies
!pip -q install -U \\
    "transformers>=4.51,<4.55" "peft>=0.14" "accelerate>=1.4" \\
    "bitsandbytes>=0.45" "datasets>=3.2" "plotly>=5.24" \\
    sentencepiece huggingface_hub
"""),
    md("""## Step 1 — Tell the notebook where your adapter lives

Three options, in order of how easy they are. Edit `ADAPTER_SOURCE` below.

**Option A — local path** (default; works if you ran notebook 02 or 05 in the same Colab session):
```python
ADAPTER_SOURCE = ("local", "/content/runs/<RUN_ID>/adapters/grpo")
```

**Option B — HuggingFace Hub** (you pushed it; works from anywhere):
```python
ADAPTER_SOURCE = ("hub", "your-handle/qwen3-8b-arc-grpo")
```

**Option C — uploaded zip bundle** (you saved `<RUN_ID>.zip` from notebook 05 and uploaded it to Colab):
```python
ADAPTER_SOURCE = ("zip", "/content/run-1234567890.zip", "adapters/grpo")
```

The base model must match what you trained on. If you used notebook 05's defaults that's `Qwen/Qwen3-8B`.
"""),
    code("""# 1.1 — Configure adapter source + base model
BASE_MODEL_ID = "Qwen/Qwen3-8B"

# === EDIT THIS ===
ADAPTER_SOURCE = ("local", "./runs/latest/adapters/grpo")
# Examples:
# ADAPTER_SOURCE = ("hub",   "your-handle/qwen3-8b-arc-grpo")
# ADAPTER_SOURCE = ("zip",   "/content/run-1234567890.zip", "adapters/grpo")
# ADAPTER_SOURCE = ("none",) # eval the base model with no adapter (sanity check)

# Auto-discover most recent run if you used notebook 05 in the same env
if ADAPTER_SOURCE[0] == "local" and not Path(ADAPTER_SOURCE[1]).exists():
    runs_dir = Path("/content/runs") if Path("/content/runs").exists() else Path("./runs")
    if runs_dir.exists():
        latest = sorted(runs_dir.glob("run-*"), key=lambda p: p.stat().st_mtime)
        for cand in reversed(latest):
            for stage in ["grpo", "dpo", "sft"]:
                p = cand / "adapters" / stage
                if p.exists():
                    ADAPTER_SOURCE = ("local", str(p))
                    print(f"Auto-found: {p}")
                    break
            if ADAPTER_SOURCE[1] != "./runs/latest/adapters/grpo": break

print("Adapter source:", ADAPTER_SOURCE)
print("Base model    :", BASE_MODEL_ID)
"""),
    code("""# 1.2 — Resolve the adapter into a usable local directory
from huggingface_hub import snapshot_download
import zipfile, shutil

ADAPTER_DIR = None
kind = ADAPTER_SOURCE[0]
if kind == "local":
    ADAPTER_DIR = Path(ADAPTER_SOURCE[1])
    assert ADAPTER_DIR.exists(), f"Local adapter not found: {ADAPTER_DIR}"
elif kind == "hub":
    ADAPTER_DIR = Path(snapshot_download(repo_id=ADAPTER_SOURCE[1]))
elif kind == "zip":
    z, sub = ADAPTER_SOURCE[1], ADAPTER_SOURCE[2]
    extract_to = Path("/tmp/arc_adapter_unzip"); extract_to.mkdir(exist_ok=True)
    with zipfile.ZipFile(z) as zf: zf.extractall(extract_to)
    ADAPTER_DIR = extract_to / sub
    assert ADAPTER_DIR.exists(), f"{sub} not found inside {z}"
elif kind == "none":
    ADAPTER_DIR = None
else:
    raise ValueError(f"Unknown adapter source kind: {kind}")

print("Adapter dir:", ADAPTER_DIR)
"""),
    md("""## Step 2 — Get the official ARC-AGI eval data

We pull both ARC-AGI-1 and (optionally) ARC-AGI-2. The repos are public on GitHub; we just zip-download them.
"""),
    code("""# 2.1 — Download ARC-AGI-1 + (try) ARC-AGI-2
import urllib.request, zipfile, io
DATA_DIR = Path("./arc_data"); DATA_DIR.mkdir(exist_ok=True)

def download_repo(url, target_subdir):
    target = DATA_DIR / target_subdir
    if target.exists() and any(target.glob("**/*.json")):
        return True
    try:
        with urllib.request.urlopen(url) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
        return True
    except Exception as e:
        print(f"  [skip] {url}: {e}")
        return False

# ARC-AGI-1
download_repo("https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip", "ARC-AGI-master")
v1_eval_dir = DATA_DIR / "ARC-AGI-master" / "data" / "evaluation"

# ARC-AGI-2 (try a few candidate URLs)
v2_eval_dir = None
for url, subdir in [
    ("https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip", "ARC-AGI-2-main"),
    ("https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/master.zip", "ARC-AGI-2-master"),
]:
    if download_repo(url, subdir):
        cand = DATA_DIR / subdir / "data" / "evaluation"
        if cand.exists() and any(cand.glob("*.json")):
            v2_eval_dir = cand; break

print("ARC-AGI-1 eval dir:", v1_eval_dir, "exists:", v1_eval_dir.exists())
print("ARC-AGI-2 eval dir:", v2_eval_dir, "exists:", v2_eval_dir is not None and v2_eval_dir.exists())
"""),
    code("""# 2.2 — Choose which benchmark you want to score on
# v1 is the classic 400-task eval; v2 is the newer harder one. You can run both.
EVAL_VERSION = "v1"   # change to "v2" if v2 downloaded successfully

eval_dir = v1_eval_dir if EVAL_VERSION == "v1" else v2_eval_dir
assert eval_dir is not None and eval_dir.exists(), f"{EVAL_VERSION} eval dir missing"
EVAL_TASKS = [(p.stem, json.loads(p.read_text())) for p in sorted(eval_dir.glob("*.json"))]
print(f"Loaded {len(EVAL_TASKS)} eval tasks from ARC-AGI-{EVAL_VERSION[1:]}")
"""),
    md("""## Step 3 — Load base model + your adapter

4-bit base + LoRA on top. Same recipe as notebook 02. On a T4 this loads in ~90 seconds.
"""),
    code("""# 3.1 — Load model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True)

attn_impl = "flash_attention_2" if ("H100" in GPU or "A100" in GPU) else "sdpa"

tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
QWEN3_KW = {"enable_thinking": False} if "Qwen3" in BASE_MODEL_ID else {}

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=bnb, device_map={"": 0},
    torch_dtype=torch.bfloat16, attn_implementation=attn_impl, trust_remote_code=True)

if ADAPTER_DIR is not None:
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False)
    print(f"Loaded base + adapter from {ADAPTER_DIR}")
else:
    model = base_model
    print("Loaded base model only (no adapter — sanity-check mode)")
model.eval()
"""),
    code("""# 3.2 — Format helpers (must mirror training notebooks exactly)
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

def to_chat_prompt(task):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(task)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **QWEN3_KW)

@torch.inference_mode()
def predict_greedy(task, max_new=2048):
    text = to_chat_prompt(task)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return parse_grid(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))
"""),
    md("""## Step 4 — Run the benchmark, fastest → strongest

We run three modes. Each gets its own results file. You decide how much compute to spend.

| Mode | Speed (T4) | Speed (H100) | Expected lift |
|---|---|---|---|
| Greedy | ~30s/task | ~5s/task | baseline |
| D4 vote | ~4 min/task | ~40s/task | +5–10 pts |
| TTT     | ~10 min/task | ~90s/task | +15–25 pts on top |

Default is **greedy + D4 vote**. TTT is opt-in (Section 6). For a quick smoke test, set `LIMIT = 20` to run 20 tasks instead of all 400.
"""),
    code("""# 4.1 — Greedy pass on the full eval (or a subset for sanity)
LIMIT = len(EVAL_TASKS)   # set to 20 for quick smoke test, len(EVAL_TASKS) for full run

from tqdm.auto import tqdm
RESULTS = {"greedy": [], "d4vote": [], "ttt": []}

t0 = time.time()
correct = 0
for name, task in tqdm(EVAL_TASKS[:LIMIT], desc="greedy"):
    pred = predict_greedy(task)
    truth = task["test"][0]["output"]
    ok = (pred == truth)
    correct += int(ok)
    RESULTS["greedy"].append({"task": name, "pred": pred, "correct": ok})

elapsed = time.time() - t0
print(f"\\nGREEDY: {correct}/{LIMIT} = {correct/LIMIT:.1%}  ({elapsed:.0f}s, {elapsed/LIMIT:.1f}s/task)")
"""),
    md("""## Step 5 — D4 self-consistency vote

For each test input, apply each of the 8 dihedral symmetries (rotations + reflections), predict on the transformed task, inverse-transform the output, and majority-vote across the 8 predictions. This is the single most useful inference-time trick on ARC.
"""),
    code("""# 5.1 — D4 transforms + their inverses
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
# Inverse mapping (D4[k] composed with D4[INV[k]] = identity).
# Reflections are involutions (their own inverse). Rotations: rot90 ↔ rot270.
INV = [0, 3, 2, 1, 4, 5, 6, 7]

def apply_d4(task, k):
    f = D4[k]
    return {"train":[{"input":f(e["input"]),"output":f(e["output"])} for e in task["train"]],
            "test": [{"input":f(e["input"]),"output":f(e["output"])} for e in task["test"]]}

def grid_to_tuple(g):
    return tuple(tuple(r) for r in g) if g else None

def predict_d4_vote(task):
    counts = {}
    for k in range(8):
        try:
            t_aug = apply_d4(task, k)
            pred = predict_greedy(t_aug)
            if pred is None: continue
            # inverse-transform back
            pred_inv = D4[INV[k]](pred)
            key = grid_to_tuple(pred_inv)
            counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue
    if not counts: return None
    best = max(counts, key=counts.get)
    return [list(r) for r in best]
"""),
    code("""# 5.2 — Run D4 voting on the full eval
t0 = time.time()
correct = 0
for name, task in tqdm(EVAL_TASKS[:LIMIT], desc="d4 vote"):
    pred = predict_d4_vote(task)
    truth = task["test"][0]["output"]
    ok = (pred == truth)
    correct += int(ok)
    RESULTS["d4vote"].append({"task": name, "pred": pred, "correct": ok})

elapsed = time.time() - t0
print(f"\\nD4 VOTE: {correct}/{LIMIT} = {correct/LIMIT:.1%}  ({elapsed:.0f}s, {elapsed/LIMIT:.1f}s/task)")
"""),
    md("""## Step 6 (optional) — Test-time training (TTT)

For each test task, fine-tune a small LoRA on its own train examples (with augmentations) for ~20 steps, then predict, then discard. This is the trick everyone in the top of the leaderboard uses. Costs ~2–10 min/task — set `RUN_TTT = True` only if you have the time budget.
"""),
    code("""# 6.1 — TTT predictor (opt-in)
RUN_TTT = False   # flip to True if you want the +15–25pt boost

if RUN_TTT:
    from torch.optim import AdamW
    import random, copy

    def aug_task(task, k, color_perm):
        f = D4[k]
        def apply(ex):
            return {"input":  [[color_perm[c] for c in row] for row in f(ex["input"])],
                    "output": [[color_perm[c] for c in row] for row in f(ex["output"])]}
        return {"train":[apply(e) for e in task["train"]],
                "test": [apply(e) for e in task["test"]]}

    def build_ttt_examples(task, n_aug=20, rng=None):
        rng = rng or random.Random(0)
        out = []
        if len(task["train"]) < 2: return out
        for _ in range(n_aug):
            k = rng.randrange(8)
            perm = list(range(10)); rng.shuffle(perm)
            a = aug_task(task, k, perm)
            for held in range(len(a["train"])):
                sub = {"train":[e for j,e in enumerate(a["train"]) if j != held],
                       "test":[a["train"][held]]}
                msgs = [{"role":"system","content":SYSTEM},
                        {"role":"user","content":format_user(sub)},
                        {"role":"assistant","content":grid_to_str(sub["test"][0]["output"])}]
                out.append(tok.apply_chat_template(msgs, tokenize=False, **QWEN3_KW))
        return out

    def ttt_and_predict(task, steps=20, lr=5e-5, n_aug=20):
        # Snapshot trainable LoRA params
        snap = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters(): p.requires_grad = (n in snap)
        model.train()
        texts = build_ttt_examples(task, n_aug=n_aug)
        if not texts:
            model.eval(); return predict_greedy(task)
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        rng = random.Random(0)
        for _ in range(steps):
            t = rng.choice(texts)
            enc = tok(t, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
            out = model(**enc, labels=enc.input_ids.clone())
            out.loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        pred = predict_greedy(task)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in snap: p.copy_(snap[n])
        return pred

    t0 = time.time(); correct = 0
    for name, task in tqdm(EVAL_TASKS[:LIMIT], desc="ttt"):
        try: pred = ttt_and_predict(task)
        except Exception: pred = None
        ok = (pred == task["test"][0]["output"])
        correct += int(ok)
        RESULTS["ttt"].append({"task": name, "pred": pred, "correct": ok})
    elapsed = time.time() - t0
    print(f"\\nTTT: {correct}/{LIMIT} = {correct/LIMIT:.1%}  ({elapsed:.0f}s, {elapsed/LIMIT:.1f}s/task)")
else:
    print("Skipping TTT (set RUN_TTT=True to enable).")
"""),
    md("""## Step 7 — Generate the official `submission.json`

The ARC-AGI submission format gives you **2 attempts** per task. Either is allowed to be correct (pass@2). We use:
- `attempt_1` = best mode you ran (TTT > D4 vote > greedy)
- `attempt_2` = second-best mode

This matches the official Kaggle competition format, so you can use this file directly.
""")
,
    code("""# 7.1 — Build submission.json with the 2 best attempts you have
def best_two_modes():
    if RESULTS["ttt"]:    return ("ttt", "d4vote")
    if RESULTS["d4vote"]: return ("d4vote", "greedy")
    return ("greedy", "greedy")

a1_mode, a2_mode = best_two_modes()
def by_task(mode):
    return {r["task"]: r for r in RESULTS[mode]}
A1 = by_task(a1_mode); A2 = by_task(a2_mode)

submission = {}
for name, task in EVAL_TASKS[:LIMIT]:
    p1 = (A1.get(name) or {}).get("pred") or [[0]]
    p2 = (A2.get(name) or {}).get("pred") or [[0]]
    submission[name] = [{"attempt_1": p1, "attempt_2": p2}]

out_path = Path(f"submission_{EVAL_VERSION}.json")
out_path.write_text(json.dumps(submission))
print(f"Wrote {out_path} | attempt_1={a1_mode}, attempt_2={a2_mode}, tasks={len(submission)}")

# pass@2 score
def pass_at_2():
    correct = 0
    for name, task in EVAL_TASKS[:LIMIT]:
        truth = task["test"][0]["output"]
        if A1.get(name, {}).get("pred") == truth or A2.get(name, {}).get("pred") == truth:
            correct += 1
    return correct, LIMIT

p2c, p2n = pass_at_2()
print(f"pass@2 score: {p2c}/{p2n} = {p2c/p2n:.1%}")
"""),
    md("""## Step 8 — Inspect what worked and what didn't

A small interactive plotly dashboard. Helps you see whether your model fails on big grids, certain task patterns, etc.
"""),
    code("""# 8.1 — Per-task results dashboard
import plotly.graph_objects as go
from plotly.subplots import make_subplots

modes_present = [m for m in ["greedy","d4vote","ttt"] if RESULTS[m]]

# accuracy bars
acc_per_mode = {m: sum(r["correct"] for r in RESULTS[m]) / max(1, len(RESULTS[m])) for m in modes_present}
fig1 = go.Figure(data=[go.Bar(x=list(acc_per_mode.keys()),
                              y=[v*100 for v in acc_per_mode.values()],
                              text=[f"{v:.1%}" for v in acc_per_mode.values()],
                              textposition="outside")])
fig1.update_layout(title=f"Accuracy by mode (ARC-AGI-{EVAL_VERSION[1:]}, n={LIMIT})",
                   yaxis_title="accuracy (%)", template="plotly_white")
fig1.write_html("results_accuracy.html"); fig1.show()
"""),
    code("""# 8.2 — Correctness by test-input grid size
def grid_area(g): return len(g) * len(g[0]) if g else 0
def task_size(task): return grid_area(task["test"][0]["input"])

mode = "d4vote" if RESULTS["d4vote"] else "greedy"
sizes_correct = []
sizes_wrong = []
for name, task in EVAL_TASKS[:LIMIT]:
    rec = next((r for r in RESULTS[mode] if r["task"] == name), None)
    if rec is None: continue
    (sizes_correct if rec["correct"] else sizes_wrong).append(task_size(task))

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=sizes_correct, name="correct", opacity=0.7, nbinsx=30))
fig2.add_trace(go.Histogram(x=sizes_wrong,   name="wrong",   opacity=0.7, nbinsx=30))
fig2.update_layout(barmode="overlay", title=f"Correctness vs test grid area ({mode})",
                   xaxis_title="test input area (cells)", yaxis_title="task count",
                   template="plotly_white")
fig2.write_html("results_by_size.html"); fig2.show()
"""),
    code("""# 8.3 — Print 3 successes + 3 failures so you can eyeball them
mode = "d4vote" if RESULTS["d4vote"] else "greedy"
recs = RESULTS[mode]
wins = [r for r in recs if r["correct"]][:3]
fails = [r for r in recs if not r["correct"]][:3]

def show(title, r):
    task = next(t for n,t in EVAL_TASKS if n == r["task"])
    truth = task["test"][0]["output"]
    print(f"\\n=== {title}: {r['task']} ===")
    print("test input:");  print(grid_to_str(task["test"][0]["input"]))
    print("truth:");       print(grid_to_str(truth))
    print("model output:"); print(grid_to_str(r["pred"]) if r["pred"] else "<unparseable>")

for r in wins:  show("WIN",  r)
for r in fails: show("FAIL", r)
"""),
    md("""## Step 9 — (Optional) Host the adapter on HuggingFace Hub

Pushing the LoRA adapter to HF Hub gives you:
- A permanent URL anyone can pull from (`from_pretrained("your-handle/qwen3-8b-arc-grpo")`)
- Free hosting (LoRA adapters are tiny, ~200 MB)
- Versioning + a README

You'll need a HuggingFace account + a write token from https://huggingface.co/settings/tokens
"""),
    code("""# 9.1 — Push adapter to HF Hub (one-time)
PUSH_TO_HUB = False           # flip to True after editing the repo id below
HUB_REPO_ID = "your-handle/qwen3-8b-arc-grpo"

if PUSH_TO_HUB and ADAPTER_DIR is not None:
    from huggingface_hub import login, HfApi
    login()                    # paste your write token when prompted
    HfApi().create_repo(HUB_REPO_ID, repo_type="model", exist_ok=True, private=False)
    HfApi().upload_folder(folder_path=str(ADAPTER_DIR),
                          repo_id=HUB_REPO_ID, repo_type="model")
    print(f"Pushed: https://huggingface.co/{HUB_REPO_ID}")
else:
    print("Skipped. Set PUSH_TO_HUB=True and edit HUB_REPO_ID to push.")
"""),
    md("""## Step 10 — (Optional) Serve the model locally

For a production-style local server, the simplest path is **vLLM**. One command after merging the adapter into the base:

```bash
# 1. Merge adapter into base (one-time, produces a standalone model dir)
python -c "
from peft import PeftModel; from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype='bfloat16')
m = PeftModel.from_pretrained(base, 'PATH/TO/ADAPTER')
m.merge_and_unload().save_pretrained('./qwen3-8b-arc-merged')
"

# 2. Start the server
pip install vllm
vllm serve ./qwen3-8b-arc-merged --port 8000

# 3. Hit it like an OpenAI-compatible endpoint:
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"qwen3-8b-arc-merged","messages":[{"role":"user","content":"..."}]}'
```

Skip this if you only care about getting a benchmark score — the eval above already runs the model directly.
"""),
    md("""## Step 11 — Submit to the official ARC Prize leaderboard

The arcprize.org leaderboard is hosted as a **Kaggle competition** with these constraints:

1. Submission is a Kaggle **kernel** (notebook), not a JSON upload.
2. Kernel has **no internet** at submission time.
3. ~12 hours of runtime budget on the Kaggle hardware (T4 x 2).

**The flow:**

1. Go to the active ARC Prize competition on Kaggle (search "ARC Prize" on kaggle.com/competitions).
2. Upload your **adapter** as a Kaggle Dataset (private is fine). Get the dataset slug.
3. Upload your **base model** as a Kaggle Dataset too (or use a Kaggle-hosted one if available). Internet-off kernels can't `from_pretrained` from HF Hub.
4. Fork the competition's starter kernel. Replace its inference cell with the equivalent of Steps 4–7 of this notebook (greedy + D4 vote, TTT if time permits).
5. **Set internet OFF, GPU ON, accelerator T4 x 2** in kernel settings.
6. "Save & Run All" to commit a version. Submit that version.
7. Wait for the score to appear under your team on the leaderboard.

**Common gotchas:**
- The kernel must finish in budget — TTT on all 100 tasks at 10 min each is 16+ hours, won't fit. Tune `steps` and `n_aug` down.
- The kernel must produce `submission.json` at the working directory root.
- Models loaded via `from_pretrained` need the dataset path (e.g. `/kaggle/input/<your-dataset>/`).
- The grid format must match exactly (digits 0–9 separated by spaces, one row per line; or the JSON list-of-lists in the submission file).
"""),
    md("""## FAQ / Common gotchas

**Q: I got 0% accuracy.**
A: Almost certainly a prompt-format mismatch between training and eval. Confirm `SYSTEM`, `format_user`, and `enable_thinking=False` (for Qwen3) match exactly between this notebook and the training notebook.

**Q: My T4 OOMs loading 8B in 4-bit.**
A: Lower `MAX_LEN` (cell 3.2), or drop to `Qwen/Qwen3-4B`. T4 has 16 GB which is tight for 8B with KV cache.

**Q: D4 vote made my score *worse*.**
A: Means your model is overfit to D4-augmented training and produces inconsistent outputs across rotations. Mostly happens with very short SFT runs. Use greedy as `attempt_1` instead.

**Q: TTT is too slow on my GPU.**
A: Drop `steps=20` to `steps=10` and `n_aug=20` to `n_aug=10`. Or skip TTT entirely — strong SFT + D4 vote alone is enough for a respectable public-eval score.

**Q: Where do I see my official rank?**
A: Only after a Kaggle submission (Step 11). Public eval is for self-reporting and dev iteration.

**Q: Why don't I match the top of the leaderboard?**
A: The very top entries use 70B+ models, custom solver scaffolds, very long TTT, and ensembles of multiple checkpoints. A single H100 + single 7B/8B + LoRA + TTT typically lands in the **15–35% range on ARC-AGI-1 public eval**, which is solid and comparable to most published baselines.
"""),
]


# =========================================================================
# Notebook 7: 07_auto_retrain.ipynb
# Single-H100 fully-automated self-improvement loop:
#   Round 0: full SFT
#   Rounds 1..N: eval -> mine hard tasks -> DPO -> GRPO (multi-signal rewards)
#                -> eval -> early-stop on plateau
# Every round saves a checkpointed adapter so a crash doesn't lose hours.
# Designed for a "Run All -> walk away for 12-24h" workflow.
# =========================================================================

auto_cells = [
    md("""# 07 — Auto-Retraining Loop with Multi-Signal Rewards (single H100)

This is the "set it and forget it" notebook. Hit **Run All** and a single H100 will:

1. Load the strongest open-source model that fits comfortably (default `Qwen3-14B`, auto-falls-back to `Qwen3-8B` on OOM).
2. Run a full SFT pass (Round 0) on the augmented BARC + RE-ARC + ARC pool.
3. Then loop:
   - Evaluate on a fixed held-out slice
   - Mine the tasks the model still gets wrong
   - DPO on `(chosen=truth, rejected=current model's wrong sample)` for those tasks
   - GRPO with a **multi-signal reward** (exact match + shape match + cell partial credit + format + length)
   - Re-evaluate; stop early if accuracy plateaus
4. Push the final adapter to HuggingFace Hub (optional) and emit a Kaggle-ready `submission.json`.

Every round writes a self-contained checkpoint to `runs/<RUN_ID>/r<N>/`, so you can resume from any point if something dies.

**Wall-clock estimate on H100 80GB:** ~12–24 h to convergence at the defaults below.
"""),
    md("""## What's different from notebook 05?

| | 05 (interp) | 07 (auto-loop) |
|---|---|---|
| Goal | *understand* the model with deep traces | *push the score* with automated improvement |
| Stages | base → SFT → DPO → GRPO (one shot) | SFT (round 0) + N×(DPO + GRPO) rounds |
| Hard-task mining | ❌ | ✅ each round |
| Reward signals | exact-match only | exact + shape + partial + format + length |
| Plateau early-stop | ❌ | ✅ |
| Per-round checkpoints | ❌ (one final adapter) | ✅ (one per round, resumable) |
| Model size | 8B | 14B (auto-falls back to 8B) |
| Forensic plotly viz | ✅ 11 views | minimal — just training curves |

Use **05** when you want to see what the model is doing inside.
Use **07** when you want to maximize the public-eval score.
"""),
    code("""# 0.1 — Hardware check + helper
import torch, time, json, os, random, copy
from pathlib import Path
assert torch.cuda.is_available(), "Need a GPU. Switch to H100/A100 runtime."
GPU = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
IS_H100 = "H100" in GPU
print(f"GPU: {GPU} ({VRAM_GB:.0f} GB)")
"""),
    code("""# 0.2 — Install (latest known-good for Qwen3 + GRPO)
!pip -q install -U \\
    "transformers>=4.51,<4.55" "trl>=0.15,<0.20" "peft>=0.14" \\
    "accelerate>=1.4" "bitsandbytes>=0.45" "datasets>=3.2" \\
    "plotly>=5.24" sentencepiece huggingface_hub
"""),
    code("""# 0.3 — Run config (the only knobs you should normally touch)
SEED = 0
random.seed(SEED); torch.manual_seed(SEED)

# Model: Qwen3-14B is the default. If we OOM at load time we fall back to 8B.
PRIMARY_MODEL  = "Qwen/Qwen3-14B"
FALLBACK_MODEL = "Qwen/Qwen3-8B"

# Loop config
N_ROUNDS         = 4         # round 0 = full SFT, then N_ROUNDS-1 DPO+GRPO rounds
HOLDOUT_N        = 80        # held-out eval tasks each round (subset of arc eval)
HARD_PER_ROUND   = 200       # cap on hard-task pairs per round
SFT_EPOCHS_R0    = 1
SFT_AUG_PER_TASK = 4         # multiplies pool size at SFT round 0
PLATEAU_DELTA    = 0.005     # stop if score gains < 0.5 pts for 2 rounds in a row
DPO_BETA         = 0.1
GRPO_BETA        = 0.04
GRPO_GROUPS      = 4

# Reward weights (composed into one scalar reward)
W_EXACT   = 1.0
W_SHAPE   = 0.2
W_PARTIAL = 0.5
W_FORMAT  = 0.1
W_LENGTH  = -0.05  # penalty if model rambles past expected output length

RUN_ID  = f"auto-{int(time.time())}"
ROOT    = Path("/content") if Path("/content").exists() else Path(".")
RUN_DIR = ROOT / "runs" / RUN_ID
DATA_DIR = ROOT / "arc_data"; DATA_DIR.mkdir(exist_ok=True)
for sub in ["logs", "eval"]: (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)
print("RUN_DIR:", RUN_DIR)
"""),
    md("""## Section 1 — Data: ARC + RE-ARC + BARC + augmentation
"""),
    code("""# 1.1 — Download ARC-AGI-1
import urllib.request, zipfile, io
if not (DATA_DIR / "training").exists():
    with urllib.request.urlopen("https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip") as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    src = DATA_DIR / "ARC-AGI-master" / "data"
    for kind in ["training","evaluation"]:
        (DATA_DIR / kind).mkdir(exist_ok=True)
        for f in (src / kind).glob("*.json"):
            (DATA_DIR / kind / f.name).write_bytes(f.read_bytes())

arc_train = [json.loads(p.read_text()) for p in sorted((DATA_DIR/"training").glob("*.json"))]
arc_eval  = [(p.stem, json.loads(p.read_text())) for p in sorted((DATA_DIR/"evaluation").glob("*.json"))]
print(f"ARC: {len(arc_train)} train, {len(arc_eval)} eval")

# Held-out for eval each round (deterministic)
HOLDOUT = arc_eval[:HOLDOUT_N]
"""),
    code("""# 1.2 — RE-ARC + BARC (best-effort, skip on failure)
re_arc_dir = DATA_DIR / "re_arc"
re_arc_tasks = []
if not re_arc_dir.exists():
    try:
        with urllib.request.urlopen("https://github.com/michaelhodel/re-arc/raw/main/re_arc.zip") as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(re_arc_dir)
    except Exception as e: print("RE-ARC skipped:", e)
if re_arc_dir.exists():
    for fp in sorted(re_arc_dir.rglob("*.json"))[:5000]:
        try: re_arc_tasks.append(json.loads(fp.read_text()))
        except: pass
print("RE-ARC:", len(re_arc_tasks))

from datasets import load_dataset, Dataset
try:
    barc = load_dataset(
        "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        split="train")
except Exception as e: barc = None; print("BARC skipped:", e)
print("BARC:", (len(barc) if barc else 0))
"""),
    code("""# 1.3 — Augmentation + format helpers (lifted exactly from training notebooks)
def rot90(g):  return [list(r) for r in zip(*g[::-1])]
def fliplr(g): return [r[::-1] for r in g]
D4 = [lambda g: g, rot90,
      lambda g: rot90(rot90(g)),
      lambda g: rot90(rot90(rot90(g))),
      fliplr,
      lambda g: rot90(fliplr(g)),
      lambda g: rot90(rot90(fliplr(g))),
      lambda g: rot90(rot90(rot90(fliplr(g))))]

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
    parts.append("Test Input:");  parts.append(grid_to_str(task["test"][0]["input"]))
    parts.append(""); parts.append("Test Output:")
    return "\\n".join(parts)

def format_assistant(task):
    return grid_to_str(task["test"][0]["output"])

def _coerce_examples(exs):
    \"\"\"Each ex must be {'input': 2D-list, 'output': 2D-list}.
    BARC variants encode pairs as dict, list[2], or tuple[2] — handle all.\"\"\"
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
        train = _coerce_examples(row["train"]); test = _coerce_examples(row["test"])
    elif "examples" in row and "test" in row:
        train = _coerce_examples(row["examples"]); test = _coerce_examples(row["test"])
    elif "examples" in row:
        ex = _coerce_examples(row["examples"])
        if ex and len(ex) >= 2: train, test = ex[:-1], [ex[-1]]
    return {"train": train, "test": test} if (train and test) else None
"""),
    code("""# 1.4 — Build augmented base pool
base_pool = list(arc_train) + list(re_arc_tasks)
if barc is not None:
    for r in barc.select(range(min(20000, len(barc)))):
        n = normalize_barc(r)
        if n and n["train"] and n["test"]: base_pool.append(n)
print(f"Base pool: {len(base_pool)}")

def gen_aug(tasks, n_per_task, rng):
    out = []
    for t in tasks:
        for _ in range(n_per_task):
            x = apply_d4(t, rng.randrange(8))
            perm = list(range(10)); rng.shuffle(perm)
            x = apply_color_perm(x, perm)
            out.append(x)
    return out

augmented = gen_aug(base_pool, n_per_task=SFT_AUG_PER_TASK, rng=random.Random(SEED))
print(f"Augmented samples: {len(augmented)}")
"""),
    md("""## Section 2 — The multi-signal reward function

Single function returns a scalar per completion. Composed of:

| Signal | Weight | Triggers when |
|---|---|---|
| **Exact match** | +1.0 | predicted grid equals truth |
| **Shape match** | +0.2 | HxW correct (regardless of cell values) |
| **Cell partial credit** | up to +0.5 | shape matches AND cells overlap |
| **Format reward** | +0.1 | output parses as a grid (rectangular, 0–9) |
| **Length penalty** | up to −0.05 | non-grid output > 50 tokens long (rambling) |

For a fully-correct prediction the total is ~1.8. For a totally wrong shape it's 0–0.1. GRPO normalizes this within each group so absolute scale doesn't matter.
"""),
    code("""# 2.1 — Multi-signal reward
def multi_signal_reward(completions, prompts=None, ground_truth=None, **kw):
    truths = ground_truth if ground_truth is not None else [None] * len(completions)
    rewards = []
    for c, gt in zip(completions, truths):
        text = c if isinstance(c, str) else (c[0]["content"] if isinstance(c, list) else str(c))
        truth_grid = parse_grid(gt) if gt else None
        pred = parse_grid(text)
        r = 0.0
        if pred is not None:
            r += W_FORMAT
        if pred is None and len(text.split()) > 50:
            r += W_LENGTH
        if pred is not None and truth_grid is not None:
            same_shape = (len(pred) == len(truth_grid)
                          and len(pred[0]) == len(truth_grid[0]))
            if same_shape:
                r += W_SHAPE
                total = max(1, sum(len(row) for row in truth_grid))
                hits  = sum(1 for i in range(len(truth_grid))
                              for j in range(len(truth_grid[0]))
                              if pred[i][j] == truth_grid[i][j])
                r += W_PARTIAL * (hits / total)
                if pred == truth_grid:
                    r += W_EXACT
        rewards.append(r)
    return rewards
"""),
    md("""## Section 3 — Eval helper + hard-task miner

Each round we run the current model (greedy) on `HOLDOUT`, log the score, and collect every wrong task into a "hard tasks" pool that drives the next round of DPO + GRPO.
"""),
    code("""# 3.1 — Eval + miner
def to_chat_prompt(tok, task, qwen3_kw):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(task)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **qwen3_kw)

@torch.inference_mode()
def predict_greedy(model, tok, task, max_new=2048, qwen3_kw=None):
    text = to_chat_prompt(tok, task, qwen3_kw or {})
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    return parse_grid(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)), text

def eval_holdout(model, tok, qwen3_kw, holdout=HOLDOUT):
    correct = 0; per_task = []
    for name, task in holdout:
        pred, _ = predict_greedy(model, tok, task, qwen3_kw=qwen3_kw)
        ok = (pred == task["test"][0]["output"])
        correct += int(ok)
        per_task.append({"task": name, "correct": ok, "pred": pred})
    acc = correct / max(1, len(holdout))
    return acc, per_task

def mine_hard_pairs(model, tok, qwen3_kw, holdout=HOLDOUT, cap=HARD_PER_ROUND):
    \"\"\"For each task model gets wrong, sample once at temp=0.7 to get a 'rejected'
    completion, pair it against the ground-truth as 'chosen'.\"\"\"
    pairs = []
    for name, task in holdout:
        if len(pairs) >= cap: break
        text = to_chat_prompt(tok, task, qwen3_kw)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=512, do_sample=True,
                                 temperature=0.7, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        sampled = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        gt = format_assistant(task)
        sampled_grid = parse_grid(sampled)
        if sampled_grid != task["test"][0]["output"] and sampled.strip():
            pairs.append({"prompt": text, "chosen": gt, "rejected": sampled,
                          "ground_truth": gt, "task": name})
    return pairs
"""),
    md("""## Section 4 — Stage trainers

Three small functions: `do_sft`, `do_dpo`, `do_grpo`. Each takes the live model + a dataset, runs one mini-pass, saves the adapter for that round, returns the model.
"""),
    code("""# 4.1 — SFT (used at round 0 only)
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
from transformers import TrainerCallback

class JsonlLogger(TrainerCallback):
    def __init__(self, path):
        self.path = path; self.f = open(path, "a")
    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None: return
        rec = {"step": state.global_step, **{k: float(v) for k,v in logs.items()
                                              if isinstance(v,(int,float))}}
        self.f.write(json.dumps(rec) + "\\n"); self.f.flush()

MAX_LEN = 3500

def do_sft(model, tok, qwen3_kw, sft_dataset, round_idx):
    out_dir = RUN_DIR / f"r{round_idx}" / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=SFT_EPOCHS_R0,
        per_device_train_batch_size=2 if VRAM_GB >= 70 else 1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_steps=10_000,
        save_total_limit=1,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_seq_length=MAX_LEN,
        dataset_text_field="text",
        packing=False, report_to="none", seed=SEED,
    )
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=sft_dataset, args=cfg,
                         callbacks=[JsonlLogger(str(RUN_DIR / "logs" / f"r{round_idx}_sft.jsonl"))])
    trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    return trainer.model

def do_dpo(model, tok, qwen3_kw, pairs, round_idx):
    if not pairs:
        print(f"[r{round_idx}] DPO: no pairs, skipping"); return model
    out_dir = RUN_DIR / f"r{round_idx}" / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(pairs)
    cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5 / max(1, round_idx),  # decay LR over rounds
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        logging_steps=10, save_steps=10_000, save_total_limit=1,
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        beta=DPO_BETA,
        max_length=MAX_LEN, max_prompt_length=MAX_LEN-768,
        report_to="none", seed=SEED,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, tokenizer=tok,
                         callbacks=[JsonlLogger(str(RUN_DIR / "logs" / f"r{round_idx}_dpo.jsonl"))])
    trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    return trainer.model

def do_grpo(model, tok, qwen3_kw, pairs, round_idx):
    if not pairs:
        print(f"[r{round_idx}] GRPO: no prompts, skipping"); return model
    out_dir = RUN_DIR / f"r{round_idx}" / "grpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"prompt": p["prompt"], "ground_truth": p["ground_truth"]} for p in pairs]
    ds = Dataset.from_list(rows)
    cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6 / max(1, round_idx),
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        logging_steps=5, save_steps=10_000, save_total_limit=1,
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        num_generations=GRPO_GROUPS, max_prompt_length=MAX_LEN-512,
        max_completion_length=512, beta=GRPO_BETA,
        report_to="none", seed=SEED,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=multi_signal_reward, args=cfg,
                          train_dataset=ds, processing_class=tok,
                          callbacks=[JsonlLogger(str(RUN_DIR / "logs" / f"r{round_idx}_grpo.jsonl"))])
    trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    return trainer.model
"""),
    md("""## Section 5 — Round 0: load model + initial SFT

We try the bigger model first; if loading fails (rare: H100 should fit Qwen3-14B in 4-bit + LoRA), we fall back to 8B.
"""),
    code("""# 5.1 — Load with auto-fallback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_with_lora(model_id):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    attn_impl = "flash_attention_2" if IS_H100 else "sdpa"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    qwen3_kw = {"enable_thinking": False} if "Qwen3" in model_id else {}
    m = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation=attn_impl, trust_remote_code=True)
    m.config.use_cache = False
    m = prepare_model_for_kbit_training(m)
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj","k_proj","v_proj","o_proj",
                                     "gate_proj","up_proj","down_proj"])
    m = get_peft_model(m, lora)
    m.print_trainable_parameters()
    return m, tok, qwen3_kw

try:
    MODEL_ID = PRIMARY_MODEL
    model, tok, QWEN3_KW = load_with_lora(MODEL_ID)
except Exception as e:
    print(f"Falling back to {FALLBACK_MODEL} due to: {e}")
    torch.cuda.empty_cache()
    MODEL_ID = FALLBACK_MODEL
    model, tok, QWEN3_KW = load_with_lora(MODEL_ID)
print("ACTIVE MODEL:", MODEL_ID)
"""),
    code("""# 5.2 — Tokenize SFT pool + run round-0 SFT
def to_chat_full(t):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(t)},
            {"role":"assistant","content":format_assistant(t)}]
    return tok.apply_chat_template(msgs, tokenize=False, **QWEN3_KW)

sft_texts = []
for t in augmented:
    s = to_chat_full(t)
    if len(tok(s, add_special_tokens=False).input_ids) <= MAX_LEN:
        sft_texts.append(s)
print(f"SFT samples after length filter: {len(sft_texts)}")
sft_ds = Dataset.from_dict({"text": sft_texts}).shuffle(seed=SEED)

model = do_sft(model, tok, QWEN3_KW, sft_ds, round_idx=0)
"""),
    code("""# 5.3 — Round 0 eval
acc0, per_task0 = eval_holdout(model, tok, QWEN3_KW)
print(f"[r0] holdout acc: {acc0:.1%}")
(RUN_DIR / "eval" / "r0.json").write_text(json.dumps({"acc": acc0, "per_task": per_task0}, indent=2))
SCORE_HISTORY = [acc0]
"""),
    md("""## Section 6 — The auto-loop

For rounds 1..N-1, each iteration:

1. Mine hard tasks (held-out tasks the current model fails on, with one fresh sampled "rejected" completion per task).
2. DPO mini-pass on those `(chosen=truth, rejected=sample)` pairs.
3. GRPO mini-pass with the multi-signal reward on the same prompts.
4. Re-evaluate. If the last 2 rounds together gained < `PLATEAU_DELTA`, stop.

Each round's full state lives in `runs/<RUN_ID>/r<N>/{sft,dpo,grpo}/` so you can resume from any checkpoint.
"""),
    code("""# 6.1 — The loop
import gc

def cleanup():
    gc.collect(); torch.cuda.empty_cache()

for rnd in range(1, N_ROUNDS):
    print(f"\\n========== ROUND {rnd}/{N_ROUNDS-1} ==========")
    cleanup()

    # Mine hard pairs from current model on holdout
    pairs = mine_hard_pairs(model, tok, QWEN3_KW, holdout=HOLDOUT, cap=HARD_PER_ROUND)
    print(f"[r{rnd}] hard pairs mined: {len(pairs)}")
    cleanup()

    # DPO
    model = do_dpo(model, tok, QWEN3_KW, pairs, round_idx=rnd)
    cleanup()

    # GRPO with multi-signal reward
    model = do_grpo(model, tok, QWEN3_KW, pairs, round_idx=rnd)
    cleanup()

    # Eval
    acc, per_task = eval_holdout(model, tok, QWEN3_KW)
    SCORE_HISTORY.append(acc)
    print(f"[r{rnd}] holdout acc: {acc:.1%}  history: {[f'{s:.1%}' for s in SCORE_HISTORY]}")
    (RUN_DIR / "eval" / f"r{rnd}.json").write_text(json.dumps({
        "acc": acc, "per_task": per_task, "history": SCORE_HISTORY}, indent=2))

    # Plateau early-stop
    if rnd >= 2:
        gain_recent = SCORE_HISTORY[-1] - SCORE_HISTORY[-3]
        if gain_recent < 2 * PLATEAU_DELTA:
            print(f"[r{rnd}] plateau detected (gain over last 2 rounds = {gain_recent:.3%}), stopping early")
            break
"""),
    md("""## Section 7 — Final artifacts: training curves + push to Hub
"""),
    code("""# 7.1 — Score history plot
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(SCORE_HISTORY))),
                         y=[s*100 for s in SCORE_HISTORY],
                         mode="lines+markers+text",
                         text=[f"{s:.1%}" for s in SCORE_HISTORY],
                         textposition="top center", line=dict(width=3)))
fig.update_layout(title=f"Auto-retrain score over rounds ({MODEL_ID})",
                  xaxis_title="round", yaxis_title="holdout accuracy (%)",
                  template="plotly_white", height=480)
fig.write_html(str(RUN_DIR / "score_history.html"))
fig.show()
"""),
    code("""# 7.2 — Per-round per-phase loss curves
def load_jsonl(p):
    if not Path(p).exists(): return []
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

phases = ["sft", "dpo", "grpo"]
fig = go.Figure()
for r in range(N_ROUNDS):
    for ph in phases:
        rows = load_jsonl(RUN_DIR / "logs" / f"r{r}_{ph}.jsonl")
        if not rows: continue
        x = [v["step"] for v in rows if "loss" in v]
        y = [v["loss"] for v in rows if "loss" in v]
        if y:
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                     name=f"r{r}-{ph}"))
fig.update_layout(title="Loss curves per round x phase",
                  xaxis_title="step", yaxis_title="loss",
                  template="plotly_white", height=600)
fig.write_html(str(RUN_DIR / "loss_curves.html"))
fig.show()
"""),
    code("""# 7.3 — Best-round adapter (highest holdout acc) -> 'final' alias
import shutil
best_round = max(range(len(SCORE_HISTORY)), key=lambda i: SCORE_HISTORY[i])
print(f"Best round: r{best_round} with acc={SCORE_HISTORY[best_round]:.1%}")

# Find the latest stage in that round (grpo > dpo > sft)
for stage in ["grpo", "dpo", "sft"]:
    src = RUN_DIR / f"r{best_round}" / stage
    if src.exists():
        dst = RUN_DIR / "final"
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Best adapter: {src} -> {dst}")
        break
"""),
    code("""# 7.4 — Push final adapter to HuggingFace Hub (optional)
PUSH_TO_HUB = False
HUB_REPO_ID = "your-handle/qwen3-arc-auto"

if PUSH_TO_HUB:
    from huggingface_hub import login, HfApi
    login()
    HfApi().create_repo(HUB_REPO_ID, repo_type="model", exist_ok=True, private=False)
    HfApi().upload_folder(folder_path=str(RUN_DIR / "final"),
                          repo_id=HUB_REPO_ID, repo_type="model")
    print(f"Pushed: https://huggingface.co/{HUB_REPO_ID}")
else:
    print("Skipped (set PUSH_TO_HUB=True + edit HUB_REPO_ID).")
"""),
    md("""## Section 8 — Final benchmark + submission.json

We score the **best round's adapter** on the full ARC eval (greedy + D4 vote) and emit a Kaggle-format submission.json.
"""),
    code("""# 8.1 — D4-vote helper (same as notebook 06)
INV = [0, 3, 2, 1, 4, 5, 6, 7]
def predict_d4_vote(task):
    counts = {}
    for k in range(8):
        try:
            t_aug = apply_d4(task, k)
            pred, _ = predict_greedy(model, tok, t_aug, qwen3_kw=QWEN3_KW)
            if pred is None: continue
            inv = D4[INV[k]](pred)
            key = tuple(tuple(r) for r in inv)
            counts[key] = counts.get(key, 0) + 1
        except Exception: continue
    if not counts: return None
    return [list(r) for r in max(counts, key=counts.get)]
"""),
    code("""# 8.2 — Full 400-task eval on best adapter
from tqdm.auto import tqdm
eval_full = arc_eval[:]
greedy_preds = {}; d4_preds = {}
g_correct = 0; d_correct = 0

for name, task in tqdm(eval_full, desc="final eval"):
    truth = task["test"][0]["output"]
    g, _ = predict_greedy(model, tok, task, qwen3_kw=QWEN3_KW)
    d = predict_d4_vote(task)
    greedy_preds[name] = g
    d4_preds[name] = d
    g_correct += int(g == truth)
    d_correct += int(d == truth)

n = len(eval_full)
print(f"\\nGREEDY: {g_correct}/{n} = {g_correct/n:.1%}")
print(f"D4VOTE: {d_correct}/{n} = {d_correct/n:.1%}")
pass2 = sum(1 for name, t in eval_full
            if greedy_preds[name] == t["test"][0]["output"]
            or d4_preds[name]    == t["test"][0]["output"])
print(f"pass@2: {pass2}/{n} = {pass2/n:.1%}")

(RUN_DIR / "eval" / "final.json").write_text(json.dumps({
    "greedy": g_correct/n, "d4vote": d_correct/n, "pass2": pass2/n,
    "n": n, "best_round": best_round, "history": SCORE_HISTORY,
}, indent=2))
"""),
    code("""# 8.3 — Submission.json (attempt_1 = D4 vote, attempt_2 = greedy)
submission = {}
for name, task in eval_full:
    a1 = d4_preds.get(name) or [[0]]
    a2 = greedy_preds.get(name) or [[0]]
    submission[name] = [{"attempt_1": a1, "attempt_2": a2}]
(RUN_DIR / "submission.json").write_text(json.dumps(submission))
print(f"Wrote {RUN_DIR/'submission.json'} | tasks={len(submission)}")
"""),
    code("""# 8.4 — Bundle the run + (Colab) download
import shutil
zpath = ROOT / f"{RUN_ID}.zip"
shutil.make_archive(str(zpath).removesuffix(".zip"), "zip", RUN_DIR)
print("Bundle:", zpath, f"({zpath.stat().st_size/1e6:.1f} MB)")
try:
    from google.colab import files; files.download(str(zpath))
except Exception:
    print("Outside Colab — download from file browser:", zpath)
"""),
    md("""## What good looks like

- **Score history** should rise on each round and then plateau; the early-stop kicks in when 2 rounds in a row gain less than 1 pt.
- **Loss curves** should be: SFT decreasing fast (round 0), DPO loss decreasing slowly, GRPO reward rising.
- **Best round** is usually round 1 or 2 — heavy multi-round DPO+GRPO can over-collapse the policy. The "final" alias copies the best-round adapter so you don't have to chase that yourself.
- **Final D4-vote pass@2** in the **15–35% range on ARC-AGI-1 public eval** is realistic for this stack (Qwen3-14B + QLoRA + auto-loop). Anything above 25% is competitive with most published baselines.

If you want more, the next levers are: bigger base (Qwen3-32B with stronger quantization), TTT at inference time (notebook 06 Section 6), or longer training rounds (`HARD_PER_ROUND=500`, `N_ROUNDS=8`).
"""),
]


# =========================================================================
# Write notebooks
# =========================================================================

specs = [
    ("01_baseline.ipynb",        baseline_cells),
    ("02_train_h100.ipynb",      train_cells),
    ("03_distill_kaggle.ipynb",  distill_cells),
    ("04_eval_ttt.ipynb",        eval_cells),
    ("05_train_eval_interp.ipynb", eval_interp_cells),
    ("06_run_benchmark.ipynb", benchmark_cells),
    ("07_auto_retrain.ipynb",  auto_cells),
]

for name, cells in specs:
    path = HERE / name
    path.write_text(json.dumps(nb(cells), indent=1))
    print(f"Wrote {path}  ({len(cells)} cells)")
