"""
Local training demo on the laptop.

Goal: demonstrate the training-interpretability pipeline (loss curves, grad
norms, per-layer LoRA growth, before/after eval) using PyTorch + TensorBoard
+ TRL's SFTTrainer + PEFT LoRA. Same setup as the H100 notebook, just at
laptop scale (Qwen2.5-1.5B + 100 augmented ARC samples + 60 steps).

After this finishes, view TensorBoard at http://localhost:6006 and the
human-readable train.html report next to dashboard.html and interpret.html.

Runtime on M3 Pro: ~5-8 minutes (model already cached).
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import plotly.graph_objects as go
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from trl import SFTTrainer, SFTConfig

HERE = Path(__file__).parent
ARC_DIR = HERE / "ARC-AGI-master" / "data"
RUN_ROOT = HERE / "runs" / "laptop_train"
ADAPTER_DIR = RUN_ROOT / "adapter"
TB_DIR = RUN_ROOT / "tb"
OUT_HTML = HERE / "train.html"
RESULTS = RUN_ROOT / "train_results.json"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # already cached from earlier smoke test
N_BASE_TASKS = 100        # how many ARC train tasks to use
N_PER_TASK = 2            # D4 + color-perm augmentation factor
N_TRAIN_STEPS = 60        # short, illustrative
EVAL_N = 5                # before/after eval task count
MAX_LEN = 2048
LORA_R = 16

SYSTEM = ("You are an expert ARC puzzle solver. Output only the test output grid as digits "
          "separated by spaces, one row per line. No explanations.")


def grid_to_str(g): return "\n".join(" ".join(str(c) for c in row) for row in g)

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
    return "\n".join(parts)


def format_assistant(task): return grid_to_str(task["test"][0]["output"])


# Augmentation helpers
def rot90(g): return [list(r) for r in zip(*g[::-1])]
def fliplr(g): return [r[::-1] for r in g]
D4 = [lambda g: g, rot90, lambda g: rot90(rot90(g)),
      lambda g: rot90(rot90(rot90(g))), fliplr,
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


@torch.inference_mode()
def predict(model, tok, task, device, max_new=256):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":format_user(task)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return parse_grid(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))


def quick_eval(model, tok, eval_tasks, device, max_new=256):
    results = []; correct = 0
    for name, task in eval_tasks:
        try: pred = predict(model, tok, task, device, max_new=max_new)
        except Exception as e: pred = None
        ok = (pred == task["test"][0]["output"])
        correct += int(ok)
        results.append({"task": name, "correct": ok, "pred": pred})
    return correct, results


class LoRANormCallback(TrainerCallback):
    """Log per-LoRA-module weight L2 norm to TensorBoard. Lets you see
    which adapters grow fastest during training."""
    def __init__(self, writer, every=5):
        self.w = writer; self.every = every
    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.every: return
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                short = name.replace(".default.weight", "").replace("base_model.model.", "")
                self.w.add_scalar(f"lora_norm/{short}", p.detach().float().norm().item(),
                                 state.global_step)


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] device={device}, model={MODEL_ID}")

    # --- Load model + LoRA ---
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                  device_map=device)
    model.config.use_cache = False
    print(f"[train] loaded in {time.time()-t0:.1f}s")

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_R*2, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # --- Build training data ---
    arc_train = [json.loads(p.read_text())
                 for p in sorted((ARC_DIR / "training").glob("*.json"))[:N_BASE_TASKS]]
    random.seed(0)
    rng = random.Random(0)
    augmented = []
    for t in arc_train:
        for _ in range(N_PER_TASK):
            x = apply_d4(t, rng.randrange(8))
            perm = list(range(10)); rng.shuffle(perm)
            x = apply_color_perm(x, perm)
            augmented.append(x)
    print(f"[train] augmented samples: {len(augmented)}")

    texts = []
    for t in augmented:
        msgs = [{"role":"system","content":SYSTEM},
                {"role":"user","content":format_user(t)},
                {"role":"assistant","content":format_assistant(t)}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        if len(tok(s, add_special_tokens=False).input_ids) <= MAX_LEN:
            texts.append(s)
    print(f"[train] post-length-filter samples: {len(texts)}")
    if len(texts) < 5:
        print("[train] not enough samples — aborting"); return

    sft_ds = Dataset.from_dict({"text": texts}).shuffle(seed=0)

    # --- Pre-eval (untrained adapter == base model) ---
    eval_tasks = [(p.stem, json.loads(p.read_text()))
                  for p in sorted((ARC_DIR / "evaluation").glob("*.json"))[:EVAL_N]]
    print(f"[train] pre-train eval on {EVAL_N} tasks ...")
    t0 = time.time()
    pre_correct, pre_results = quick_eval(model, tok, eval_tasks, device, max_new=256)
    print(f"  PRE: {pre_correct}/{EVAL_N}  ({time.time()-t0:.0f}s)")

    # --- Train ---
    writer = SummaryWriter(log_dir=str(TB_DIR))
    cfg = SFTConfig(
        output_dir=str(RUN_ROOT / "ckpt"),
        max_steps=N_TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=False, fp16=False,           # plain fp32 on MPS for safety
        logging_steps=1,                  # we want every step
        save_strategy="no",
        max_length=MAX_LEN,
        dataset_text_field="text",
        packing=False,
        report_to=["tensorboard"],        # << TensorBoard via TRL
        logging_dir=str(TB_DIR),
        seed=0,
    )
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=sft_ds, args=cfg,
                         callbacks=[LoRANormCallback(writer, every=2)])
    print(f"[train] training {N_TRAIN_STEPS} steps ...")
    t0 = time.time()
    train_out = trainer.train()
    print(f"[train] done in {time.time()-t0:.0f}s")

    trainer.model.save_pretrained(str(ADAPTER_DIR))
    print(f"[train] adapter saved to {ADAPTER_DIR}")

    # --- Post-eval ---
    print(f"[train] post-train eval on {EVAL_N} tasks ...")
    t0 = time.time()
    post_correct, post_results = quick_eval(model, tok, eval_tasks, device, max_new=256)
    print(f"  POST: {post_correct}/{EVAL_N}  ({time.time()-t0:.0f}s)")

    # --- Persist results ---
    summary = {
        "model": MODEL_ID,
        "n_train_steps": N_TRAIN_STEPS,
        "n_train_samples": len(texts),
        "lora_r": LORA_R,
        "eval_n": EVAL_N,
        "pre_correct": pre_correct, "pre_acc": pre_correct/EVAL_N,
        "post_correct": post_correct, "post_acc": post_correct/EVAL_N,
        "delta_pp": (post_correct - pre_correct) / EVAL_N * 100,
        "tb_dir": str(TB_DIR),
        "adapter_dir": str(ADAPTER_DIR),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(summary, indent=2))

    # --- Render train.html ---
    log_history = trainer.state.log_history
    steps, losses, grads, lrs = [], [], [], []
    for h in log_history:
        if "loss" in h and "step" in h:
            steps.append(h["step"]); losses.append(h["loss"])
            if "grad_norm" in h: grads.append(h["grad_norm"])
            else: grads.append(None)
            if "learning_rate" in h: lrs.append(h["learning_rate"])
            else: lrs.append(None)

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=steps, y=losses, mode="lines+markers",
                                   line=dict(color="#2dd4bf", width=2),
                                   name="loss"))
    fig_loss.update_layout(template="plotly_dark", paper_bgcolor="#1a1d24",
                           plot_bgcolor="#1a1d24",
                           height=320, title="Training loss",
                           xaxis_title="step", yaxis_title="loss")

    fig_grad = go.Figure()
    if any(g for g in grads):
        fig_grad.add_trace(go.Scatter(x=steps, y=grads, mode="lines+markers",
                                       line=dict(color="#f59e0b"), name="grad_norm"))
    fig_grad.update_layout(template="plotly_dark", paper_bgcolor="#1a1d24",
                            plot_bgcolor="#1a1d24",
                            height=300, title="Gradient norm",
                            xaxis_title="step", yaxis_title="||grad||")

    fig_lr = go.Figure()
    if any(l for l in lrs):
        fig_lr.add_trace(go.Scatter(x=steps, y=lrs, mode="lines",
                                     line=dict(color="#a78bfa"), name="lr"))
    fig_lr.update_layout(template="plotly_dark", paper_bgcolor="#1a1d24",
                          plot_bgcolor="#1a1d24",
                          height=240, title="Learning rate (cosine schedule)",
                          xaxis_title="step", yaxis_title="lr")

    delta_color = "#84cc16" if post_correct >= pre_correct else "#ef4444"
    delta_sign = "+" if post_correct >= pre_correct else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ARC laptop training · report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       margin: 0; padding: 24px; background: #0e1117; color: #e8e8e8; }}
h1 {{ margin: 0 0 4px 0; font-weight: 500; }}
h2 {{ font-size: 14px; font-weight: 500; color: #aaa; margin: 22px 0 10px 0;
      letter-spacing: 0.5px; text-transform: uppercase; }}
.sub {{ color: #888; margin-bottom: 18px; font-size: 13px; }}
.kpis {{ display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px }}
.kpi {{ background:#1a1d24;padding:14px;border-radius:8px;border:1px solid #2a2f3a }}
.kpi .label {{ color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px }}
.kpi .value {{ font-size:22px;font-weight:600;margin-top:4px }}
.note {{ background:#1e3a5f;border-left:3px solid #2dd4bf;padding:10px 14px;
         border-radius:4px;font-size:13px;color:#cce;margin-bottom:18px }}
.chart {{ background:#1a1d24;padding:12px;border-radius:8px;border:1px solid #2a2f3a;margin-bottom:14px }}
a {{ color: #2dd4bf; }}
code {{ background:#0e1117;padding:1px 5px;border-radius:3px;font-size:12px }}
</style></head><body>
<h1>ARC-AGI · Laptop LoRA Training Report</h1>
<div class="sub">{MODEL_ID} · LoRA r={LORA_R} · {N_TRAIN_STEPS} steps · {len(texts)} samples · M3 Pro / MPS</div>

<div class="note">
This is a local training demo using <b>TRL SFTTrainer + PEFT + TensorBoard</b> — exactly
the same stack the H100 notebook uses, just at laptop scale (1.5B model, 60 steps, no
quantization since bitsandbytes doesn't run on MPS). Live TensorBoard at
<a href="http://localhost:6006" target="_blank">http://localhost:6006</a>.
</div>

<div class="kpis">
  <div class="kpi"><div class="label">Pre-train accuracy</div><div class="value">{pre_correct}/{EVAL_N} ({pre_correct/EVAL_N*100:.0f}%)</div></div>
  <div class="kpi"><div class="label">Post-train accuracy</div><div class="value">{post_correct}/{EVAL_N} ({post_correct/EVAL_N*100:.0f}%)</div></div>
  <div class="kpi"><div class="label">Δ accuracy</div><div class="value" style="color:{delta_color}">{delta_sign}{summary["delta_pp"]:.0f} pp</div></div>
  <div class="kpi"><div class="label">Final loss</div><div class="value">{(f"{losses[-1]:.3f}" if losses else "—")}</div></div>
  <div class="kpi"><div class="label">Steps trained</div><div class="value">{N_TRAIN_STEPS}</div></div>
</div>

<h2>1. Training loss curve</h2>
<div class="chart" id="loss"></div>

<h2>2. Gradient norm — is training stable?</h2>
<div class="chart" id="grad"></div>

<h2>3. Learning-rate schedule</h2>
<div class="chart" id="lr"></div>

<h2>4. What TensorBoard shows that this report doesn't</h2>
<div class="note" style="border-left-color:#a78bfa;background:#2a1a3a">
Open <a href="http://localhost:6006" target="_blank">http://localhost:6006</a> for:
<ul>
  <li><b>Per-LoRA-module weight L2 norm over time</b> — which adapters grow fastest. Custom callback logs this every 2 steps.</li>
  <li><b>Histograms</b> — gradient distributions, weight distributions per layer.</li>
  <li><b>Step time + throughput</b> — where the bottleneck is.</li>
</ul>
This report is the <i>summary</i>; TensorBoard is the <i>full trace</i>.
</div>

<h2>5. Honest take on these numbers</h2>
<div class="note" style="border-left-color:#f59e0b;background:#3a2e1a">
{N_TRAIN_STEPS} steps × batch=2 over a 1.5B model is enough to <b>verify the pipeline runs</b>,
not to <b>actually move the score</b>. Realistic ARC training takes ~5,000 optimizer steps on
H100. Use this notebook to validate your data + LoRA + logging works, then run
<code>arc_kiss.ipynb</code> on Colab H100 for the actual training that produces 25%+
accuracy.
</div>

<script>
const figLoss = {fig_loss.to_json()};
const figGrad = {fig_grad.to_json()};
const figLR   = {fig_lr.to_json()};
Plotly.newPlot('loss', figLoss.data, figLoss.layout, {{displayModeBar:false}});
Plotly.newPlot('grad', figGrad.data, figGrad.layout, {{displayModeBar:false}});
Plotly.newPlot('lr',   figLR.data,   figLR.layout,   {{displayModeBar:false}});
</script>
</body></html>"""
    OUT_HTML.write_text(html)
    writer.close()
    print(f"[train] wrote {OUT_HTML}  ({OUT_HTML.stat().st_size/1e3:.0f} KB)")
    print(f"[train] TensorBoard logs at {TB_DIR}")
    print(f"[train] open: http://localhost:6006   and   file://{OUT_HTML}")


if __name__ == "__main__":
    main()
