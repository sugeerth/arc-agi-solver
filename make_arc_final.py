"""
Build /Users/fullfocus/arc-agi/arc_final.ipynb — the single Colab notebook.

What's in it (top to bottom):
  1-11. Inference pipeline: load Qwen2.5-7B 4-bit, eval on full ARC-AGI public set,
        greedy + D4 self-consistency vote, write submission.json. Same as arc_5pct
        but with per-task timings tracked.
  12-15. INLINE plotly dashboard: KPI banner, accuracy bar chart, failure-mode
         breakdown, per-task timing scatter, rendered grids of correct + failed
         tasks (so you literally SEE what the model got right and wrong).

Run All on Colab GPU. Everything renders in the notebook itself + bundle.zip
downloads.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "arc_final.ipynb"


def md(t): return ("md", t)
def code(t): return ("code", t)


def nb(cells):
    out = []
    for kind, source in cells:
        src = source if isinstance(source, list) else [source]
        norm = [(line + "\n" if (not line.endswith("\n") and i != len(src) - 1) else line)
                for i, line in enumerate(src)]
        out.append({"cell_type": "markdown" if kind == "md" else "code",
                    "metadata": {}, "source": norm,
                    **({"execution_count": None, "outputs": []} if kind == "code" else {})})
    return {"cells": out,
            "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info": {"name":"python","version":"3.10"}},
            "nbformat": 4, "nbformat_minor": 5}


CELLS = [
    md("""# ARC-AGI · Final Single-Notebook Pipeline + Inline Dashboard

**Goal:** clear 5% on ARC-AGI-1 public eval **and** see the full picture inline — no separate dashboard, no zip, just open the notebook after Run All and everything is rendered.

### Strategy (no training)

1. Load **Qwen2.5-7B-Instruct** in 4-bit (fits any Colab GPU, T4 16 GB or up).
2. Greedy decoding on all 400 ARC-AGI public eval tasks.
3. **D4 self-consistency voting** — 8 dihedral symmetries, inverse-transform, majority-vote.
4. pass@2 = greedy OR D4 correct.

### What gets rendered inline

- Headline KPI banner with the three accuracies + a "5% cleared" badge
- Accuracy bar chart with the 5% target line
- **Failure-mode breakdown** (correct / wrong shape / wrong colors / unparseable)
- Per-task timing scatter (color-coded by correctness)
- **Sample correct tasks** with input + truth grids rendered as colored cells
- **Sample failed tasks** with input + truth + prediction rendered side-by-side

### Expected

Greedy ~4%, D4 ~9–12%, pass@2 ~10–14%. Wall time ~30–60 min on T4, ~10 min on A100/H100.

Just `Runtime → Run all`.
"""),
    code("""# 1 — Hardware + install
import torch, time, json, os
from pathlib import Path
assert torch.cuda.is_available(), "Set Runtime → GPU."
GPU = torch.cuda.get_device_name(0)
VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU} ({VRAM:.0f} GB)")
"""),
    code("""!pip -q install -U \\
    "transformers>=4.46" "accelerate>=1.1" "bitsandbytes>=0.44" \\
    "plotly>=5.20" sentencepiece huggingface_hub
"""),
    code("""# 2 — Run config
MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
LIMIT       = None             # None = full 400 eval; set 100 for faster smoke
RUN_GREEDY  = True
RUN_D4_VOTE = True
MAX_NEW     = 1024

ROOT    = Path("/content") if Path("/content").exists() else Path(".")
RUN_ID  = f"arcfinal-{int(time.time())}"
RUN_DIR = ROOT / "runs" / RUN_ID; RUN_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "arc_data"; DATA_DIR.mkdir(exist_ok=True)
print("RUN_DIR:", RUN_DIR)
"""),
    code("""# 3 — Get ARC-AGI-1 eval set
import urllib.request, zipfile, io
if not (DATA_DIR / "evaluation").exists():
    print("Downloading ARC-AGI-1 ...")
    with urllib.request.urlopen("https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip") as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    src = DATA_DIR / "ARC-AGI-master" / "data"
    (DATA_DIR / "evaluation").mkdir(exist_ok=True)
    for f in (src / "evaluation").glob("*.json"):
        (DATA_DIR / "evaluation" / f.name).write_bytes(f.read_bytes())

arc_eval = [(p.stem, json.loads(p.read_text()))
            for p in sorted((DATA_DIR/"evaluation").glob("*.json"))]
EVAL_SUBSET = arc_eval[:LIMIT] if LIMIT else arc_eval
print(f"Eval tasks: {len(EVAL_SUBSET)}")
"""),
    code("""# 4 — Load Qwen2.5-7B in 4-bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True)
attn_impl = "flash_attention_2" if ("H100" in GPU or "A100" in GPU) else "sdpa"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map={"":0},
    torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
model.eval()
print("Model loaded.")
"""),
    code("""# 5 — Format / parse / D4 helpers
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
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

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
"""),
    code("""# 6 — Inference helpers (with timing)
@torch.inference_mode()
def predict_greedy(task, max_new=MAX_NEW):
    text = to_chat_prompt(task)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return parse_grid(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))

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
"""),
    code("""# 7 — Greedy pass with per-task timing
from tqdm.auto import tqdm
greedy_preds = {}; greedy_times = {}; g_correct = 0

if RUN_GREEDY:
    t0 = time.time()
    for name, task in tqdm(EVAL_SUBSET, desc="greedy"):
        truth = task["test"][0]["output"]
        ts = time.time()
        pred = predict_greedy(task)
        greedy_times[name] = time.time() - ts
        greedy_preds[name] = pred
        if pred == truth:
            g_correct += 1
    print(f"\\nGREEDY: {g_correct}/{len(EVAL_SUBSET)} = {g_correct/len(EVAL_SUBSET):.1%}  ({time.time()-t0:.0f}s)")
"""),
    code("""# 8 — D4 vote pass with per-task timing
d4_preds = {}; d4_times = {}; d_correct = 0

if RUN_D4_VOTE:
    t0 = time.time()
    for name, task in tqdm(EVAL_SUBSET, desc="d4 vote"):
        truth = task["test"][0]["output"]
        ts = time.time()
        pred = predict_d4_vote(task)
        d4_times[name] = time.time() - ts
        d4_preds[name] = pred
        if pred == truth:
            d_correct += 1
    print(f"\\nD4 VOTE: {d_correct}/{len(EVAL_SUBSET)} = {d_correct/len(EVAL_SUBSET):.1%}  ({time.time()-t0:.0f}s)")
else:
    d4_preds = dict(greedy_preds); d4_times = dict(greedy_times); d_correct = g_correct
"""),
    code("""# 9 — pass@2 + scores.json + submission.json
n = len(EVAL_SUBSET)
pass2 = sum(1 for name, t in EVAL_SUBSET
            if (greedy_preds.get(name) == t["test"][0]["output"])
               or (d4_preds.get(name) == t["test"][0]["output"]))

submission = {}
for name, task in EVAL_SUBSET:
    a1 = d4_preds.get(name) or [[0]]
    a2 = greedy_preds.get(name) or [[0]]
    submission[name] = [{"attempt_1": a1, "attempt_2": a2}]
(RUN_DIR / "submission.json").write_text(json.dumps(submission))

best_acc = max(g_correct, d_correct, pass2) / n
scores = {"model": MODEL_ID, "run_id": RUN_ID, "n": n,
          "greedy_correct": g_correct, "greedy_acc": g_correct/n,
          "d4_correct": d_correct, "d4_acc": d_correct/n,
          "pass2_correct": pass2, "pass2_acc": pass2/n,
          "best_acc": best_acc, "cleared_5pct": best_acc >= 0.05}
(RUN_DIR / "scores.json").write_text(json.dumps(scores, indent=2))

print(f"\\nGREEDY: {g_correct}/{n} = {g_correct/n:.1%}")
print(f"D4    : {d_correct}/{n} = {d_correct/n:.1%}")
print(f"PASS@2: {pass2}/{n} = {pass2/n:.1%}")
print(f"\\n{'✅ CLEARED 5%' if best_acc >= 0.05 else '❌ Below 5%'} — best score {best_acc:.1%}")
"""),
    md("""---

# 📊 Inline Dashboard

Everything below renders right here in the notebook. No download required.
"""),
    code("""# 12 — Headline KPI banner + accuracy bar chart
import plotly.graph_objects as go
from IPython.display import HTML, display

target_color = "#84cc16" if best_acc >= 0.05 else "#ef4444"
target_text  = "✅ CLEARED 5%" if best_acc >= 0.05 else "❌ Below 5%"

display(HTML(f'''
<style>
.arc-kpis {{ display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:14px 0; }}
.arc-kpi {{ background:#1a1d24;padding:14px 16px;border-radius:8px;border:1px solid #2a2f3a;color:#e8e8e8; }}
.arc-kpi .lbl {{ color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px; }}
.arc-kpi .val {{ font-size:22px;font-weight:600;margin-top:4px; }}
</style>
<div class="arc-kpis">
  <div class="arc-kpi"><div class="lbl">Tasks</div><div class="val">{n}</div></div>
  <div class="arc-kpi"><div class="lbl">Greedy</div><div class="val">{g_correct/n:.1%}</div></div>
  <div class="arc-kpi"><div class="lbl">D4 Vote</div><div class="val">{d_correct/n:.1%}</div></div>
  <div class="arc-kpi"><div class="lbl">pass@2</div><div class="val">{pass2/n:.1%}</div></div>
  <div class="arc-kpi" style="background:{target_color};color:#000"><div class="lbl" style="color:#000">Target</div><div class="val">{target_text}</div></div>
</div>
'''))

methods = ["Greedy", "D4 Vote", "pass@2"]
accs = [g_correct/n*100, d_correct/n*100, pass2/n*100]
fig = go.Figure(go.Bar(x=methods, y=accs,
                       marker_color=["#2dd4bf","#f59e0b","#84cc16"],
                       text=[f"{a:.1f}%" for a in accs], textposition="outside",
                       textfont=dict(size=14)))
fig.add_hline(y=5, line=dict(color="red", dash="dash", width=2),
              annotation_text="5% target", annotation_position="top right",
              annotation_font_color="red")
fig.update_layout(title=f"ARC-AGI public eval — {MODEL_ID}",
                  yaxis_title="Accuracy (%)", template="plotly_dark",
                  paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                  height=380, showlegend=False)
fig.show()
"""),
    code("""# 13 — Failure-mode breakdown
def classify(pred, truth):
    if pred is None: return "unparseable"
    if pred == truth: return "correct"
    if len(pred) != len(truth) or len(pred[0]) != len(truth[0]): return "wrong_shape"
    same = sum(1 for i in range(len(truth)) for j in range(len(truth[0])) if pred[i][j] == truth[i][j])
    pct = same / max(1, len(truth)*len(truth[0]))
    if pct >= 0.85: return "wrong_few_cells"
    if pct >= 0.5:  return "wrong_partial"
    return "wrong_many_cells"

def best_pred(name, truth):
    for src in (d4_preds, greedy_preds):
        if src.get(name) == truth: return src[name]
    return d4_preds.get(name) or greedy_preds.get(name)

cats = {"correct":0, "wrong_shape":0, "wrong_few_cells":0,
        "wrong_partial":0, "wrong_many_cells":0, "unparseable":0}
for name, t in EVAL_SUBSET:
    truth = t["test"][0]["output"]
    cats[classify(best_pred(name, truth), truth)] += 1

fig2 = go.Figure(go.Bar(
    x=list(cats.values()),
    y=[k.replace("_"," ") for k in cats],
    orientation="h",
    marker_color=["#84cc16","#f59e0b","#ef4444","#dc2626","#991b1b","#6b7280"],
    text=[str(v) for v in cats.values()], textposition="outside"))
fig2.update_layout(title="Failure-mode breakdown (best of greedy/D4)",
                   xaxis_title="task count", template="plotly_dark",
                   paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                   height=350)
fig2.show()
"""),
    code("""# 14 — Per-task timing scatter (color = correct/wrong)
xs = list(range(len(EVAL_SUBSET)))
ys_g = [greedy_times.get(name, 0) for name, _ in EVAL_SUBSET]
ys_d = [d4_times.get(name, 0)     for name, _ in EVAL_SUBSET]
g_ok = [(greedy_preds.get(name) == t["test"][0]["output"]) for name, t in EVAL_SUBSET]
d_ok = [(d4_preds.get(name)     == t["test"][0]["output"]) for name, t in EVAL_SUBSET]
hovers = [name for name, _ in EVAL_SUBSET]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=xs, y=ys_g, mode="markers",
                          marker=dict(size=8, color=["#84cc16" if ok else "#ef4444" for ok in g_ok],
                                     symbol="circle"),
                          name="greedy", hovertext=hovers, hoverinfo="text+y"))
fig3.add_trace(go.Scatter(x=xs, y=ys_d, mode="markers",
                          marker=dict(size=8, color=["#84cc16" if ok else "#ef4444" for ok in d_ok],
                                     symbol="diamond"),
                          name="D4 vote", hovertext=hovers, hoverinfo="text+y"))
fig3.update_layout(title="Per-task time (sec) — green=correct, red=wrong",
                   xaxis_title="task index", yaxis_title="seconds",
                   template="plotly_dark", paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                   height=380)
fig3.show()
"""),
    code("""# 15 — Sample correct + failed tasks rendered as colored grids
ARC_COLORS = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00",
              "#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25"]

def grid_to_svg(grid, cell=14, max_size=180):
    if not grid: return '<div style="color:#888;font-size:11px">[unparseable]</div>'
    h, w = len(grid), len(grid[0])
    cell = max(4, min(cell, max_size // max(h, w)))
    rects = []
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            color = ARC_COLORS[c] if 0 <= c <= 9 else "#444"
            rects.append(f'<rect x="{j*cell}" y="{i*cell}" width="{cell}" height="{cell}" '
                        f'fill="{color}" stroke="#1a1d24" stroke-width="0.5"/>')
    return (f'<svg width="{w*cell}" height="{h*cell}" style="background:#000;border-radius:3px">'
            f'{"".join(rects)}</svg>'
            f'<div style="font-size:10px;color:#888;margin-top:2px">{h}×{w}</div>')

correct = []; failed = []
for name, t in EVAL_SUBSET:
    truth = t["test"][0]["output"]
    pred = best_pred(name, truth)
    if pred == truth: correct.append((name, t, pred))
    else: failed.append((name, t, pred))

def render_block(title, items, show_pred=True):
    blocks = [f'<h3 style="color:#e8e8e8;margin:18px 0 8px 0">{title}</h3>']
    for name, task, pred in items:
        truth = task["test"][0]["output"]
        test_in = task["test"][0]["input"]
        cells = [
            f'<div style="min-width:80px;font-family:monospace;font-size:12px;color:#888">{name}</div>',
            f'<div><div style="font-size:10px;color:#888">test input</div>{grid_to_svg(test_in)}</div>',
            f'<div><div style="font-size:10px;color:#888">truth</div>{grid_to_svg(truth)}</div>',
        ]
        if show_pred:
            cells.append(f'<div><div style="font-size:10px;color:#888">prediction</div>{grid_to_svg(pred)}</div>')
        blocks.append(
            '<div style="background:#1a1d24;padding:10px;border-radius:6px;'
            'border:1px solid #2a2f3a;margin-bottom:8px;'
            'display:flex;gap:14px;align-items:center;flex-wrap:wrap">'
            + "".join(cells) + '</div>')
    return "".join(blocks)

display(HTML(render_block(f"✅ Correctly solved ({len(correct)} total · showing first 8)",
                          correct[:8], show_pred=False)))
display(HTML(render_block(f"❌ Failed ({len(failed)} total · showing first 6 with prediction)",
                          failed[:6], show_pred=True)))
"""),
    code("""# 16 — Bundle + Colab download
import shutil
zpath = ROOT / f"{RUN_ID}.zip"
shutil.make_archive(str(zpath).removesuffix(".zip"), "zip", RUN_DIR)
print(f"Bundle: {zpath} ({zpath.stat().st_size/1e6:.2f} MB)")
print(f"Contents: scores.json, submission.json")
try:
    from google.colab import files
    files.download(str(zpath))
except Exception:
    print(f"Not on Colab — pull from {zpath}")
"""),
    md("""---

## What's next

If you cleared 5%:
- **Submit `submission.json`** to the ARC Prize Kaggle competition for the official leaderboard score.
- For top-100 (~25–35%): switch to `arc_kiss.ipynb` on Colab Pro+ H100 to actually train. ~5–6 h, ~$5–15. Realistic 15–35%.
- For top-50 (~45%+): add test-time training (`04_eval_ttt.ipynb` with `RUN_TTT=True`).

If you didn't clear 5%:
- Bump `LIMIT` to `None` if you set a small number — small samples are noisy.
- Confirm cell 4 finished without errors.
- Verify the model id is `Qwen/Qwen2.5-7B-Instruct` and not a smaller variant.

The bundle zip contains everything reproducible (scores + submission). The inline dashboard above gives you the picture without needing to download anything.
"""),
]


if __name__ == "__main__":
    OUT.write_text(json.dumps(nb(CELLS), indent=1))
    print(f"Wrote {OUT}  ({len(CELLS)} cells)")
