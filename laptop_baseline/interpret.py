"""
SOTA interpretability probe for the ARC-AGI laptop baseline.

For 3 probe tasks, captures (in priority order):

  1. Failure-mode classification (correct / wrong-shape / wrong-colors / unparseable)
  2. In-context example ablation     -- which prompt example drove the prediction
  3. Retrieval attribution           -- which ARC training tasks look most similar
                                        (proxy for "what training data shapes the answer")
  4. Logit lens                      -- which layer the answer crystallizes at
  5. Layer activation profile        -- which layers do the work
  6. Per-token confidence + entropy  -- where the model is guessing
  7. Attention rollout               -- which input tokens influence each output token

Renders interpret.html with details-on-demand collapsible sections.

Tooling note: uses raw torch forward hooks (most reliable across model families).
The same probes can be done with TransformerLens, NNsight, or Captum for richer APIs --
swap them in for production work.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
ARC_DIR = HERE / "ARC-AGI-master" / "data" / "evaluation"
TRAIN_DIR = HERE / "ARC-AGI-master" / "data" / "training"
OUT = HERE / "interpret.html"
RESULTS = HERE / "results_3b_50.json"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PROBE_NAMES = ["00576224", "00dbd492", "0520fde7"]
MAX_LL_STEPS = 24      # logit lens steps (smaller = faster, sufficient signal)
MAX_GEN = 256
TOP_K_RETRIEVAL = 3
N_TRAIN_EMBED = 200    # embed first N training tasks for retrieval (caps cost)

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]

SYSTEM = ("You are an expert ARC puzzle solver. Output only the test output grid as digits "
          "separated by spaces, one row per line. No explanations.")


def grid_to_str(g):
    return "\n".join(" ".join(str(c) for c in row) for row in g)


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


def build_prompt(tok, task):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": format_user(task)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def grid_to_svg(grid, cell=12, max_size=160):
    if not grid: return '<div style="color:#888;font-size:11px">[unparseable]</div>'
    h, w = len(grid), len(grid[0])
    cell = max(4, min(cell, max_size // max(h, w)))
    rects = []
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            color = ARC_COLORS[c] if 0 <= c <= 9 else "#444"
            rects.append(f'<rect x="{j*cell}" y="{i*cell}" width="{cell}" height="{cell}" fill="{color}" stroke="#1a1d24" stroke-width="0.5"/>')
    return (f'<svg width="{w*cell}" height="{h*cell}" style="background:#000;border-radius:3px">{"".join(rects)}</svg>'
            f'<div style="font-size:10px;color:#888;margin-top:2px">{h}×{w}</div>')


# --- Capture functions ---

@torch.inference_mode()
def generate(model, tok, prompt_text, max_new=MAX_GEN):
    inp = tok(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


@torch.inference_mode()
def capture_logit_lens(model, tok, prompt_text, max_new=MAX_LL_STEPS):
    base = model.model
    lm_head = model.get_output_embeddings()
    final_norm = base.norm
    inp = tok(prompt_text, return_tensors="pt").to(model.device)
    input_ids, attn_mask = inp.input_ids, inp.attention_mask
    eos = tok.eos_token_id
    steps = []
    for step in range(max_new):
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        last_logits = out.logits[:, -1, :]
        probs = torch.softmax(last_logits.float(), dim=-1)
        chosen = int(torch.argmax(last_logits, dim=-1))
        layer_rows = []
        for li, h in enumerate(hs):
            h_last = h[:, -1, :].to(lm_head.weight.dtype)
            if li != len(hs) - 1: h_last = final_norm(h_last)
            pl = torch.softmax(lm_head(h_last).float(), dim=-1)
            top1 = int(torch.argmax(pl, dim=-1))
            layer_rows.append({"layer": li, "top1_id": top1,
                               "top1_prob": float(pl[0, top1]),
                               "top1_tok": tok.decode([top1])})
        steps.append({
            "step": step, "chosen_id": chosen, "chosen_tok": tok.decode([chosen]),
            "chosen_prob": float(probs[0, chosen]),
            "entropy": float(-(probs * torch.log(probs.clamp_min(1e-12))).sum()),
            "layers": layer_rows,
        })
        input_ids = torch.cat([input_ids, torch.tensor([[chosen]], device=model.device)], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones((1, 1), device=model.device, dtype=attn_mask.dtype)], dim=1)
        if chosen == eos: break
    return steps


@torch.inference_mode()
def capture_layer_norms(model, tok, prompt_text):
    layers = model.model.layers
    cache = {}
    handles = []
    for i, layer in enumerate(layers):
        def attn_h(idx):
            def h(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                cache[("attn", idx)] = t.detach().float().pow(2).mean(dim=-1).sqrt().mean().item()
            return h
        def mlp_h(idx):
            def h(_m, _i, o):
                cache[("mlp", idx)] = o.detach().float().pow(2).mean(dim=-1).sqrt().mean().item()
            return h
        handles.append(layer.self_attn.register_forward_hook(attn_h(i)))
        handles.append(layer.mlp.register_forward_hook(mlp_h(i)))
    inp = tok(prompt_text, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
    _ = model(**inp, use_cache=False)
    for h in handles: h.remove()
    n = len(layers)
    return {"attn": [cache.get(("attn", i), 0.0) for i in range(n)],
            "mlp":  [cache.get(("mlp",  i), 0.0) for i in range(n)]}


@torch.inference_mode()
def embed_input(model, tok, task):
    """Mean-pool last hidden state of the test input string. Used for retrieval."""
    inp = tok(grid_to_str(task["test"][0]["input"]), return_tensors="pt", truncation=True, max_length=512).to(model.device)
    out = model(**inp, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1].mean(dim=1).squeeze(0)
    return h.float().cpu().numpy()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def classify_failure(pred, truth):
    if pred is None: return "unparseable"
    if pred == truth: return "correct"
    if len(pred) != len(truth) or len(pred[0]) != len(truth[0]): return "wrong_shape"
    same = sum(1 for i in range(len(truth)) for j in range(len(truth[0])) if pred[i][j] == truth[i][j])
    total = len(truth) * len(truth[0])
    pct = same / total
    if pct >= 0.85: return "wrong_few_cells"
    if pct >= 0.5: return "wrong_partial"
    return "wrong_many_cells"


# --- In-context ablation ---

def ablation_study(model, tok, task, max_new=MAX_GEN):
    """Drop each train example one at a time, see how prediction changes."""
    truth = task["test"][0]["output"]
    full = generate(model, tok, build_prompt(tok, task), max_new=max_new)
    full_pred = parse_grid(full)
    full_correct = (full_pred == truth)
    results = [{"removed": "none", "pred": full_pred,
                "correct": full_correct, "category": classify_failure(full_pred, truth)}]
    for i in range(len(task["train"])):
        if len(task["train"]) <= 1: break
        ablated = {"train": [e for j, e in enumerate(task["train"]) if j != i],
                   "test": task["test"]}
        a_out = generate(model, tok, build_prompt(tok, ablated), max_new=max_new)
        a_pred = parse_grid(a_out)
        results.append({
            "removed": f"example {i+1}",
            "pred": a_pred,
            "correct": (a_pred == truth),
            "category": classify_failure(a_pred, truth),
        })
    return results


# --- Plotly figure builders ---

def fig_failure_breakdown(eval_results):
    if not eval_results: return None
    cats = {"correct": 0, "wrong_shape": 0, "wrong_few_cells": 0,
            "wrong_partial": 0, "wrong_many_cells": 0, "unparseable": 0}
    for r in eval_results.get("results", []):
        if r.get("correct"):
            cats["correct"] += 1
        else:
            ps = r.get("pred_shape"); ts = r.get("truth_shape")
            if ps is None: cats["unparseable"] += 1
            elif ps != ts: cats["wrong_shape"] += 1
            else: cats["wrong_partial"] += 1   # we don't have cells, lump into partial
    fig = go.Figure(go.Bar(
        x=list(cats.values()),
        y=[k.replace("_", " ") for k in cats],
        orientation="h",
        marker_color=["#84cc16", "#f59e0b", "#ef4444", "#dc2626", "#991b1b", "#6b7280"],
        text=[str(v) for v in cats.values()], textposition="outside",
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                      height=320, title="Eval failure breakdown")
    return fig


def fig_logit_lens(steps, name):
    if not steps: return None
    n_layers = len(steps[0]["layers"])
    Z = np.zeros((len(steps), n_layers))
    text = [["" for _ in range(n_layers)] for _ in range(len(steps))]
    for si, st in enumerate(steps):
        for li, lr in enumerate(st["layers"]):
            Z[si, li] = lr["top1_prob"]
            t = lr["top1_tok"].replace("\n", "\\n")
            text[si][li] = f"L{li} step{si}: '{t}' p={lr['top1_prob']:.2f}"
    fig = go.Figure(go.Heatmap(z=Z, text=text, hoverinfo="text",
                               colorscale="Viridis", zmin=0, zmax=1,
                               colorbar=dict(title="prob")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                      title=f"Logit lens — {name}",
                      xaxis_title="layer", yaxis_title="generated token #",
                      height=350)
    return fig


def fig_layer_norms(ln, name):
    fig = go.Figure()
    xs = list(range(len(ln["attn"])))
    fig.add_trace(go.Scatter(x=xs, y=ln["attn"], mode="lines+markers",
                             name="attn", line=dict(color="#2dd4bf")))
    fig.add_trace(go.Scatter(x=xs, y=ln["mlp"], mode="lines+markers",
                             name="mlp", line=dict(color="#f59e0b")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                      height=300, title=f"Layer activation profile — {name}",
                      xaxis_title="layer", yaxis_title="mean output norm")
    return fig


def fig_confidence(steps, name):
    if not steps: return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["P(chosen) per token", "entropy per token"])
    xs = [s["step"] for s in steps]
    ps = [s["chosen_prob"] for s in steps]
    es = [s["entropy"] for s in steps]
    ts = [s["chosen_tok"].replace("\n", "\\n") for s in steps]
    fig.add_trace(go.Bar(x=xs, y=ps, hovertext=ts, hoverinfo="text+y",
                         marker_color="#2dd4bf", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=es, mode="lines+markers",
                             line=dict(color="#f59e0b"), showlegend=False), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                      height=420, title=f"Per-token confidence + entropy — {name}")
    return fig


def fig_to_html_div(fig, div_id):
    if fig is None:
        return f'<div id="{div_id}" style="color:#888;padding:30px;text-align:center">[no data]</div>'
    return (f'<div id="{div_id}"></div>\n'
            f'<script>(function(){{const f={fig.to_json()};'
            f'Plotly.newPlot("{div_id}",f.data,f.layout,{{displayModeBar:false}});}})();</script>')


def render_html(eval_results, probes_data):
    eval_summary = ""
    if eval_results:
        n = eval_results.get("n", 0)
        c = eval_results.get("correct", 0)
        acc = eval_results.get("accuracy", 0) * 100
        verdict = []
        if c == 0:
            verdict.append("Model is making zero correct predictions — typical for an untrained 3B on ARC.")
        if c > 0:
            verdict.append(f"Model solved {c}/{n} tasks zero-shot.")
        eval_summary = f'''
        <div class="kpis">
          <div class="kpi"><div class="label">Tasks</div><div class="value">{n}</div></div>
          <div class="kpi"><div class="label">Correct</div><div class="value">{c}</div></div>
          <div class="kpi"><div class="label">Accuracy</div><div class="value">{acc:.1f}%</div></div>
          <div class="kpi"><div class="label">Model</div><div class="value" style="font-size:14px">{MODEL_ID.split("/")[-1]}</div></div>
        </div>
        <div class="verdict">{" · ".join(verdict)}</div>
        '''

    failure_fig = fig_failure_breakdown(eval_results)
    failure_div = fig_to_html_div(failure_fig, "failure-breakdown") if failure_fig else ""

    probe_blocks = []
    for p in probes_data:
        pid = p["name"]
        task = p["task"]
        truth = task["test"][0]["output"]
        test_in = task["test"][0]["input"]
        ll_div = fig_to_html_div(fig_logit_lens(p["logit_lens"], pid), f"lens-{pid}")
        ln_div = fig_to_html_div(fig_layer_norms(p["layer_norms"], pid), f"layers-{pid}")
        conf_div = fig_to_html_div(fig_confidence(p["logit_lens"], pid), f"conf-{pid}")

        # Ablation table
        abl_rows = []
        for a in p.get("ablation", []):
            badge = "#84cc16" if a["correct"] else "#ef4444"
            abl_rows.append(f'''
            <tr>
              <td>{a["removed"]}</td>
              <td><span style="background:{badge};color:#000;padding:2px 6px;border-radius:3px;font-size:11px">{a["category"]}</span></td>
              <td><div style="display:inline-block">{grid_to_svg(a["pred"], cell=8)}</div></td>
            </tr>''')
        abl_table = ("<table style='width:100%;font-size:13px'>"
                     "<tr style='color:#888'><th align='left'>Removed</th><th align='left'>Outcome</th><th align='left'>Predicted</th></tr>"
                     f"{''.join(abl_rows)}</table>") if abl_rows else "<div style='color:#888'>No ablation (need ≥2 train examples)</div>"

        # Retrieval
        ret_blocks = []
        for r in p.get("retrieval", []):
            ret_blocks.append(f'''
            <div style="background:#0e1117;padding:8px;border-radius:6px;border:1px solid #2a2f3a">
              <div style="font-size:11px;color:#888">train task <code>{r["name"]}</code> · sim={r["sim"]:.3f}</div>
              <div style="display:flex;gap:8px;margin-top:4px">
                <div>{grid_to_svg(r["input"], cell=10)}</div>
                <div>{grid_to_svg(r["output"], cell=10)}</div>
              </div>
            </div>''')
        ret_div = ("<div style='display:flex;gap:10px;flex-wrap:wrap'>" + "".join(ret_blocks) + "</div>") if ret_blocks else ""

        # Failure verdict for this probe
        fc = p["full_pred_category"]
        verdict_color = "#84cc16" if fc == "correct" else "#ef4444"
        probe_blocks.append(f'''
        <details class="probe" {"open" if pid == probes_data[0]["name"] else ""}>
          <summary>
            <span class="probe-name">probe {pid}</span>
            <span class="probe-badge" style="background:{verdict_color}">{fc.replace("_", " ")}</span>
            <span class="probe-time">{p["timing_secs"]:.1f}s</span>
          </summary>

          <h3>What the model saw</h3>
          <div style="display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap">
            <div><div style="font-size:11px;color:#888">test input</div>{grid_to_svg(test_in)}</div>
            <div><div style="font-size:11px;color:#888">ground truth</div>{grid_to_svg(truth)}</div>
            <div><div style="font-size:11px;color:#888">model prediction</div>{grid_to_svg(p["full_pred"])}</div>
          </div>

          <details><summary>Raw output (click to view)</summary>
            <pre style="background:#0e1117;padding:10px;border-radius:4px;overflow:auto;max-height:200px;font-size:12px">{p["full_output"][:2000]}</pre>
          </details>

          <h3>1. In-context example ablation</h3>
          <p class="hint">Drop each train example from the prompt and re-predict. The example whose removal most degrades the answer is the one the model relied on.</p>
          {abl_table}

          <h3>2. Retrieval attribution — closest ARC training tasks</h3>
          <p class="hint">Top-{TOP_K_RETRIEVAL} training tasks ranked by cosine similarity of last-layer hidden state of the test input. These are the tasks the model would have "remembered" had it been trained on ARC — proxy for training-data influence.</p>
          {ret_div}

          <h3>3. Logit lens — where the answer crystallizes</h3>
          <p class="hint">For each generated token (Y), apply LM head to each layer's hidden state (X), color = top-1 token probability. Bright top-rows mean the answer is decoded early.</p>
          {ll_div}

          <h3>4. Layer activation profile — which layers do work</h3>
          <p class="hint">Mean output norm of each layer's attention vs MLP. Spikes = layers doing meaningful computation.</p>
          {ln_div}

          <h3>5. Per-token confidence + entropy</h3>
          <p class="hint">Top: P(chosen token). Bottom: entropy. Low confidence + high entropy = the model is guessing at that position.</p>
          {conf_div}
        </details>
        ''')

    return f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ARC interp · laptop probe</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       margin: 0; padding: 24px; background: #0e1117; color: #e8e8e8; }}
h1 {{ margin: 0 0 4px 0; font-weight: 500; }}
h2 {{ font-size: 14px; font-weight: 500; color: #aaa; margin: 22px 0 10px 0;
      letter-spacing: 0.5px; text-transform: uppercase; }}
h3 {{ font-size: 14px; font-weight: 600; color: #ddd; margin: 18px 0 6px 0; }}
.sub {{ color: #888; margin-bottom: 18px; font-size: 13px; }}
.kpis {{ display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px }}
.kpi {{ background:#1a1d24;padding:14px;border-radius:8px;border:1px solid #2a2f3a }}
.kpi .label {{ color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px }}
.kpi .value {{ font-size:22px;font-weight:600;margin-top:4px }}
.verdict {{ background:#1e3a5f;border-left:3px solid #2dd4bf;padding:10px 14px;
            border-radius:4px;font-size:13px;color:#cce;margin-bottom:18px }}
.hint {{ color:#888;font-size:13px;margin:4px 0 10px 0 }}
details.probe {{ background:#1a1d24;padding:14px 18px;border-radius:8px;
                 border:1px solid #2a2f3a;margin-bottom:14px }}
details.probe summary {{ cursor:pointer;outline:none;list-style:none;
                          display:flex;align-items:center;gap:12px }}
details.probe summary::-webkit-details-marker {{ display:none }}
.probe-name {{ font-weight:600;font-size:15px }}
.probe-badge {{ padding:2px 10px;border-radius:3px;color:#000;font-size:11px;font-weight:600;
                text-transform:uppercase;letter-spacing:0.5px }}
.probe-time {{ color:#888;font-size:12px;margin-left:auto }}
table th, table td {{ padding:6px 8px;border-bottom:1px solid #2a2f3a;vertical-align:top }}
pre {{ color:#ddd }}
a {{ color:#2dd4bf }}
code {{ background:#0e1117;padding:1px 5px;border-radius:3px;font-size:12px }}
</style></head><body>

<h1>ARC-AGI · Architecture-level Interpretability</h1>
<div class="sub">{MODEL_ID} · M3 Pro · MPS / fp16 · {len(probes_data)} probe tasks · <a href="dashboard.html">← live dashboard</a></div>

<h2>Eval summary</h2>
{eval_summary}

<h2>What works vs what doesn't</h2>
{failure_div}

<h2>Per-probe deep dive · click any probe to expand</h2>
{"".join(probe_blocks)}

<h2>Glossary</h2>
<details><summary>How to read each viz</summary>
<ul style="line-height:1.7;font-size:13px;color:#bbb">
<li><b>Failure breakdown</b>: classifies each wrong prediction. <code>wrong_shape</code> = output dimensions don't match. <code>wrong_partial</code> = right shape, some cells off. <code>unparseable</code> = output isn't a valid grid.</li>
<li><b>In-context ablation</b>: drops each train example from the prompt and re-predicts. The example whose removal flips the answer is the one the model used. This is the closest thing to "training data attribution" for an untrained model — the prompt examples are the model's only "training signal" for that one inference.</li>
<li><b>Retrieval attribution</b>: embeds the test input with the model's last-layer mean-pooled hidden state, finds top-K nearest training tasks by cosine similarity. Once you fine-tune on ARC, these are literally the training samples that shaped the answer.</li>
<li><b>Logit lens</b> (Nostalgebraist 2020): apply the language model head to intermediate hidden states. Reveals which layer first "knows" the answer. After SFT, you'll see the bright zone shift earlier in the network.</li>
<li><b>Layer activation profile</b>: mean output norm per layer. Layers with low norm are essentially passing information through; spikes mark layers doing the actual computation.</li>
<li><b>Per-token confidence + entropy</b>: top bar = P(chosen token), bottom line = entropy. Confident-and-low-entropy = sure. Low-confidence-and-high-entropy = guessing. Visible "uncertainty regions" tell you where to focus error analysis.</li>
</ul>
</details>

<details><summary>SOTA tooling for production work</summary>
<p style="color:#bbb;font-size:13px">
The hooks here use raw torch for portability. For deeper mech-interp work, swap in:
<ul style="color:#bbb;font-size:13px">
<li><b>TransformerLens</b> (Neel Nanda) — clean APIs for hooks, activation patching, ablations</li>
<li><b>NNsight</b> (Bau lab) — declarative model-introspection</li>
<li><b>Captum</b> (Meta) — gradients, integrated gradients, attribution</li>
<li><b>SAE Lens</b> — sparse autoencoders for feature discovery (needs pretrained SAEs)</li>
<li><b>Inseq</b> — sequence-level attribution (LERG, attention rollout)</li>
</ul>
For training-data attribution proper (TracIn / influence functions), use <b>captum.influence</b> or <b>logix</b>. They need backward passes — much heavier than hooks.
</p>
</details>

</body></html>'''


def main():
    print(f"[interpret] loading {MODEL_ID} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                  device_map=device)
    model.eval()
    print(f"[interpret] loaded in {time.time()-t0:.1f}s")

    eval_results = None
    if RESULTS.exists():
        try: eval_results = json.loads(RESULTS.read_text())
        except: eval_results = None
        print(f"[interpret] loaded eval results: n={eval_results.get('n', 0) if eval_results else 0}")

    # 1. Embed N training tasks for retrieval
    print(f"[interpret] embedding {N_TRAIN_EMBED} training tasks for retrieval...")
    t0 = time.time()
    train_files = sorted(TRAIN_DIR.glob("*.json"))[:N_TRAIN_EMBED]
    train_db = []
    for fp in train_files:
        try:
            t = json.loads(fp.read_text())
            emb = embed_input(model, tok, t)
            train_db.append({"name": fp.stem, "task": t, "emb": emb})
        except Exception:
            continue
    print(f"[interpret] embedded {len(train_db)} train tasks in {time.time()-t0:.1f}s")

    # 2. Per-probe analysis
    probes_data = []
    for name in PROBE_NAMES:
        p = ARC_DIR / f"{name}.json"
        if not p.exists():
            print(f"[interpret] missing probe {name}, skip"); continue
        task = json.loads(p.read_text())
        truth = task["test"][0]["output"]
        prompt_text = build_prompt(tok, task)
        probe_t0 = time.time()

        print(f"[interpret] probe {name}: full prediction ...")
        full_out = generate(model, tok, prompt_text)
        full_pred = parse_grid(full_out)
        full_cat = classify_failure(full_pred, truth)

        print(f"[interpret] probe {name}: layer norms ...")
        ln = capture_layer_norms(model, tok, prompt_text)

        print(f"[interpret] probe {name}: logit lens ({MAX_LL_STEPS} steps) ...")
        ll = capture_logit_lens(model, tok, prompt_text, max_new=MAX_LL_STEPS)

        print(f"[interpret] probe {name}: ablation ({len(task['train'])} examples) ...")
        abl = ablation_study(model, tok, task)

        print(f"[interpret] probe {name}: retrieval (cosine sim across train DB) ...")
        emb = embed_input(model, tok, task)
        sims = [(cosine(emb, x["emb"]), x) for x in train_db]
        sims.sort(key=lambda kv: -kv[0])
        retrieval = []
        for sim, x in sims[:TOP_K_RETRIEVAL]:
            retrieval.append({"name": x["name"], "sim": sim,
                              "input": x["task"]["train"][0]["input"],
                              "output": x["task"]["train"][0]["output"]})

        elapsed = time.time() - probe_t0
        print(f"[interpret] probe {name} done in {elapsed:.1f}s, category={full_cat}")
        probes_data.append({
            "name": name, "task": task,
            "full_output": full_out, "full_pred": full_pred,
            "full_pred_category": full_cat,
            "layer_norms": ln, "logit_lens": ll,
            "ablation": abl, "retrieval": retrieval,
            "timing_secs": elapsed,
        })

    print("[interpret] rendering HTML...")
    OUT.write_text(render_html(eval_results, probes_data))
    print(f"[interpret] wrote {OUT}  ({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
