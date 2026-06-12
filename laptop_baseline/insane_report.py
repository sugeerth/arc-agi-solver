"""
Build an over-the-top, single-file HTML report covering everything the laptop
ARC-AGI pipeline produced: zero-shot baseline, LoRA fine-tune, training-time
internals (loss, grad-norm, LR, entropy, token-acc), per-LoRA-module weight
growth across all 28 layers, ARC eval-task previews rendered as actual colored
grids, methodology blurbs, hardware breakdown.

Reads:
  - runs/laptop_train/train_results.json
  - runs/laptop_train/tb/events.out.tfevents.*           (TensorBoard scalars)
  - results.json                                          (1.5B zero-shot baseline)
  - results_3b_50.json                                    (3B zero-shot baseline)
  - ARC-AGI-master/data/evaluation/*.json                 (task previews)

Writes: insane_report.html (self-contained, plotly via CDN).
"""
import json
import platform
import time
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
RUN = HERE / "runs" / "laptop_train"
TRAIN_RES = RUN / "train_results.json"
TB_DIR = RUN / "tb"
ARC_DIR = HERE / "ARC-AGI-master" / "data" / "evaluation"
OUT = HERE / "insane_report.html"

# ARC standard palette
ARC_COLORS = ["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00",
              "#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25"]


def load_tb(tb_dir):
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    series = {}
    for tag in ea.Tags()["scalars"]:
        evs = ea.Scalars(tag)
        series[tag] = ([e.step for e in evs], [e.value for e in evs])
    return series


def grid_to_svg(grid, cell=14, pad=1):
    """Render an ARC grid as an inline SVG."""
    h = len(grid); w = len(grid[0])
    W = w*cell + (w+1)*pad
    H = h*cell + (h+1)*pad
    parts = [f'<svg width="{W}" height="{H}" style="background:#222;border-radius:4px">']
    for r,row in enumerate(grid):
        for c,v in enumerate(row):
            x = pad + c*(cell+pad)
            y = pad + r*(cell+pad)
            color = ARC_COLORS[v] if 0 <= v < 10 else "#555"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def main():
    print(f"[report] reading {TRAIN_RES}")
    if not TRAIN_RES.exists():
        print(f"[report] no training results yet — run train_demo.py first"); return
    summary = json.loads(TRAIN_RES.read_text())

    print(f"[report] reading TensorBoard from {TB_DIR}")
    tb = load_tb(TB_DIR)

    # baselines
    base1_5b = json.loads((HERE / "results.json").read_text()) if (HERE / "results.json").exists() else None
    base3b = json.loads((HERE / "results_3b_50.json").read_text()) if (HERE / "results_3b_50.json").exists() else None

    # eval task previews
    eval_files = sorted(ARC_DIR.glob("*.json"))[: summary["eval_n"]]
    preview_html = []
    for fp in eval_files:
        t = json.loads(fp.read_text())
        train_n = len(t["train"])
        test_in = t["test"][0]["input"]
        test_out = t["test"][0].get("output")
        h = len(test_in); w = len(test_in[0])
        oh = len(test_out) if test_out else 0
        ow = len(test_out[0]) if test_out else 0
        block = (f'<div class="task">'
                 f'<div class="task-name">{fp.stem}</div>'
                 f'<div class="task-meta">{train_n} train examples · test {h}×{w} → {oh}×{ow}</div>'
                 f'<div class="task-grids">'
                 f'<div><div class="glabel">test in</div>{grid_to_svg(test_in)}</div>'
                 + (f'<div><div class="glabel">truth</div>{grid_to_svg(test_out)}</div>' if test_out else "")
                 + f'</div></div>')
        preview_html.append(block)

    # --- training metric series ---
    def s(tag): return tb.get(tag, ([],[]))
    loss_x, loss_y = s("train/loss")
    grad_x, grad_y = s("train/grad_norm")
    lr_x, lr_y = s("train/learning_rate")
    ent_x, ent_y = s("train/entropy")
    tok_x, tok_y = s("train/mean_token_accuracy")

    # --- LoRA norm aggregation: per layer, per proj, separating A/B ---
    lora_tags = [t for t in tb if t.startswith("lora_norm/")]
    # extract layer index and (proj, ab)
    lora_struct = defaultdict(dict)  # lora_struct[(layer, proj, ab)] = (xs, ys)
    for tag in lora_tags:
        # lora_norm/model.layers.N.self_attn.PROJ.lora_X
        try:
            rest = tag.split("lora_norm/model.layers.")[1]
            layer, rest = rest.split(".", 1)
            layer = int(layer)
            # rest: self_attn.PROJ.lora_X
            parts = rest.split(".")
            proj = parts[1]  # q_proj etc.
            ab = parts[2].split("_")[-1].upper()  # A or B
            lora_struct[(layer, proj, ab)] = s(tag)
        except Exception:
            continue

    layers = sorted({k[0] for k in lora_struct})
    projs = ["q_proj","k_proj","v_proj","o_proj"]

    # final-norm heatmap matrix per AB
    def final_norm(layer, proj, ab):
        xs, ys = lora_struct.get((layer, proj, ab), ([],[0]))
        return ys[-1] if ys else 0

    heat_A = [[final_norm(L, p, "A") for L in layers] for p in projs]
    heat_B = [[final_norm(L, p, "B") for L in layers] for p in projs]

    # Top 8 fastest-growing modules by final B norm (B starts at 0 in LoRA)
    top = sorted(lora_struct.items(),
                 key=lambda kv: (kv[1][1][-1] if kv[1][1] else 0) if kv[0][2]=="B" else -1,
                 reverse=True)
    top = [(k,v) for k,v in top if k[2]=="B"][:8]

    # KPIs
    delta_pp = summary["delta_pp"]
    delta_color = "#84cc16" if delta_pp >= 0 else "#ef4444"
    delta_sign = "+" if delta_pp >= 0 else ""
    final_loss = loss_y[-1] if loss_y else 0
    min_loss = min(loss_y) if loss_y else 0
    sys_str = f"{platform.system()} {platform.machine()} · Python {platform.python_version()}"
    n_train = summary["n_train_samples"]
    n_steps = summary["n_train_steps"]
    model_id = summary["model"]
    lora_r = summary["lora_r"]

    # comparison table rows
    cmp_rows = []
    if base1_5b:
        acc = base1_5b.get("accuracy", base1_5b.get("acc", 0))
        n = base1_5b.get("n", base1_5b.get("total", 0))
        sec = base1_5b.get("total_secs") or base1_5b.get("wall_secs") or 0
        cmp_rows.append(("Qwen2.5-1.5B zero-shot (raw grids)", f"{acc*100:.1f}%", f"{n}", f"{sec:.0f}s", "baseline.py"))
    if base3b:
        acc = base3b.get("accuracy", 0)
        n = base3b.get("n", 0)
        sec = base3b.get("total_secs", 0)
        cmp_rows.append(("Qwen2.5-3B zero-shot (raw grids)", f"{acc*100:.1f}%", f"{n}", f"{sec:.0f}s", "baseline.py --model …-3B"))
    cmp_rows.append((
        f"Qwen2.5-1.5B + LoRA r={lora_r} ({n_steps}-step SFT)",
        f"{summary['pre_acc']*100:.1f}% → {summary['post_acc']*100:.1f}% ({delta_sign}{delta_pp:.0f} pp)",
        f"{summary['eval_n']}", "—", "train_demo.py"))

    cmp_html = "\n".join(
        f"<tr><td>{a}</td><td><b>{b}</b></td><td>{c}</td><td>{d}</td><td><code>{e}</code></td></tr>"
        for a,b,c,d,e in cmp_rows
    )

    # -------- Build HTML --------
    plotly = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    head = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>ARC-AGI · Laptop Pipeline · Insane Report</title>""" + plotly + """
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',sans-serif;
     margin:0;padding:28px;background:#0a0c10;color:#e8e8e8;line-height:1.5;
     max-width:1400px;margin:0 auto}
h1{font-size:30px;font-weight:600;margin:0 0 4px 0;
   background:linear-gradient(90deg,#2dd4bf,#a78bfa);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sub{color:#8a94a6;margin-bottom:24px;font-size:13px}
h2{font-size:13px;font-weight:600;color:#94a3b8;margin:28px 0 12px 0;
   letter-spacing:1px;text-transform:uppercase;padding-bottom:6px;
   border-bottom:1px solid #1e2530}
h3{font-size:15px;font-weight:600;color:#cbd5e1;margin:18px 0 8px 0}
.hero{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px}
.kpi{background:linear-gradient(135deg,#161b25,#11151d);
     padding:16px;border-radius:10px;border:1px solid #232a36;position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
             background:linear-gradient(90deg,#2dd4bf,#a78bfa)}
.kpi .label{color:#7a8499;font-size:10px;text-transform:uppercase;letter-spacing:0.8px}
.kpi .value{font-size:22px;font-weight:700;margin-top:6px}
.kpi .sub2{color:#5e6a7e;font-size:11px;margin-top:2px}
.note{background:linear-gradient(135deg,#0e2030,#13283c);
      border-left:3px solid #2dd4bf;padding:12px 16px;border-radius:6px;
      font-size:13px;color:#cbd5e1;margin-bottom:18px}
.warn{border-left-color:#f59e0b;background:linear-gradient(135deg,#2d2010,#3a2814)}
.purp{border-left-color:#a78bfa;background:linear-gradient(135deg,#1f1530,#251a3a)}
.chart{background:#11151d;padding:14px;border-radius:10px;border:1px solid #232a36;margin-bottom:14px}
table{border-collapse:collapse;width:100%;background:#11151d;border-radius:10px;
      overflow:hidden;border:1px solid #232a36}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #1e2530;font-size:13px}
th{background:#161b25;color:#94a3b8;font-weight:600;font-size:11px;
   text-transform:uppercase;letter-spacing:0.5px}
tr:last-child td{border-bottom:none}
code{background:#0a0c10;padding:2px 6px;border-radius:3px;font-size:11px;color:#86efac}
.tasks{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.task{background:#11151d;padding:12px;border-radius:8px;border:1px solid #232a36}
.task-name{font-family:'SF Mono',monospace;font-size:11px;color:#a78bfa;margin-bottom:2px}
.task-meta{font-size:10px;color:#5e6a7e;margin-bottom:8px}
.task-grids{display:flex;gap:10px;align-items:flex-start}
.glabel{font-size:9px;color:#5e6a7e;text-transform:uppercase;margin-bottom:3px;letter-spacing:0.5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.tag{display:inline-block;background:#1e2530;color:#86efac;padding:2px 8px;
     border-radius:10px;font-size:11px;margin-right:6px}
</style></head><body>
"""

    hero = f"""
<h1>ARC-AGI · Laptop Pipeline · Insane Report</h1>
<div class="sub">{model_id} · LoRA r={lora_r} · {n_steps} SFT steps · {n_train} augmented samples · {sys_str} · MPS fp16/fp32</div>

<div class="hero">
  <div class="kpi"><div class="label">Pre-train acc</div><div class="value">{summary['pre_correct']}/{summary['eval_n']}</div><div class="sub2">{summary['pre_acc']*100:.0f}% · base model</div></div>
  <div class="kpi"><div class="label">Post-train acc</div><div class="value">{summary['post_correct']}/{summary['eval_n']}</div><div class="sub2">{summary['post_acc']*100:.0f}% · w/ adapter</div></div>
  <div class="kpi"><div class="label">Δ accuracy</div><div class="value" style="color:{delta_color}">{delta_sign}{delta_pp:.0f} pp</div><div class="sub2">over {summary['eval_n']} tasks</div></div>
  <div class="kpi"><div class="label">Final loss</div><div class="value">{final_loss:.3f}</div><div class="sub2">min {min_loss:.3f}</div></div>
  <div class="kpi"><div class="label">Trainable</div><div class="value">4.36 M</div><div class="sub2">0.28% of 1.55 B</div></div>
  <div class="kpi"><div class="label">Steps × samples</div><div class="value">{n_steps} × {n_train}</div><div class="sub2">batch=1, accum=2</div></div>
</div>

<div class="note">
  <b>What this is</b> &mdash; an end-to-end demo of the same training stack the H100 notebook uses
  (PEFT LoRA + TRL SFTTrainer + TensorBoard), scaled down to fit a laptop's MPS backend.
  Pre-eval and post-eval are head-to-head on the <i>same</i> 5 ARC public-eval tasks so
  the delta is honest. Every chart below is sourced from the actual TensorBoard event files
  written during training &mdash; no fake numbers.
</div>
"""

    cmp_section = f"""
<h2>1 &middot; Head-to-head: every approach we ran locally</h2>
<table>
<thead><tr><th>Approach</th><th>Accuracy</th><th>N</th><th>Wall</th><th>Script</th></tr></thead>
<tbody>{cmp_html}</tbody>
</table>
<div class="note warn" style="margin-top:12px">
  <b>Reality check.</b> ARC-AGI is hard. 60 steps of SFT on 120 samples of a 1.5 B model
  is barely enough to verify the pipeline runs &mdash; not enough to move the needle on a
  benchmark where SOTA fine-tunes use 50 k+ samples and 5 000+ steps on H100. The
  pre &harr; post delta should be read as &ldquo;the LoRA stack works end-to-end,&rdquo;
  not as &ldquo;1.5 B can solve ARC.&rdquo; For real accuracy use <code>arc_kiss.ipynb</code>
  (Colab H100) or program synthesis with <code>solve_with_code.py</code>.
</div>
"""

    training_section = f"""
<h2>2 &middot; Training-time internals (live TensorBoard data)</h2>
<div class="grid2">
  <div class="chart" id="loss"></div>
  <div class="chart" id="grad"></div>
</div>
<div class="grid2">
  <div class="chart" id="lr"></div>
  <div class="chart" id="ent"></div>
</div>
<div class="chart" id="tok"></div>
"""

    lora_section = f"""
<h2>3 &middot; LoRA weight growth across all {len(layers)} layers</h2>
<div class="note purp">
  Each transformer layer gets <b>4 LoRA adapters</b> (q, k, v, o projections), each split into
  A and B matrices. <b>A</b> initializes Gaussian, <b>B</b> initializes to zero &mdash; so
  <b>final-norm(B) ≈ how much that adapter was used during training</b>. Brighter cells =
  more learning. This is the kind of view that tells you whether your LoRA budget is
  going to the right places.
</div>
<h3>3a &middot; Final-norm heatmap (B matrices &mdash; signal of actual adaptation)</h3>
<div class="chart" id="heatB"></div>
<h3>3b &middot; Final-norm heatmap (A matrices &mdash; for completeness)</h3>
<div class="chart" id="heatA"></div>
<h3>3c &middot; Top-{len(top)} fastest-growing B matrices over training</h3>
<div class="chart" id="topb"></div>
"""

    tasks_section = f"""
<h2>4 &middot; The {summary['eval_n']} eval tasks (this is what the model was scored against)</h2>
<div class="tasks">{''.join(preview_html)}</div>
"""

    method_section = """
<h2>5 &middot; Method &amp; honest take</h2>
<div class="note">
<h3 style="margin-top:0">Why fp32 training on MPS instead of bf16?</h3>
PyTorch's MPS backend supports fp16 inference fine but bf16 training is flaky and the AMP
autocast can silently produce NaNs in LoRA grads on some kernels. We pay ~2× the memory
and ~2× the step time for safety. On H100 this would be bf16 with bitsandbytes 4-bit base.
<h3>Why 4 modules (q, k, v, o) and not MLPs?</h3>
ARC's transformations live in <i>relational</i> patterns &mdash; pulling the same cells across
the grid, comparing color sets across examples. Attention projections matter more than
position-wise MLPs for this. <code>r=16</code> on 4 modules &times; 28 layers = 4.36 M trainable
params, the sweet spot between &ldquo;enough capacity&rdquo; and &ldquo;fits a laptop.&rdquo;
<h3>Why D4 + color-permutation augmentation?</h3>
Each base task generates 2 augmented samples: a random rotation/flip from the dihedral-4
group, then a random permutation of color IDs 0-9. This is exactly the augmentation set
used in the BARC paper and the 2024 ARC-Prize winners. It teaches the model to be
invariant to ARC's arbitrary color labels and orientations.
</div>
"""

    plot_js = f"""
<script>
const dark = {{template:'plotly_dark',paper_bgcolor:'#11151d',plot_bgcolor:'#11151d',
              font:{{family:'-apple-system, sans-serif',size:11,color:'#cbd5e1'}},
              margin:{{l:50,r:20,t:40,b:40}},height:280}};

Plotly.newPlot('loss',[{{x:{loss_x},y:{loss_y},mode:'lines+markers',
  line:{{color:'#2dd4bf',width:2}},marker:{{size:4}},name:'loss'}}],
  Object.assign({{}},dark,{{title:'Training loss (per optimizer step)',
  xaxis:{{title:'step'}},yaxis:{{title:'cross-entropy loss'}}}}));

Plotly.newPlot('grad',[{{x:{grad_x},y:{grad_y},mode:'lines+markers',
  line:{{color:'#f59e0b',width:2}},marker:{{size:4}},name:'grad_norm'}}],
  Object.assign({{}},dark,{{title:'Gradient norm — stability indicator',
  xaxis:{{title:'step'}},yaxis:{{title:'||grad||'}}}}));

Plotly.newPlot('lr',[{{x:{lr_x},y:{lr_y},mode:'lines',
  line:{{color:'#a78bfa',width:2}},name:'lr'}}],
  Object.assign({{}},dark,{{title:'Learning-rate schedule (cosine, 5% warmup)',
  xaxis:{{title:'step'}},yaxis:{{title:'lr',exponentformat:'e'}}}}));

Plotly.newPlot('ent',[{{x:{ent_x},y:{ent_y},mode:'lines+markers',
  line:{{color:'#ec4899',width:2}},marker:{{size:4}},name:'entropy'}}],
  Object.assign({{}},dark,{{title:'Output entropy — is the model committing?',
  xaxis:{{title:'step'}},yaxis:{{title:'mean entropy'}}}}));

Plotly.newPlot('tok',[{{x:{tok_x},y:{tok_y},mode:'lines+markers',
  line:{{color:'#84cc16',width:2}},marker:{{size:4}},name:'token-acc'}}],
  Object.assign({{}},dark,{{title:'Mean token accuracy on training batches',
  xaxis:{{title:'step'}},yaxis:{{title:'accuracy',range:[0,1]}},height:240}}));

Plotly.newPlot('heatB',[{{z:{json.dumps(heat_B)},
  x:{json.dumps(['L'+str(L) for L in layers])},
  y:{json.dumps(projs)},
  type:'heatmap',colorscale:'Viridis',colorbar:{{title:'||B||'}}}}],
  Object.assign({{}},dark,{{title:'LoRA B-matrix final norm (proj × layer)',height:260,
  xaxis:{{title:'layer index (0 = embedding-adjacent, {max(layers)} = head-adjacent)'}}}}));

Plotly.newPlot('heatA',[{{z:{json.dumps(heat_A)},
  x:{json.dumps(['L'+str(L) for L in layers])},
  y:{json.dumps(projs)},
  type:'heatmap',colorscale:'Plasma',colorbar:{{title:'||A||'}}}}],
  Object.assign({{}},dark,{{title:'LoRA A-matrix final norm (proj × layer)',height:260,
  xaxis:{{title:'layer index'}}}}));

Plotly.newPlot('topb',[
""" + ",\n".join([
        f"{{x:{json.dumps(v[0])},y:{json.dumps(v[1])},mode:'lines',name:'L{k[0]} {k[1]} B',line:{{width:2}}}}"
        for k, v in top
    ]) + f"""
],Object.assign({{}},dark,{{title:'Top {len(top)} adapters by final ||B|| over training',
xaxis:{{title:'step'}},yaxis:{{title:'||B||'}},height:340}}));
</script>
"""

    footer = f"""
<div class="sub" style="margin-top:32px;border-top:1px solid #1e2530;padding-top:14px">
  Generated {time.strftime('%Y-%m-%d %H:%M')} · data sources:
  <code>{TRAIN_RES.name}</code>, <code>{TB_DIR.relative_to(HERE)}/events.*</code>,
  <code>results.json</code>, <code>results_3b_50.json</code>,
  <code>{ARC_DIR.relative_to(HERE)}</code>.
</div>
</body></html>"""

    OUT.write_text(head + hero + cmp_section + training_section + lora_section
                   + tasks_section + method_section + plot_js + footer)
    print(f"[report] wrote {OUT}  ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
