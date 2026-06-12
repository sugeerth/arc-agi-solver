"""
Live dashboard generator. Parses run log, renders to dashboard.html.

Adds (vs the previous version):
- Actual ARC grid rendering (input + truth) for the most recent N completed tasks
- Failure-mode breakdown derived from log-known correctness
- Status banner clarifying training vs inference
- Interpretability section that lights up after interpret.html is generated
"""
import re
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
LOG = HERE / "run_3b_50.log"
OUT = HERE / "dashboard.html"
ARC_DIR = HERE / "ARC-AGI-master" / "data" / "evaluation"
INTERP_HTML = HERE / "interpret.html"

LINE = re.compile(r"\[\s*(\d+)/(\d+)\]\s+(\S+)\s+ok=(\d)\s+(\d+(?:\.\d+)?)s\s+acc=([\d.]+)%")
FINAL = re.compile(r"FINAL:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%\s*\(([\d.]+)s")

# Standard ARC color palette (matches arcprize.org)
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 azure
    "#870C25",  # 9 maroon
]


def grid_to_svg(grid, cell=14, max_size=180):
    if not grid: return '<div style="color:#888">[unparseable]</div>'
    h, w = len(grid), len(grid[0])
    # Scale down if grid is very large
    cell = min(cell, max_size // max(h, w))
    cell = max(cell, 4)
    width, height = w * cell, h * cell
    rects = []
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            color = ARC_COLORS[c] if 0 <= c <= 9 else "#444"
            rects.append(
                f'<rect x="{j*cell}" y="{i*cell}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="#1a1d24" stroke-width="0.5"/>'
            )
    return (f'<svg width="{width}" height="{height}" '
            f'style="background:#000;border-radius:3px">{"".join(rects)}</svg>'
            f'<div style="font-size:10px;color:#888;margin-top:2px">{h}×{w}</div>')


def parse():
    if not LOG.exists():
        return {"status": "waiting", "tasks": [], "total": 0, "model": "?"}
    text = LOG.read_text()

    m = re.search(r"model=(\S+)", text)
    model = m.group(1) if m else "?"

    tasks = []
    for ln in text.splitlines():
        mm = LINE.match(ln.strip())
        if not mm: continue
        idx, total, name, ok, secs, acc = mm.groups()
        tasks.append({
            "idx": int(idx), "total": int(total), "name": name,
            "ok": bool(int(ok)), "secs": float(secs), "acc": float(acc),
        })

    final = None
    fm = FINAL.search(text)
    if fm:
        c, n, acc, ts = fm.groups()
        final = {"correct": int(c), "n": int(n), "accuracy": float(acc), "total_secs": float(ts)}

    status = "done" if final else ("running" if tasks else "loading")
    return {"status": status, "tasks": tasks, "model": model, "final": final,
            "total": tasks[0]["total"] if tasks else 0}


def render_recent_grids(tasks, count=4):
    """For the most recent N completed tasks, render input + truth grids."""
    if not tasks: return ""
    blocks = []
    for t in tasks[-count:][::-1]:  # newest first
        task_path = ARC_DIR / f"{t['name']}.json"
        if not task_path.exists():
            continue
        try:
            data = json.loads(task_path.read_text())
            test_input = data["test"][0]["input"]
            truth = data["test"][0]["output"]
        except Exception:
            continue
        badge_color = "#84cc16" if t["ok"] else "#ef4444"
        badge_text = "CORRECT" if t["ok"] else "WRONG"
        blocks.append(f'''
        <div style="background:#1a1d24;padding:12px;border-radius:8px;border:1px solid #2a2f3a;
                    display:flex;gap:18px;align-items:center;margin-bottom:10px;">
            <div style="min-width:80px;">
                <div style="font-family:monospace;font-size:12px;color:#888;">task #{t['idx']}</div>
                <div style="font-family:monospace;font-size:11px;">{t['name']}</div>
                <div style="display:inline-block;padding:2px 8px;border-radius:3px;
                            background:{badge_color};color:#000;font-size:10px;font-weight:600;
                            margin-top:4px;">{badge_text}</div>
                <div style="font-size:11px;color:#888;margin-top:2px;">{t['secs']:.1f}s</div>
            </div>
            <div>
                <div style="font-size:10px;color:#888;margin-bottom:3px;">test input</div>
                {grid_to_svg(test_input)}
            </div>
            <div>
                <div style="font-size:10px;color:#888;margin-bottom:3px;">ground truth</div>
                {grid_to_svg(truth)}
            </div>
        </div>
        ''')
    return "".join(blocks)


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>ARC laptop baseline · live</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       margin: 0; padding: 24px; background: #0e1117; color: #e8e8e8; }}
h1 {{ margin: 0 0 4px 0; font-weight: 500; }}
h2 {{ font-size: 14px; font-weight: 500; color: #aaa;
      margin: 24px 0 10px 0; letter-spacing: 0.5px; text-transform: uppercase; }}
.sub {{ color: #888; margin-bottom: 18px; font-size: 13px; }}
.banner {{ background: #1e3a5f; border-left: 3px solid #2dd4bf;
           padding: 10px 14px; border-radius: 4px; font-size: 13px;
           color: #cce; margin-bottom: 20px; }}
.kpis {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
        margin-bottom: 20px; }}
.kpi {{ background: #1a1d24; padding: 14px 16px; border-radius: 8px;
       border: 1px solid #2a2f3a; }}
.kpi .label {{ color: #888; font-size: 11px; text-transform: uppercase;
              letter-spacing: 0.5px; }}
.kpi .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
.charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.chart {{ background: #1a1d24; padding: 12px; border-radius: 8px;
         border: 1px solid #2a2f3a; }}
.chart.full {{ grid-column: 1 / -1; }}
.status-running {{ color: #2dd4bf; }}
.status-done {{ color: #84cc16; }}
.status-loading, .status-waiting {{ color: #f59e0b; }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
       background: currentColor; margin-right: 6px;
       animation: pulse 1.6s infinite; vertical-align: middle; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1 }} 50% {{ opacity: 0.3 }} }}
a {{ color: #2dd4bf; }}
</style>
</head><body>
<h1>ARC-AGI Laptop Baseline · Live</h1>
<div class="sub">{model} · M3 Pro · MPS / fp16 · auto-refresh every 5s · <span class="status-{status}"><span class="dot"></span>{status_label}</span></div>

<div class="banner">
<b>This is INFERENCE, not training.</b> No weights are being updated → no loss curve.
The relevant signal is whether predictions match truth, and how the model fails.
For training graphs (loss / grad norm / DPO margin / GRPO reward) run the H100 notebook on Colab.
</div>

<div class="kpis">
  <div class="kpi"><div class="label">Tasks done</div><div class="value">{done} / {total}</div></div>
  <div class="kpi"><div class="label">Correct</div><div class="value">{correct}</div></div>
  <div class="kpi"><div class="label">Accuracy</div><div class="value">{acc_pct:.1f}%</div></div>
  <div class="kpi"><div class="label">Avg time / task</div><div class="value">{avg_time:.1f}s</div></div>
  <div class="kpi"><div class="label">ETA</div><div class="value">{eta}</div></div>
</div>

<h2>Eval metrics</h2>
<div class="charts">
  <div class="chart full" id="acc"></div>
  <div class="chart" id="time"></div>
  <div class="chart" id="correctness"></div>
</div>

<h2>Recent tasks · what the model just saw</h2>
<div>{recent_grids}</div>

<h2>Architecture-level interpretability</h2>
<div class="banner" style="border-left-color:{interp_color};background:{interp_bg};color:#cce">
{interp_text}
</div>

<script>
const tasks = {tasks_json};
const layoutBase = {{
  paper_bgcolor: '#1a1d24', plot_bgcolor: '#1a1d24',
  font: {{ color: '#e8e8e8', family: 'system-ui' }},
  margin: {{ l: 50, r: 20, t: 40, b: 40 }},
  xaxis: {{ gridcolor: '#2a2f3a', zerolinecolor: '#2a2f3a' }},
  yaxis: {{ gridcolor: '#2a2f3a', zerolinecolor: '#2a2f3a' }},
}};
if (tasks.length > 0) {{
  Plotly.newPlot('acc', [{{
    x: tasks.map(t => t.idx), y: tasks.map(t => t.acc),
    mode: 'lines+markers', line: {{ color: '#2dd4bf', width: 2 }},
    marker: {{ size: 6 }}, name: 'running acc'
  }}], {{...layoutBase, title: 'Running accuracy (%) over tasks completed', height: 280}}, {{displayModeBar: false}});
  Plotly.newPlot('time', [{{
    x: tasks.map(t => t.idx), y: tasks.map(t => t.secs), type: 'bar',
    marker: {{ color: tasks.map(t => t.ok ? '#84cc16' : '#ef4444') }},
    text: tasks.map(t => t.name), hoverinfo: 'x+y+text',
  }}], {{...layoutBase, title: 'Time per task (sec) — green=correct red=wrong', height: 280}}, {{displayModeBar: false}});
  Plotly.newPlot('correctness', [{{
    x: tasks.map(t => t.idx), y: tasks.map(t => 1), type: 'bar',
    marker: {{ color: tasks.map(t => t.ok ? '#84cc16' : '#ef4444') }},
    text: tasks.map(t => t.ok ? '✓' : '✗'),
    textposition: 'inside', hoverinfo: 'x+text',
  }}], {{...layoutBase, title: 'Correctness sequence', height: 280, yaxis: {{...layoutBase.yaxis, visible: false}}}}, {{displayModeBar: false}});
}} else {{
  document.getElementById('acc').innerHTML = '<div style="padding:60px;text-align:center;color:#888">Waiting for first task to complete...</div>';
}}
</script>
</body></html>
"""


def render():
    s = parse()
    tasks = s["tasks"]
    total = s["total"] or 50
    done = len(tasks)
    correct = sum(1 for t in tasks if t["ok"])
    acc = (correct / done * 100) if done else 0.0
    avg_time = (sum(t["secs"] for t in tasks) / done) if done else 0.0
    remaining = max(0, total - done)
    eta_sec = remaining * avg_time
    eta_str = "—" if not done else f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
    if s["status"] == "done": eta_str = "done"
    status_label = {"loading": "loading model", "running": "evaluating",
                    "done": "completed", "waiting": "waiting"}[s["status"]]

    if INTERP_HTML.exists():
        interp_text = ('Layer profile, logit lens, and per-token confidence views are ready: '
                       '<a href="interpret.html">open interpret.html →</a>')
        interp_color = "#84cc16"
        interp_bg = "#1a3a1a"
    else:
        interp_text = ('Will fire automatically when the main eval finishes. '
                       'Captures: layer activation profiles · logit lens (where the answer '
                       'crystallizes) · per-token confidence/entropy on 3 probe tasks. '
                       'Refresh this page after the run to see them.')
        interp_color = "#f59e0b"
        interp_bg = "#3a2e1a"

    html = HTML.format(
        model=s["model"], status=s["status"], status_label=status_label,
        done=done, total=total, correct=correct, acc_pct=acc,
        avg_time=avg_time, eta=eta_str,
        tasks_json=json.dumps(tasks),
        recent_grids=render_recent_grids(tasks, count=4) or
                     '<div style="color:#888;padding:20px;background:#1a1d24;border-radius:6px">'
                     'Will populate once tasks start completing.</div>',
        interp_text=interp_text, interp_color=interp_color, interp_bg=interp_bg,
    )
    OUT.write_text(html)


if __name__ == "__main__":
    render()
    print(f"wrote {OUT} at {time.strftime('%H:%M:%S')}")
