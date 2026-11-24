"""
Build humaneval_final.ipynb — single Colab notebook
that runs Qwen2.5-Coder-7B-Instruct on HumanEval (Python code generation),
executes the generated code against unit tests, and renders an inline dashboard
with the actual code shown for sample passes and failures.

Same KISS shape as gsm8k/arc_final. HumanEval is "reachable":
  - Qwen2.5-7B-Instruct  pass@1 ~58%
  - Qwen2.5-Coder-7B-Instruct pass@1 ~85%   <-- default
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "humaneval_final.ipynb"


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
            "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python","version":"3.10"}},
            "nbformat": 4, "nbformat_minor": 5}


CELLS = [
    md("""# HumanEval · Qwen2.5-Coder-7B + sandboxed code execution

**Why this notebook:** HumanEval is a *reachable* coding benchmark — Qwen2.5-Coder-7B-Instruct cracks **~85% pass@1**. The signal is unambiguous: model writes Python, we execute it, tests pass or fail. Beautiful.

**Strategy:** no training.
1. Load **Qwen2.5-Coder-7B-Instruct** in 4-bit (specifically tuned for code).
2. For each of the 164 HumanEval problems: prompt with function signature + docstring → completion.
3. Exec `prompt + completion + test` in a sandboxed subprocess with timeout.
4. Score pass@1 (greedy) and optionally pass@5 (5 sampled completions, pass if ANY one passes).
5. Inline dashboard: KPI banner + accuracy bar + sample passing code + sample failing code with error trace.

**Hardware:** any Colab GPU. **Wall time:** ~15–20 min on T4 for pass@1, ~1 h for pass@5.

`Runtime → Run all`.
"""),
    code("""# 1 — Hardware + install
import torch, time, json, re, subprocess, tempfile
from pathlib import Path
assert torch.cuda.is_available(), "Set Runtime → GPU."
GPU = torch.cuda.get_device_name(0)
VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU} ({VRAM:.0f} GB)")
"""),
    code("""!pip -q install -U \\
    "transformers>=4.46" "accelerate>=1.1" "bitsandbytes>=0.44" \\
    "datasets>=3.1" "plotly>=5.20" sentencepiece huggingface_hub
"""),
    code("""# 2 — Run config
MODEL_ID    = "Qwen/Qwen2.5-Coder-7B-Instruct"   # Coder variant > base on coding
LIMIT       = None                                 # None = all 164 problems
RUN_PASS_AT_5 = False                              # +1h on T4 for an additional ~5pp
N_SAMPLES   = 5
TEMPERATURE = 0.6
MAX_NEW     = 768
EXEC_TIMEOUT_S = 5

ROOT    = Path("/content") if Path("/content").exists() else Path(".")
RUN_ID  = f"humaneval-{int(time.time())}"
RUN_DIR = ROOT / "runs" / RUN_ID; RUN_DIR.mkdir(parents=True, exist_ok=True)
print("RUN_DIR:", RUN_DIR)
"""),
    code("""# 3 — Load HumanEval test set (164 problems)
from datasets import load_dataset
ds = load_dataset("openai_humaneval", split="test")
problems = [{"task_id": ds[i]["task_id"], "prompt": ds[i]["prompt"],
             "test": ds[i]["test"], "entry_point": ds[i]["entry_point"],
             "canonical_solution": ds[i]["canonical_solution"]}
            for i in range(len(ds))]
PROBLEMS = problems[:LIMIT] if LIMIT else problems
print(f"HumanEval: {len(PROBLEMS)} problems")
print()
print("=== Sample prompt ===")
print(PROBLEMS[0]["prompt"])
"""),
    code("""# 4 — Load Qwen2.5-Coder-7B in 4-bit
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
    code("""# 5 — Helpers: prompt format, code extraction, sandboxed exec
SYSTEM = ("You are an expert Python programmer. Complete the function body to satisfy the docstring. "
          "Output only Python code in a ```python ... ``` block. No prose.")

def to_chat(prompt):
    user = f"Complete this Python function:\\n\\n```python\\n{prompt}```"
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

CODE_RE = re.compile(r"```python\\s*\\n(.*?)```", re.DOTALL)
ANY_CODE_RE = re.compile(r"```\\s*\\n(.*?)```", re.DOTALL)

def extract_code(text, prompt):
    \"\"\"Pull function body out of the model response. Strategy:
    1. If ```python ... ``` block — take it.
    2. Else if ```...``` block — take it.
    3. Else — take everything before the first triple-quote prose.\"\"\"
    m = CODE_RE.search(text)
    if m: return m.group(1).strip()
    m = ANY_CODE_RE.search(text)
    if m: return m.group(1).strip()
    return text.strip()

def execute_solution(prompt, completion, test_code, entry_point, timeout_s=EXEC_TIMEOUT_S):
    \"\"\"Run prompt + completion + test in a sandboxed subprocess.
    Returns (passed: bool, error_msg: str|None).\"\"\"
    # If completion looks like a complete function definition, use it as-is.
    # Else, treat it as a function body to append to the prompt.
    if f"def {entry_point}" in completion:
        full = completion + "\\n\\n" + test_code + f"\\n\\ncheck({entry_point})"
    else:
        full = prompt + completion + "\\n\\n" + test_code + f"\\n\\ncheck({entry_point})"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full); fname = f.name
    try:
        r = subprocess.run(["python3", fname], capture_output=True, text=True,
                          timeout=timeout_s)
        if r.returncode == 0: return True, None
        err = (r.stderr or r.stdout or "").strip()
        return False, err.split("\\n")[-1][:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        try: Path(fname).unlink()
        except: pass
"""),
    code("""# 6 — Inference helpers
@torch.inference_mode()
def predict_greedy(prompt, max_new=MAX_NEW):
    text = to_chat(prompt)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

@torch.inference_mode()
def predict_sample(prompt, temperature=TEMPERATURE, max_new=MAX_NEW):
    text = to_chat(prompt)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                         temperature=temperature, top_p=0.95,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
"""),
    code("""# 7 — pass@1 greedy run
from tqdm.auto import tqdm
greedy_results = []
g_correct = 0

t0 = time.time()
for p in tqdm(PROBLEMS, desc="pass@1 greedy"):
    ts = time.time()
    raw = predict_greedy(p["prompt"])
    code_str = extract_code(raw, p["prompt"])
    passed, err = execute_solution(p["prompt"], code_str, p["test"], p["entry_point"])
    if passed: g_correct += 1
    greedy_results.append({
        "task_id": p["task_id"], "prompt": p["prompt"],
        "code": code_str, "passed": passed, "err": err,
        "secs": time.time() - ts,
    })
print(f"\\npass@1: {g_correct}/{len(PROBLEMS)} = {g_correct/len(PROBLEMS):.1%}  ({time.time()-t0:.0f}s)")
"""),
    code("""# 8 — pass@5 sampled run (optional)
sampled_results = []
sc_correct = 0

if RUN_PASS_AT_5:
    t0 = time.time()
    for p in tqdm(PROBLEMS, desc=f"pass@{N_SAMPLES} sampled"):
        ts = time.time()
        any_pass = False
        attempts = []
        first_pass_code = None
        first_err = None
        for _ in range(N_SAMPLES):
            try:
                raw = predict_sample(p["prompt"])
                code_str = extract_code(raw, p["prompt"])
                ok, err = execute_solution(p["prompt"], code_str, p["test"], p["entry_point"])
                attempts.append({"ok": ok, "err": err})
                if ok and not any_pass:
                    any_pass = True; first_pass_code = code_str
                if not ok and first_err is None:
                    first_err = err
            except Exception as e:
                attempts.append({"ok": False, "err": f"{type(e).__name__}: {e}"})
        if any_pass: sc_correct += 1
        sampled_results.append({
            "task_id": p["task_id"], "prompt": p["prompt"],
            "n_attempts": len(attempts), "n_pass": sum(1 for a in attempts if a["ok"]),
            "passed_at_least_once": any_pass, "first_pass_code": first_pass_code,
            "first_err": first_err, "secs": time.time() - ts,
        })
    print(f"\\npass@{N_SAMPLES}: {sc_correct}/{len(PROBLEMS)} = {sc_correct/len(PROBLEMS):.1%}  ({time.time()-t0:.0f}s)")
else:
    sampled_results = [{"task_id": r["task_id"], "passed_at_least_once": r["passed"],
                        "n_pass": int(r["passed"]), "n_attempts": 1,
                        "first_pass_code": r["code"] if r["passed"] else None,
                        "first_err": r["err"], "secs": r["secs"]}
                       for r in greedy_results]
    sc_correct = g_correct
"""),
    code("""# 9 — Scores summary
n = len(PROBLEMS)
best_acc = max(g_correct, sc_correct) / n

scores = {"model": MODEL_ID, "run_id": RUN_ID, "n": n,
          "pass1_correct": g_correct, "pass1_acc": g_correct/n,
          "passN_correct": sc_correct, "passN_acc": sc_correct/n,
          "N": N_SAMPLES if RUN_PASS_AT_5 else 1,
          "best_acc": best_acc,
          "above_50pct": best_acc >= 0.50,
          "above_75pct": best_acc >= 0.75}
(RUN_DIR / "scores.json").write_text(json.dumps(scores, indent=2))

print(f"\\npass@1: {g_correct}/{n} = {g_correct/n:.1%}")
print(f"pass@{N_SAMPLES if RUN_PASS_AT_5 else 1}: {sc_correct}/{n} = {sc_correct/n:.1%}")
print(f"\\nBest score: {best_acc:.1%}")
"""),
    md("""---

# 📊 Inline Dashboard

Renders right here.
"""),
    code("""# 10 — KPI banner + accuracy bar
import plotly.graph_objects as go
from IPython.display import HTML, display

if best_acc >= 0.75: tcolor, ttext = "#84cc16", "✅ ELITE 75%+"
elif best_acc >= 0.50: tcolor, ttext = "#f59e0b", "✅ COMPETENT 50%+"
else: tcolor, ttext = "#ef4444", "❌ Below 50%"

passN_label = f"pass@{N_SAMPLES}" if RUN_PASS_AT_5 else "(pass@N skipped)"
display(HTML(f'''
<style>
.he-kpis {{ display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:14px 0; }}
.he-kpi {{ background:#1a1d24;padding:14px 16px;border-radius:8px;border:1px solid #2a2f3a;color:#e8e8e8; }}
.he-kpi .lbl {{ color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px; }}
.he-kpi .val {{ font-size:22px;font-weight:600;margin-top:4px; }}
</style>
<div class="he-kpis">
  <div class="he-kpi"><div class="lbl">Problems</div><div class="val">{n}</div></div>
  <div class="he-kpi"><div class="lbl">pass@1</div><div class="val">{g_correct/n:.1%}</div></div>
  <div class="he-kpi"><div class="lbl">{passN_label}</div><div class="val">{sc_correct/n:.1%}</div></div>
  <div class="he-kpi"><div class="lbl">Greedy avg time</div><div class="val">{sum(r['secs'] for r in greedy_results)/n:.1f}s</div></div>
  <div class="he-kpi" style="background:{tcolor};color:#000"><div class="lbl" style="color:#000">Tier</div><div class="val">{ttext}</div></div>
</div>
'''))

methods = ["pass@1", passN_label]
accs = [g_correct/n*100, sc_correct/n*100]
fig = go.Figure(go.Bar(x=methods, y=accs,
                       marker_color=["#2dd4bf","#f59e0b"],
                       text=[f"{a:.1f}%" for a in accs], textposition="outside",
                       textfont=dict(size=14)))
fig.add_hline(y=50, line=dict(color="#f59e0b", dash="dash", width=2),
              annotation_text="50%", annotation_position="top right")
fig.add_hline(y=75, line=dict(color="#84cc16", dash="dash", width=2),
              annotation_text="75% elite", annotation_position="top right")
fig.update_yaxes(range=[0, 100])
fig.update_layout(title=f"HumanEval — {MODEL_ID}",
                  yaxis_title="Accuracy (%)", template="plotly_dark",
                  paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                  height=400, showlegend=False)
fig.show()
"""),
    code("""# 11 — Failure mode breakdown
def err_category(err):
    if err is None: return "passed"
    if "TIMEOUT" in err: return "timeout"
    if "AssertionError" in err: return "assertion (wrong output)"
    if "SyntaxError" in err or "IndentationError" in err: return "syntax"
    if "NameError" in err or "AttributeError" in err: return "name/attr"
    if "TypeError" in err: return "type"
    if "IndexError" in err or "KeyError" in err: return "index/key"
    return "other"

cats = {}
for r in greedy_results:
    cat = err_category(r["err"])
    cats[cat] = cats.get(cat, 0) + 1

cat_order = ["passed", "assertion (wrong output)", "syntax", "name/attr",
             "type", "index/key", "timeout", "other"]
xs = []; ys = []
for k in cat_order:
    if k in cats: xs.append(cats[k]); ys.append(k)
colors = ["#84cc16", "#ef4444", "#dc2626", "#f59e0b", "#a855f7", "#3b82f6", "#6b7280", "#71717a"]

fig2 = go.Figure(go.Bar(x=xs, y=ys, orientation="h",
                        marker_color=colors[:len(xs)],
                        text=[str(v) for v in xs], textposition="outside"))
fig2.update_layout(title="Failure mode breakdown (pass@1)",
                   xaxis_title="problem count", template="plotly_dark",
                   paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                   height=350)
fig2.show()
"""),
    code("""# 12 — Per-problem timing scatter
import numpy as np
xs = list(range(len(PROBLEMS)))
ys = [r["secs"] for r in greedy_results]

fig3 = go.Figure(go.Scatter(
    x=xs, y=ys, mode="markers",
    marker=dict(size=7, color=["#84cc16" if r["passed"] else "#ef4444" for r in greedy_results]),
    hovertext=[r["task_id"] for r in greedy_results], hoverinfo="text+y"))
fig3.update_layout(title="Per-problem time (sec) — green=passed, red=failed",
                   xaxis_title="problem index", yaxis_title="seconds",
                   template="plotly_dark", paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                   height=380)
fig3.show()
print(f"avg time: {np.mean(ys):.1f}s/problem · max: {np.max(ys):.1f}s")
"""),
    code("""# 13 — Sample passing solutions (first 4) shown in full
def render_solution(r, ok=True):
    color = "#84cc16" if ok else "#ef4444"
    badge = "✓ PASSED" if ok else "✗ FAILED"
    err_html = ""
    if not ok and r.get("err"):
        err_html = (f'<div style="background:#0e1117;color:#ef4444;padding:8px;border-radius:4px;'
                    f'font-family:monospace;font-size:12px;margin-top:8px">{r["err"]}</div>')
    code_str = (r["code"][:1200] if "code" in r else (r.get("first_pass_code") or ""))[:1200]
    return f'''
    <div style="background:#1a1d24;padding:14px;border-radius:8px;border:1px solid #2a2f3a;
                margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div style="font-family:monospace;font-size:13px;color:#aaa">{r["task_id"]}</div>
        <div style="background:{color};color:#000;padding:2px 10px;border-radius:3px;
                    font-size:11px;font-weight:600">{badge}</div>
      </div>
      <details><summary style="color:#888;font-size:12px;cursor:pointer">prompt (click to expand)</summary>
        <pre style="background:#0e1117;color:#bbb;padding:10px;border-radius:4px;
                    font-size:12px;white-space:pre-wrap;overflow:auto;max-height:200px">{r["prompt"][:800]}</pre>
      </details>
      <div style="font-size:11px;color:#888;margin-top:8px;margin-bottom:4px">model output:</div>
      <pre style="background:#0e1117;color:#84cc16;padding:10px;border-radius:4px;
                  font-size:12px;white-space:pre-wrap;overflow:auto;max-height:300px">{code_str}</pre>
      {err_html}
    </div>'''

passed = [r for r in greedy_results if r["passed"]][:4]
failed = [r for r in greedy_results if not r["passed"]][:4]

display(HTML('<h3 style="color:#e8e8e8;margin:18px 0 10px 0">✅ Sample passing solutions (first 4)</h3>'))
display(HTML("".join(render_solution(r, True) for r in passed)))

display(HTML('<h3 style="color:#e8e8e8;margin:18px 0 10px 0">❌ Sample failing solutions with error trace (first 4)</h3>'))
display(HTML("".join(render_solution(r, False) for r in failed)))
"""),
    code("""# 14 — Bundle + Colab download
import shutil
zpath = ROOT / f"{RUN_ID}.zip"

# Save full results to RUN_DIR for the bundle
(RUN_DIR / "greedy_results.json").write_text(json.dumps(greedy_results, indent=2))
if RUN_PASS_AT_5:
    (RUN_DIR / "sampled_results.json").write_text(json.dumps(sampled_results, indent=2))

shutil.make_archive(str(zpath).removesuffix(".zip"), "zip", RUN_DIR)
print(f"Bundle: {zpath} ({zpath.stat().st_size/1e6:.2f} MB)")
print("Contents: scores.json, greedy_results.json")
try:
    from google.colab import files
    files.download(str(zpath))
except Exception:
    print(f"Not on Colab — pull from {zpath}")
"""),
    md("""---

## What good looks like

- **Qwen2.5-Coder-7B-Instruct pass@1**: ~85% (matches paper; this notebook's default model)
- **Qwen2.5-7B-Instruct (base)**: ~58% — switch by editing `MODEL_ID`
- **pass@5** (with `RUN_PASS_AT_5=True`): +5–10 pp on top of pass@1

If you see significantly lower than 80%:
- Confirm cell 4 says it loaded `Qwen2.5-Coder-7B-Instruct` not the base
- Run cell 3 — confirm the prompt format looks right
- Try `LIMIT=20` for a quick smoke test

## Going further

HumanEval cracked? Same model handles:
- **MBPP** — easier coding (~75% on Coder-7B)
- **HumanEval+** (evalplus) — adversarial test cases (~75% on Coder-7B)
- **LiveCodeBench** — competition coding (harder, ~30%)
- **BigCodeBench** — practical coding (~50%)

Or pivot to math: `gsm8k_final.ipynb` with the same dashboard pattern.
"""),
]


if __name__ == "__main__":
    OUT.write_text(json.dumps(nb(CELLS), indent=1))
    print(f"Wrote {OUT}  ({len(CELLS)} cells)")
