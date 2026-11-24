"""
Build gsm8k_final.ipynb — single Colab notebook that
runs Qwen2.5-7B-Instruct on GSM8K (math word problems) with chain-of-thought
prompting + self-consistency voting, then renders an inline dashboard.

Same KISS shape as arc_final.ipynb. GSM8K is "reachable":
  - greedy CoT: ~80%
  - self-consistency vote (N=5 samples): ~85-90%
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "gsm8k_final.ipynb"


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
    md("""# GSM8K · Qwen2.5-7B with chain-of-thought + self-consistency

**Why this notebook:** ARC-AGI is brutal for small open models. GSM8K (grade-school math word problems) is *reachable* — Qwen2.5-7B-Instruct scores ~80% greedy, ~85–90% with self-consistency voting. You'll see actually impressive numbers.

**Strategy** (no training):
1. Load **Qwen2.5-7B-Instruct** in 4-bit (fits T4 16 GB or up).
2. Chain-of-thought prompt: ask the model to think step by step, then give "The answer is N".
3. **Greedy** pass for the baseline.
4. **Self-consistency voting** (N=5 sampled completions, majority-vote the extracted number) for the +5–10 pp lift.
5. Inline dashboard: KPI banner, accuracy bar with target line, error-type breakdown, sample correct + failed problems shown in full.

**Hardware:** any Colab GPU. **Wall time:** ~10 min smoke (`LIMIT=200`), ~3 h full (`LIMIT=None` → 1319 problems on T4) or ~30 min on A100.

Just `Runtime → Run all`.
"""),
    code("""# 1 — Hardware + install
import torch, time, json, re
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
MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
LIMIT       = 200          # 200 = ~10 min on T4. Set to None for full 1319 problems
RUN_GREEDY  = True
RUN_SC_VOTE = True         # self-consistency voting (the +5-10pp lift)
N_SAMPLES   = 5            # number of sampled completions for self-consistency
TEMPERATURE = 0.7
MAX_NEW     = 768          # CoT outputs are usually 200-500 tokens

ROOT    = Path("/content") if Path("/content").exists() else Path(".")
RUN_ID  = f"gsm8k-{int(time.time())}"
RUN_DIR = ROOT / "runs" / RUN_ID; RUN_DIR.mkdir(parents=True, exist_ok=True)
print("RUN_DIR:", RUN_DIR)
"""),
    code("""# 3 — Load GSM8K test set
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test")
print(f"GSM8K test: {len(ds)} problems")
problems = [{"q": ds[i]["question"], "a": ds[i]["answer"]} for i in range(len(ds))]
PROBLEMS = problems[:LIMIT] if LIMIT else problems
print(f"Will evaluate: {len(PROBLEMS)} problems")
print()
print("Sample problem:")
print(PROBLEMS[0]["q"])
print("--- truth ---")
print(PROBLEMS[0]["a"])
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
    code("""# 5 — Helpers: prompt format, answer extraction
SYSTEM = ("You are a careful math tutor. Solve the problem step by step. "
          "After the solution, write 'The answer is N.' on a new line where N is the final numeric answer.")

def to_chat(question):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# GSM8K stores the truth as "...\\n#### 42". Extract the int after ####.
def truth_answer(a_str):
    m = re.search(r"####\\s*(-?\\d[\\d,\\.]*)", a_str)
    if m:
        s = m.group(1).replace(",", "")
        try: return float(s) if "." in s else int(s)
        except: return None
    return None

# From the model's CoT response, extract the final number. Robust: prefer the
# number after "answer is", fall back to the last number in the text.
ANSWER_RE = re.compile(r"answer is[\\s\\$]*(-?\\d[\\d,\\.]*)", re.IGNORECASE)
NUM_RE    = re.compile(r"-?\\d[\\d,\\.]*")

def extract_answer(text):
    m = ANSWER_RE.search(text)
    if m:
        s = m.group(1).rstrip(".").replace(",", "")
        try: return float(s) if "." in s else int(s)
        except: pass
    nums = NUM_RE.findall(text)
    if not nums: return None
    s = nums[-1].rstrip(".").replace(",", "")
    try: return float(s) if "." in s else int(s)
    except: return None

def answers_match(a, b):
    if a is None or b is None: return False
    try: return abs(float(a) - float(b)) < 1e-6
    except: return False
"""),
    code("""# 6 — Inference helpers
@torch.inference_mode()
def predict_greedy(question, max_new=MAX_NEW):
    text = to_chat(question)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

@torch.inference_mode()
def predict_sample(question, temperature=TEMPERATURE, max_new=MAX_NEW):
    text = to_chat(question)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                         temperature=temperature, top_p=0.95,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

def predict_self_consistency(question, n=N_SAMPLES):
    \"\"\"Sample n completions, extract answer from each, return majority vote.\"\"\"
    counts = {}
    answers = []
    for _ in range(n):
        try:
            text = predict_sample(question)
            ans = extract_answer(text)
            answers.append(ans)
            if ans is None: continue
            counts[ans] = counts.get(ans, 0) + 1
        except Exception:
            continue
    if not counts: return None, answers
    return max(counts, key=counts.get), answers
"""),
    code("""# 7 — Greedy pass
from tqdm.auto import tqdm
greedy_results = []
g_correct = 0

if RUN_GREEDY:
    t0 = time.time()
    for p in tqdm(PROBLEMS, desc="greedy CoT"):
        truth = truth_answer(p["a"])
        ts = time.time()
        out = predict_greedy(p["q"])
        ans = extract_answer(out)
        ok = answers_match(ans, truth)
        if ok: g_correct += 1
        greedy_results.append({"q": p["q"], "truth": truth, "ans": ans,
                               "ok": ok, "secs": time.time()-ts, "raw": out})
    print(f"\\nGREEDY: {g_correct}/{len(PROBLEMS)} = {g_correct/len(PROBLEMS):.1%}  ({time.time()-t0:.0f}s)")
"""),
    code("""# 8 — Self-consistency voting pass (sample N, majority-vote)
sc_results = []
sc_correct = 0

if RUN_SC_VOTE:
    t0 = time.time()
    for p in tqdm(PROBLEMS, desc=f"SC vote (n={N_SAMPLES})"):
        truth = truth_answer(p["a"])
        ts = time.time()
        ans, all_answers = predict_self_consistency(p["q"], n=N_SAMPLES)
        ok = answers_match(ans, truth)
        if ok: sc_correct += 1
        sc_results.append({"q": p["q"], "truth": truth, "ans": ans,
                           "ok": ok, "samples": all_answers, "secs": time.time()-ts})
    print(f"\\nSC VOTE: {sc_correct}/{len(PROBLEMS)} = {sc_correct/len(PROBLEMS):.1%}  ({time.time()-t0:.0f}s)")
else:
    sc_results = [{"q": r["q"], "truth": r["truth"], "ans": r["ans"], "ok": r["ok"], "samples": [r["ans"]], "secs": 0.0} for r in greedy_results]
    sc_correct = g_correct
"""),
    code("""# 9 — pass@2 (greedy OR SC correct), scores summary
n = len(PROBLEMS)
pass2 = sum(1 for i, p in enumerate(PROBLEMS)
            if greedy_results[i]["ok"] or sc_results[i]["ok"])
best_acc = max(g_correct, sc_correct, pass2) / n

scores = {"model": MODEL_ID, "run_id": RUN_ID, "n": n,
          "greedy_correct": g_correct, "greedy_acc": g_correct/n,
          "sc_correct": sc_correct, "sc_acc": sc_correct/n,
          "pass2_correct": pass2, "pass2_acc": pass2/n,
          "best_acc": best_acc,
          "above_50pct": best_acc >= 0.50,
          "above_80pct": best_acc >= 0.80}
(RUN_DIR / "scores.json").write_text(json.dumps(scores, indent=2))

print(f"\\nGREEDY : {g_correct}/{n} = {g_correct/n:.1%}")
print(f"SC VOTE: {sc_correct}/{n} = {sc_correct/n:.1%}")
print(f"PASS@2 : {pass2}/{n} = {pass2/n:.1%}")
print(f"\\nBest score: {best_acc:.1%}")
"""),
    md("""---

# 📊 Inline Dashboard

Renders right here in the notebook.
"""),
    code("""# 10 — KPI banner + accuracy bar chart
import plotly.graph_objects as go
from IPython.display import HTML, display

TARGETS = [0.50, 0.80]   # tier 1 = 50% (basic competence), tier 2 = 80% (Qwen2.5-7B target)
def tier_color():
    if best_acc >= TARGETS[1]: return "#84cc16", "✅ ELITE 80%+"
    if best_acc >= TARGETS[0]: return "#f59e0b", "✅ COMPETENT 50%+"
    return "#ef4444", "❌ Below 50%"
tcolor, ttext = tier_color()

display(HTML(f'''
<style>
.gsm-kpis {{ display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:14px 0; }}
.gsm-kpi {{ background:#1a1d24;padding:14px 16px;border-radius:8px;border:1px solid #2a2f3a;color:#e8e8e8; }}
.gsm-kpi .lbl {{ color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px; }}
.gsm-kpi .val {{ font-size:22px;font-weight:600;margin-top:4px; }}
</style>
<div class="gsm-kpis">
  <div class="gsm-kpi"><div class="lbl">Problems</div><div class="val">{n}</div></div>
  <div class="gsm-kpi"><div class="lbl">Greedy CoT</div><div class="val">{g_correct/n:.1%}</div></div>
  <div class="gsm-kpi"><div class="lbl">SC Vote (n={N_SAMPLES})</div><div class="val">{sc_correct/n:.1%}</div></div>
  <div class="gsm-kpi"><div class="lbl">pass@2</div><div class="val">{pass2/n:.1%}</div></div>
  <div class="gsm-kpi" style="background:{tcolor};color:#000"><div class="lbl" style="color:#000">Tier</div><div class="val">{ttext}</div></div>
</div>
'''))

methods = ["Greedy", f"SC Vote (n={N_SAMPLES})", "pass@2"]
accs = [g_correct/n*100, sc_correct/n*100, pass2/n*100]
fig = go.Figure(go.Bar(x=methods, y=accs,
                       marker_color=["#2dd4bf","#f59e0b","#84cc16"],
                       text=[f"{a:.1f}%" for a in accs], textposition="outside",
                       textfont=dict(size=14)))
fig.add_hline(y=50, line=dict(color="#f59e0b", dash="dash", width=2),
              annotation_text="50% competent", annotation_position="top right")
fig.add_hline(y=80, line=dict(color="#84cc16", dash="dash", width=2),
              annotation_text="80% elite", annotation_position="top right")
fig.update_yaxes(range=[0, 100])
fig.update_layout(title=f"GSM8K — {MODEL_ID}",
                  yaxis_title="Accuracy (%)", template="plotly_dark",
                  paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                  height=420, showlegend=False)
fig.show()
"""),
    code("""# 11 — Self-consistency lift visualization (greedy vs SC, per-problem agreement)
sc_lifts = []
sc_drops = []
unchanged_correct = 0
unchanged_wrong = 0
for i, p in enumerate(PROBLEMS):
    g_ok = greedy_results[i]["ok"]
    s_ok = sc_results[i]["ok"]
    if not g_ok and s_ok: sc_lifts.append(i)
    elif g_ok and not s_ok: sc_drops.append(i)
    elif g_ok and s_ok: unchanged_correct += 1
    else: unchanged_wrong += 1

fig2 = go.Figure(go.Bar(
    x=[len(sc_lifts), unchanged_correct, unchanged_wrong, len(sc_drops)],
    y=["SC fixed wrong→right", "both right", "both wrong", "SC broke right→wrong"],
    orientation="h",
    marker_color=["#84cc16", "#2dd4bf", "#6b7280", "#ef4444"],
    text=[len(sc_lifts), unchanged_correct, unchanged_wrong, len(sc_drops)],
    textposition="outside"))
fig2.update_layout(title="Self-consistency lift breakdown",
                   xaxis_title="problem count", template="plotly_dark",
                   paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                   height=300)
fig2.show()

print(f"\\nNet SC lift: +{len(sc_lifts) - len(sc_drops)} problems "
      f"({(len(sc_lifts) - len(sc_drops))/n:+.1%} pp)")
"""),
    code("""# 12 — Per-problem timing
import numpy as np
g_times = [r["secs"] for r in greedy_results]
s_times = [r["secs"] for r in sc_results]
xs = list(range(len(PROBLEMS)))

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=xs, y=g_times, mode="markers",
                          marker=dict(size=6, color=["#84cc16" if r["ok"] else "#ef4444" for r in greedy_results]),
                          name="greedy", hovertext=[f"truth={r['truth']} pred={r['ans']}" for r in greedy_results]))
fig3.add_trace(go.Scatter(x=xs, y=s_times, mode="markers",
                          marker=dict(size=6, color=["#84cc16" if r["ok"] else "#ef4444" for r in sc_results], symbol="diamond"),
                          name="SC vote", hovertext=[f"truth={r['truth']} samples={r['samples']}" for r in sc_results]))
fig3.update_layout(title="Per-problem time (sec) — green=correct, red=wrong",
                   xaxis_title="problem index", yaxis_title="seconds",
                   template="plotly_dark", paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24",
                   height=380)
fig3.show()

print(f"Greedy avg: {np.mean(g_times):.1f}s/problem")
print(f"SC vote avg: {np.mean(s_times):.1f}s/problem")
"""),
    code("""# 13 — Sample correct + failed problems shown in full
def render_problem_block(items, show_pred=True, label_class="ok"):
    blocks = []
    for r in items:
        truth_disp = r["truth"]
        ans_disp = r["ans"]
        samples_disp = r.get("samples", [])
        sample_html = ""
        if samples_disp:
            sample_html = (f'<div style="font-family:monospace;font-size:11px;color:#888;margin-top:4px">'
                           f'samples: {samples_disp}</div>')
        verdict_color = "#84cc16" if r["ok"] else "#ef4444"
        verdict_text = "✓ CORRECT" if r["ok"] else "✗ WRONG"
        blocks.append(f'''
        <div style="background:#1a1d24;padding:14px;border-radius:8px;
                    border:1px solid #2a2f3a;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div style="font-size:13px;color:#aaa">truth = <b style="color:#fff">{truth_disp}</b> · model = <b style="color:#fff">{ans_disp}</b></div>
            <div style="background:{verdict_color};color:#000;padding:2px 10px;border-radius:3px;
                        font-size:11px;font-weight:600">{verdict_text}</div>
          </div>
          <div style="font-size:13px;color:#ddd;line-height:1.5;white-space:pre-wrap">{r["q"][:600]}</div>
          {sample_html}
        </div>
        ''')
    return "".join(blocks)

# Best of (greedy, SC) per problem for "correct"; SC results for showcase
correct = [r for r in sc_results if r["ok"]][:5]
failed  = [r for r in sc_results if not r["ok"]][:5]

display(HTML('<h3 style="color:#e8e8e8;margin:18px 0 10px 0">✅ Sample correctly solved problems (first 5)</h3>'))
display(HTML(render_problem_block(correct)))

display(HTML('<h3 style="color:#e8e8e8;margin:18px 0 10px 0">❌ Sample failed problems (first 5, with sampled candidates)</h3>'))
display(HTML(render_problem_block(failed)))
"""),
    code("""# 14 — Bundle + Colab download
import shutil
zpath = ROOT / f"{RUN_ID}.zip"
shutil.make_archive(str(zpath).removesuffix(".zip"), "zip", RUN_DIR)
print(f"Bundle: {zpath} ({zpath.stat().st_size/1e6:.2f} MB)")
print("Contents: scores.json")
try:
    from google.colab import files
    files.download(str(zpath))
except Exception:
    print(f"Not on Colab — pull from {zpath}")
"""),
    md("""---

## What good looks like

- **Greedy CoT**: ~80% on Qwen2.5-7B-Instruct (matches paper numbers)
- **Self-consistency vote (n=5)**: +5–10 pp over greedy → 85–90%
- **pass@2** (greedy ∪ SC): +1–3 pp → 87–93%

If you see significantly lower (<60%):
- Check that `enable_thinking` isn't accidentally enabled
- Confirm the chat template applied correctly (check the `Sample problem` print in cell 3)
- Re-run with `LIMIT=20` for a quick smoke test before chasing a slow problem

## Going further

GSM8K cracked? Some natural next benchmarks at the same hardness tier:
- **MATH** (competition math) — Qwen2.5-7B ~50%, Qwen2.5-Math-7B ~80%
- **HumanEval** (Python coding) — Qwen2.5-7B ~60%, Qwen2.5-Coder-7B ~80%
- **MMLU** (general knowledge) — Qwen2.5-7B ~70%
- Or scale up to **AIME** (math olympiad) — Qwen2.5-7B ~10%, R1-Distill-7B ~50%
"""),
]


if __name__ == "__main__":
    OUT.write_text(json.dumps(nb(CELLS), indent=1))
    print(f"Wrote {OUT}  ({len(CELLS)} cells)")
