# ARC-AGI on Colab H100 — Quick Start

## What's in this bundle

7 notebooks. **You only need to run one** for a top-100 attempt: `07_auto_retrain.ipynb`.

| Notebook | Purpose | Run on |
|---|---|---|
| `01_baseline.ipynb` | Zero-shot Qwen2.5-7B sanity check | T4 (free Colab) |
| `02_train_h100.ipynb` | One-shot QLoRA SFT (Qwen2.5-14B) | H100 |
| `03_distill_kaggle.ipynb` | Distill 14B→3B for cheap inference | Kaggle T4×2 |
| `04_eval_ttt.ipynb` | Eval with test-time training | T4 / H100 |
| `05_train_eval_interp.ipynb` | base→SFT→DPO→GRPO + 11 interp views | H100 |
| **`07_auto_retrain.ipynb`** | **Automated self-improvement loop with multi-signal rewards. The score-maximizer.** | **H100** |
| `06_run_benchmark.ipynb` | Run a saved adapter against the full 400-task eval, generate `submission.json` | T4 / H100 |

---

## The 5-step Colab flow

### 1. New notebook → set runtime

- Colab: `Runtime` → `Change runtime type` → **A100 / L4 / H100** (H100 is Pro+)
- Confirm with `!nvidia-smi` (should show 80 GB H100)

### 2. Upload the bundle

Easiest:

```python
# In a Colab cell:
from google.colab import files
uploaded = files.upload()   # pick arc_agi_pipeline.zip

import zipfile, os
zipfile.ZipFile("arc_agi_pipeline.zip").extractall("/content/arc_pipeline")
os.chdir("/content/arc_pipeline")
!ls
```

### 3. Open `07_auto_retrain.ipynb`

In the Colab file panel, double-click `07_auto_retrain.ipynb` → it opens as the active notebook.

### 4. (Optional) Tune the speed knobs

Find cell `0.3 — Run config` and adjust if you want a fast iteration vs a strong score:

| Setting | Fast (4–6 h) | Default (12–24 h) | Strong (24–36 h) |
|---|---|---|---|
| `PRIMARY_MODEL` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-14B` | `Qwen/Qwen3-14B` |
| `N_ROUNDS` | 2 | 4 | 6 |
| `HARD_PER_ROUND` | 80 | 200 | 400 |
| `SFT_AUG_PER_TASK` | 2 | 4 | 6 |

### 5. `Runtime` → `Run all`

The notebook is fully self-contained: pulls data from GitHub, model from HF, runs the auto-loop, evals on the full 400-task ARC-AGI-1 public eval, writes `submission.json`, bundles everything to `runs/<RUN_ID>.zip`, and offers a download link in the last cell.

**Wall-clock on H100 80 GB at default settings:** ~12–24 hours.

---

## What to bring back here for analysis

Once it finishes, you'll have a file like `runs/auto-1730800000.zip` in Colab. Download it and share back:

1. The **`runs/<RUN_ID>/eval/final.json`** file — top-line greedy / D4-vote / pass@2 scores
2. The **score-history line** the notebook prints (e.g. `[r3] holdout acc: 31.2% history: ['12.5%','24.0%','29.5%','31.2%']`)
3. The **`runs/<RUN_ID>/score_history.html`** plot (open in browser, attach screenshot)
4. The **`submission.json`** if you want me to sanity-check format / coverage
5. The full bundle zip if you want me to dig into `loss_curves.html` and per-round eval JSON

With those I can:
- Diagnose any plateau (which round did gains stop, why)
- Recommend whether to push to bigger model, more rounds, or TTT-at-inference
- Help with the Kaggle competition submission if the public-eval score is in top-100 territory

---

## Gotchas to watch for

- **First-run download**: ~30 GB of weights + datasets. Allow ~10 min on Colab's connection before training actually starts.
- **OOM on Qwen3-14B**: the notebook has automatic fallback to `Qwen3-8B`. If you see "Falling back to Qwen3-8B", that's normal — H100 is occasionally tight on the bigger model with QLoRA + DPO ref + GRPO rollouts.
- **Colab idle disconnects**: Pro+ keeps you alive longer but isn't unlimited. Use a browser tab keepalive, or split into multiple runs by lowering `N_ROUNDS`.
- **Plateau early-stop**: the loop stops on its own if 2 rounds gain < 1 pt total. That's healthy — over-training shrinks the policy.

---

## If you have less time / less compute

- **Just want a submission.json from existing weights?** Skip 07, run `06_run_benchmark.ipynb` instead. Point it at any HF Hub adapter (yours or someone else's) and it produces `submission.json` in ~30–60 min on T4.
- **Want forensic plots of what training did?** Use `05_train_eval_interp.ipynb` instead of 07. It generates 11 plotly views (logit lens, layer norms, top neurons, LoRA deltas, etc.) — not a score-maximizer, but tells you *why* training helped.
