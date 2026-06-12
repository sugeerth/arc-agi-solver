# ARC-AGI Agentic Solver

An agentic program-synthesis framework for [ARC-AGI](https://github.com/fchollet/ARC-AGI): parallel LLM coding agents write Python `solve()` programs per task, and a sandboxed verifier accepts a solution only if it reproduces **every** training pair and the held-out test output. The result: **22 of 23 attempted eval tasks solved**, where small zero-shot LLMs solve **0**.

## Headline results

All numbers below come directly from the result artifacts committed in [`laptop_baseline/`](laptop_baseline/) — open the JSON files to audit them.

| Approach | Tasks solved | Artifact |
|---|---|---|
| **Agentic solver (program synthesis + sandbox verification)** | **22 / 23** | [`laptop_baseline/results_agent.json`](laptop_baseline/results_agent.json) |
| Qwen2.5-1.5B-Instruct, zero-shot grid prediction | 0 / 10 | [`laptop_baseline/results.json`](laptop_baseline/results.json) |
| Qwen2.5-3B-Instruct, zero-shot grid prediction | 0 / 20 | [`laptop_baseline/results_3b_50.json`](laptop_baseline/results_3b_50.json) |
| Qwen2.5-Coder-3B-Instruct, code generation (2 attempts/task) | 0 / 20 | [`laptop_baseline/results_coder3b.json`](laptop_baseline/results_coder3b.json) |
| Deterministic transform library (no LLM) | 2 / 400 eval (33 / 400 train) | [`laptop_baseline/results_det.json`](laptop_baseline/results_det.json) |

The one agent failure (`2a5f8217`) was a sandbox restriction, not a reasoning failure: the agent's program used `frozenset`, which the verifier's whitelisted-builtins sandbox does not expose (`train_err:NameError: name 'frozenset' is not defined`).

## How the agentic loop works

```
                 ┌──────────────────────────────────────────────┐
                 │  agent_framework.py curate                   │
                 │  · pick small (≤12×12) unsolved eval tasks   │
                 │  · exclude tasks the deterministic solver    │
                 │    already cracks                            │
                 │  · strip test outputs (agents can't cheat)   │
                 │  · shard into agent_chunks/chunk_{0..3}.json │
                 └───────────────┬──────────────────────────────┘
                                 │
              ┌─────────┬────────┴─┬─────────┐
              ▼         ▼          ▼         ▼
          agent_0    agent_1    agent_2   agent_3      4 parallel LLM coding agents,
              │         │          │         │          each writes a Python solve()
              └─────────┴────┬─────┴─────────┘          per task → agent_outputs/
                             ▼
                 ┌──────────────────────────────────────────────┐
                 │  agent_framework.py verify                   │
                 │  · exec solve() in a restricted sandbox      │
                 │    (whitelisted builtins, 3s SIGALRM timeout)│
                 │  · must reproduce EVERY train pair exactly   │
                 │  · then must match the held-out test output  │
                 │  · union best results across agents          │
                 │  → results_agent.json                        │
                 └───────────────┬──────────────────────────────┘
                                 ▼
                 agent_framework.py report   (vs deterministic baseline)
```

Key design choices:

- **Programs, not grids.** Instead of asking a model to emit the output grid token-by-token (which the zero-shot baselines show fails at 0%), each agent writes a Python program expressing the task's transformation rule. Programs generalize; copied grids don't.
- **Verification is the product.** A solution is only counted if the code reproduces all training pairs *and* the test output, executed fresh in a sandbox. There is no partial credit and no self-grading.
- **Anti-cheating curation.** Test outputs are stripped from the task files agents see (`curate` writes train pairs + test *input* only).
- **Ensembling by union.** Solutions from 4 independent agents are merged; the best verification status per task wins.

## Repo layout

```
laptop_baseline/            The agentic framework + all baselines (the headline result)
  agent_framework.py        curate → verify → report orchestrator + sandbox (start here)
  agent_chunks/             task shards given to the agents (test outputs stripped)
  agent_outputs/            raw solve() programs returned by each agent
  results_agent.json        verified agent results: 22/23 solved
  results.json              Qwen2.5-1.5B zero-shot baseline: 0/10
  results_3b_50.json        Qwen2.5-3B zero-shot baseline: 0/20
  results_coder3b.json      Qwen2.5-Coder-3B code-gen baseline: 0/20
  results_det.json          deterministic transform-library baseline: 2/400 eval
  det_solver.py             the no-LLM deterministic solver (D4 / recolor / tile / crop ...)
  baseline.py               zero-shot grid-prediction baseline runner
  solve_with_code.py        code-generation baseline runner
  train_demo.py             local LoRA fine-tuning demo (Apple Silicon / MPS)
  interpret.py              interpretability probes over the local model
  dashboard.html, insane_report.html, train.html   generated visual reports

arc_agi_pipeline/           7-notebook H100/Colab pipeline (scaling-up track)
  01_baseline.ipynb         zero-shot sanity check
  02_train_h100.ipynb       QLoRA SFT (Qwen2.5-14B)
  03_distill_kaggle.ipynb   distill 14B → 3B
  04_eval_ttt.ipynb         eval with test-time training
  05_train_eval_interp.ipynb  base → SFT → DPO → GRPO + 11 interpretability views
  06_run_benchmark.ipynb    full 400-task eval → submission.json
  07_auto_retrain.ipynb     automated self-improvement loop (multi-signal rewards)
  QUICK_START.md            how to run the bundle on Colab

arc_adapter_dl/             QLoRA adapter (Qwen2.5-7B, r=16, trained on a 3090)
                            config + training log + local-inference script committed;
                            weights excluded for size (see .gitignore)

arc_*.ipynb                 standalone notebook variants (5%-budget, KISS, final)
gsm8k_final.ipynb           same training/eval scaffold applied to GSM8K
humaneval_final.ipynb       same scaffold applied to HumanEval
make_*.py                   builders that generate the notebooks above
```

## How to run

The framework needs the ARC-AGI-1 dataset (excluded from this repo — it's a third-party dataset):

```bash
git clone https://github.com/sugeerth/arc-agi-solver
cd arc-agi-solver
git clone https://github.com/fchollet/ARC-AGI laptop_baseline/ARC-AGI-master
pip install numpy
```

Reproduce the verification of the committed agent solutions (re-executes every `solve()` in the sandbox against the real dataset):

```bash
cd laptop_baseline
python3 agent_framework.py verify    # re-runs all programs in agent_outputs/ → results_agent.json
python3 agent_framework.py report    # comparison vs deterministic baseline
```

Run the no-LLM deterministic baseline:

```bash
python3 det_solver.py
```

Run a fresh agent pass: `python3 agent_framework.py curate` shards unsolved tasks into `agent_chunks/`; point your LLM coding agents (any agent that can read a JSON chunk and write `[{"task_id": ..., "code": "...def solve(grid)..."}]` to `agent_outputs/agent_{i}.json`) at the chunks; then `verify`.

For the H100 scaling track, see [`arc_agi_pipeline/QUICK_START.md`](arc_agi_pipeline/QUICK_START.md) — `07_auto_retrain.ipynb` is a self-contained Colab notebook that runs an automated train → mine-hard-tasks → retrain loop with plateau early-stopping.

## Honest limitations

- **Curated task subset.** The 23 agent tasks were deliberately curated: small grids (≤12×12 on the first train pair), drawn from the public eval set, excluding tasks the deterministic solver already solves. This is *not* a full 400-task eval score and is not comparable to ARC-AGI leaderboard numbers.
- **Not a same-model comparison.** The agents are large frontier-LLM coding agents (orchestrated as 4 parallel Claude subagents); the 0% baselines are small local models (1.5B–3B). The comparison demonstrates that *agentic program synthesis with hard verification* succeeds where *zero-shot grid prediction* fails — not that the same model improved.
- **Overlapping but not identical baseline task sets.** The zero-shot baselines ran on the first 10–20 eval tasks alphabetically; 4 of those (e.g. `00dbd492`, `03560426`, `0692e18c`, `08573cc6`) were also solved by the agent, but the sets are not identical.
- **Public eval set.** Frontier models may have seen ARC-AGI-1 public eval tasks in pretraining. The verifier guards against grid memorization (programs must generalize across all train pairs), but cannot rule out task familiarity.
- **Adapter weights not committed.** The QLoRA adapter (154MB) exceeds GitHub's file limit; its config, training log, and inference script are committed instead.

## License

Code in this repo is MIT. The ARC-AGI dataset is © François Chollet, Apache-2.0, distributed from [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI).
