"""
Run a reasoning-distilled model on ARC-AGI on the laptop.

Defaults to DeepSeek-R1-Distill-Qwen-7B. The model emits long <think>...</think>
reasoning blocks before the answer, so we need a grid extractor that finds the
LAST grid-shaped block, not the first.

Usage:
    python3 -u baseline_distill.py --n 10 --max-new 4096
"""
import argparse, json, time, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
EVAL_DIR = HERE / "ARC-AGI-master" / "data" / "evaluation"

SYSTEM = ("You are an expert ARC puzzle solver. Think through the puzzle, then output "
          "the test output grid as digits separated by spaces, one row per line.")


def grid_to_str(g): return "\n".join(" ".join(str(c) for c in row) for row in g)


def extract_last_grid(s):
    """Walk lines from end backwards, collect a contiguous block of grid lines.
    Robust to long thinking blocks before the answer."""
    if not s: return None
    # Strip <think> blocks first if present
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.strip()
    lines = s.splitlines()
    rows = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            if rows: break
            continue
        try:
            row = [int(c) for c in line.split()]
            if not all(0 <= c <= 9 for c in row):  # ARC cells are 0-9
                if rows: break
                continue
            rows.append(row)
        except ValueError:
            if rows: break
            continue
    if not rows: return None
    rows.reverse()
    width = len(rows[0])
    if any(len(r) != width for r in rows): return None
    if len(rows) > 30 or width > 30: return None  # ARC max
    return rows


def format_user(task):
    parts = ["Solve this ARC puzzle. Each grid is 2D digits 0-9 (colors).",
             "Find the rule from examples, apply to test input.",
             "End your answer with the test output grid alone (digits, spaces, newlines).", ""]
    for i, ex in enumerate(task["train"], 1):
        parts.append(f"Example {i} Input:")
        parts.append(grid_to_str(ex["input"]))
        parts.append(f"Example {i} Output:")
        parts.append(grid_to_str(ex["output"]))
        parts.append("")
    parts.append("Test Input:")
    parts.append(grid_to_str(task["test"][0]["input"]))
    parts.append("")
    parts.append("Test Output:")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--out", default=str(HERE / "results_distill.json"))
    ap.add_argument("--log-completion", action="store_true",
                    help="Log first task's full completion for debugging.")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"device={device} dtype={dtype} model={args.model} n={args.n} max_new={args.max_new}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s")

    eval_files = sorted(EVAL_DIR.glob("*.json"))[: args.n]
    print(f"eval tasks: {len(eval_files)}")

    results = []
    correct = 0
    t_run = time.time()
    for i, fp in enumerate(eval_files):
        task = json.loads(fp.read_text())
        truth = task["test"][0]["output"]
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": format_user(task)},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
        t_t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        if i == 0 and args.log_completion:
            print(f"\n=== sample completion (task {fp.stem}) ===\n{completion[:1500]}\n=== end ===\n")
        pred = extract_last_grid(completion)
        ok = pred == truth
        correct += int(ok)
        dt = time.time() - t_t0
        n_out_tokens = out.shape[1] - inp.input_ids.shape[1]
        results.append({
            "task": fp.stem,
            "correct": ok,
            "pred_shape": (len(pred), len(pred[0])) if pred else None,
            "truth_shape": (len(truth), len(truth[0])),
            "out_tokens": int(n_out_tokens),
            "secs": round(dt, 2),
        })
        running = correct / (i + 1)
        print(f"[{i+1:3d}/{len(eval_files)}] {fp.stem}  ok={int(ok)}  "
              f"{dt:5.1f}s  out_tok={n_out_tokens}  acc={running:.1%}")

    total = time.time() - t_run
    summary = {"model": args.model, "n": len(eval_files), "correct": correct,
               "accuracy": correct / len(eval_files), "total_secs": round(total, 1),
               "results": results}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nFINAL: {correct}/{len(eval_files)} = {correct/len(eval_files):.1%}  "
          f"({total:.0f}s, {total/len(eval_files):.1f}s/task)")
    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
