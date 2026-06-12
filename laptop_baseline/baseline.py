"""
ARC-AGI laptop baseline — zero-shot, MPS, fp16.

Runs the chosen Qwen Instruct model on the first N tasks of ARC-AGI public eval
and prints per-task + aggregate accuracy with wall-clock timing.

Usage:
    python3 baseline.py --model Qwen/Qwen2.5-1.5B-Instruct --n 50
"""
import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
EVAL_DIR = HERE / "ARC-AGI-master" / "data" / "evaluation"

SYSTEM = ("You are an expert ARC puzzle solver. Output only the test output grid as digits "
          "separated by spaces, one row per line. No explanations.")


def grid_to_str(g):
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def parse_grid(s):
    rows = []
    for line in s.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append([int(c) for c in line.split()])
        except ValueError:
            return None
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        return None
    return rows


def format_user(task):
    parts = [
        "You are solving an ARC puzzle. Each grid is a 2D array of digits 0-9 (colors).",
        "Find the transformation rule from the input/output examples, then apply it to the test input.",
        "",
    ]
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"device={device}  dtype={dtype}  model={args.model}  n={args.n}")

    # Load model + tokenizer
    t_load_0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
    model.eval()
    print(f"loaded in {time.time()-t_load_0:.1f}s")

    eval_files = sorted(EVAL_DIR.glob("*.json"))[: args.n]
    print(f"eval tasks: {len(eval_files)}")

    results = []
    correct = 0
    t0 = time.time()
    for i, fp in enumerate(eval_files):
        task = json.loads(fp.read_text())
        truth = task["test"][0]["output"]
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": format_user(task)},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        t_task_0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        completion = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_grid(completion)
        ok = pred == truth
        correct += int(ok)
        dt = time.time() - t_task_0
        results.append(
            {
                "task": fp.stem,
                "correct": ok,
                "pred_shape": (len(pred), len(pred[0])) if pred else None,
                "truth_shape": (len(truth), len(truth[0])),
                "secs": round(dt, 2),
            }
        )
        running = correct / (i + 1)
        print(f"[{i+1:3d}/{len(eval_files)}] {fp.stem}  ok={int(ok)}  {dt:5.1f}s  acc={running:.1%}")

    total = time.time() - t0
    summary = {
        "model": args.model,
        "n": len(eval_files),
        "correct": correct,
        "accuracy": correct / len(eval_files),
        "total_secs": round(total, 1),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(
        f"\nFINAL: {correct}/{len(eval_files)} = {correct/len(eval_files):.1%}  "
        f"({total:.0f}s, {total/len(eval_files):.1f}s/task)"
    )
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
