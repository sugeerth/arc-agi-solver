"""
ARC-AGI program synthesis baseline.

Instead of asking the model to emit the grid as tokens (which small LMs are bad
at), ask it to write a Python function `solve(grid)`. We then:
  1. Execute the candidate function on the train examples (we know the answers).
  2. Iff all train examples pass, apply solve() to the test input -> prediction.
  3. If any train example fails, retry with the failure as feedback (up to N tries).

This is the BARC / Greenblatt approach. Even small models succeed on simple
patterns because the model only needs to identify the rule -- Python handles
execution perfectly.

Usage:
    python3 -u solve_with_code.py --n 20 --tries 3 --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import copy
import json
import re
import signal
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
EVAL_DIR = HERE / "ARC-AGI-master" / "data" / "evaluation"

SYSTEM = """You are an expert ARC puzzle solver who writes Python code.

You will be shown 2-4 input/output grid pairs that demonstrate a transformation rule.
Your job: write a Python function `solve(grid)` that takes a 2D list of ints (0-9, where each int is a color) and returns the transformed 2D list.

Rules:
- Output ONLY a Python code block. No prose, no explanation.
- Use only stdlib + numpy (imported as `np`) and copy.
- The function must work for the test input, even though it isn't shown.
- Common ARC operations: rotation, reflection, color recoloring, scaling/tiling, flood fill, finding connected components, counting, symmetry."""

USER_TEMPLATE = """Here are the training examples for this puzzle:

{examples}

Write a Python function `solve(grid)` that produces the correct output for ALL training examples. Output only the code:

```python
import numpy as np
import copy

def solve(grid):
    # your implementation
    ...
```"""

RETRY_TEMPLATE = """Your previous code failed on training example {idx}.
Input was:
{inp}
Expected output:
{want}
Your code produced:
{got}

Rewrite the function to handle this case correctly. Output only the corrected code."""


def grid_to_py(grid):
    return "[\n  " + ",\n  ".join(repr(row) for row in grid) + "\n]"


def grid_show(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def format_examples(task):
    parts = []
    for i, ex in enumerate(task["train"], 1):
        parts.append(f"Example {i} input:\n{grid_show(ex['input'])}\n"
                     f"Example {i} output:\n{grid_show(ex['output'])}")
    return "\n\n".join(parts)


def extract_code(s):
    """Pull a ```python ... ``` block out of the model's response. Fall back to
    any ```...``` block. Return None if no code block."""
    m = re.search(r"```python\s*\n(.*?)```", s, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", s, re.DOTALL)
    if m: return m.group(1)
    return None


class TimeoutError(Exception): pass

def _alarm_handler(signum, frame):
    raise TimeoutError("solve() exceeded time limit")


def execute_solve(code_str, input_grid, timeout_s=3):
    """Compile + exec the code, then call solve(input_grid). Returns
    (output_grid_or_None, error_str_or_None)."""
    try:
        compiled = compile(code_str, "<solver>", "exec")
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"

    sandbox = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "filter": filter, "float": float, "int": int,
            "isinstance": isinstance, "len": len, "list": list, "map": map,
            "max": max, "min": min, "range": range, "reversed": reversed,
            "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "zip": zip, "ValueError": ValueError, "KeyError": KeyError,
            "IndexError": IndexError, "TypeError": TypeError, "Exception": Exception,
            "print": lambda *a, **k: None,
            "__import__": __import__,
        },
    }
    try:
        import numpy as np
        sandbox["np"] = np
        sandbox["copy"] = copy
    except Exception:
        pass

    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_s)
        exec(compiled, sandbox)
        if "solve" not in sandbox:
            signal.alarm(0)
            return None, "no solve() defined"
        result = sandbox["solve"](copy.deepcopy(input_grid))
        signal.alarm(0)
    except TimeoutError as e:
        return None, str(e)
    except Exception as e:
        signal.alarm(0)
        return None, f"{type(e).__name__}: {e}"

    if result is None:
        return None, "solve() returned None"
    try:
        # normalize numpy -> list of lists
        import numpy as np
        if isinstance(result, np.ndarray):
            result = result.tolist()
        if not isinstance(result, list) or not result:
            return None, "result not a non-empty list"
        if not all(isinstance(r, list) for r in result):
            return None, "result not a 2D list"
        result = [[int(c) for c in row] for row in result]
        if any(any(c < 0 or c > 9 for c in row) for row in result):
            return None, "cell out of [0,9]"
    except Exception as e:
        return None, f"normalize error: {e}"
    return result, None


def evaluate_on_train(code_str, task):
    """Returns (all_pass: bool, first_failure: dict or None)."""
    for i, ex in enumerate(task["train"]):
        out, err = execute_solve(code_str, ex["input"])
        if err:
            return False, {"idx": i+1, "inp": ex["input"], "want": ex["output"],
                          "got": None, "err": err}
        if out != ex["output"]:
            return False, {"idx": i+1, "inp": ex["input"], "want": ex["output"],
                          "got": out, "err": None}
    return True, None


@torch.inference_mode()
def gen(model, tok, messages, device, max_new=2048):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id, repetition_penalty=1.05)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)


def solve_task(model, tok, task, device, max_tries=3, max_new=2048):
    """Try up to max_tries to find a solve() that passes all training examples.
    Returns dict with status, prediction, code, attempts."""
    attempts = []
    examples = format_examples(task)
    user = USER_TEMPLATE.format(examples=examples)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]

    code = None
    for attempt in range(max_tries):
        resp = gen(model, tok, messages, device, max_new=max_new)
        code = extract_code(resp)
        if not code:
            attempts.append({"raw": resp[:300], "err": "no code block"})
            continue
        ok, fail = evaluate_on_train(code, task)
        attempts.append({"code_len": len(code), "ok": ok,
                         "err": fail.get("err") if fail else None,
                         "fail_idx": fail.get("idx") if fail else None})
        if ok:
            test_pred, err = execute_solve(code, task["test"][0]["input"])
            return {"status": "passed_train", "code": code, "pred": test_pred,
                    "attempts": attempts}
        # build retry message with feedback
        fb = RETRY_TEMPLATE.format(
            idx=fail["idx"],
            inp=grid_show(fail["inp"]),
            want=grid_show(fail["want"]),
            got=(grid_show(fail["got"]) if fail["got"] else f"[{fail['err']}]"),
        )
        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "user", "content": fb})

    # Out of tries — try the LAST candidate on test anyway
    if code:
        test_pred, _ = execute_solve(code, task["test"][0]["input"])
        return {"status": "failed_train", "code": code, "pred": test_pred,
                "attempts": attempts}
    return {"status": "no_code", "code": None, "pred": None, "attempts": attempts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", default=str(HERE / "results_codesolve.json"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device} model={args.model} n={args.n} max_tries={args.tries}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                  device_map=device)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s")

    eval_files = sorted(EVAL_DIR.glob("*.json"))[: args.n]
    print(f"tasks: {len(eval_files)}")

    results = []
    correct = 0
    passed_train_count = 0
    t_run = time.time()
    for i, fp in enumerate(eval_files):
        task = json.loads(fp.read_text())
        truth = task["test"][0]["output"]
        t_t0 = time.time()
        try:
            r = solve_task(model, tok, task, device, max_tries=args.tries, max_new=args.max_new)
        except Exception as e:
            r = {"status": f"error:{e}", "code": None, "pred": None, "attempts": []}
        ok = (r["pred"] == truth)
        correct += int(ok)
        if r["status"] == "passed_train":
            passed_train_count += 1
        results.append({
            "task": fp.stem, "status": r["status"], "correct": ok,
            "pred_shape": (len(r["pred"]), len(r["pred"][0])) if r["pred"] else None,
            "truth_shape": (len(truth), len(truth[0])),
            "n_attempts": len(r["attempts"]),
            "secs": round(time.time() - t_t0, 1),
        })
        rt = (i+1)
        print(f"[{i+1:3d}/{len(eval_files)}] {fp.stem}  status={r['status']:15s}  "
              f"correct={int(ok)}  attempts={len(r['attempts'])}  "
              f"{results[-1]['secs']:5.1f}s  acc={correct/rt:.1%} "
              f"train-pass={passed_train_count}/{rt}")

    total = time.time() - t_run
    summary = {
        "model": args.model, "n": len(eval_files), "max_tries": args.tries,
        "correct": correct, "accuracy": correct / len(eval_files),
        "passed_train": passed_train_count,
        "passed_train_pct": passed_train_count / len(eval_files),
        "total_secs": round(total, 1),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}")
    print(f"  PROGRAM SYNTHESIS RESULTS")
    print(f"  Model: {args.model}")
    print(f"  Accuracy on test: {correct}/{len(eval_files)} = {correct/len(eval_files):.1%}")
    print(f"  Passed all train examples: {passed_train_count}/{len(eval_files)} = {passed_train_count/len(eval_files):.1%}")
    print(f"  Wall time: {total:.0f}s ({total/len(eval_files):.1f}s/task)")
    print(f"{'='*60}")
    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
