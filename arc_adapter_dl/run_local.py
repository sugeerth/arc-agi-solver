"""Local ARC inference on Apple Silicon.

Loads Qwen2.5-7B base + the QLoRA adapter trained on vast.ai 3090.
Two modes:
  --hf    : transformers + MPS (works out-of-the-box, slower, ~14GB RAM)
  --mlx   : MLX backend (3-5x faster, requires `mlx-lm` and a GGUF/MLX base)

Usage:
  python3 run_local.py --task /path/to/task.json
  python3 run_local.py --eval-n 10                  # eval first 10 ARC tasks
  python3 run_local.py --interactive                # paste a grid, get an answer
"""
import argparse, json, sys, time
from pathlib import Path

ADAPTER_DIR = Path(__file__).parent
ARC_DIR = Path(__file__).parent.parent / "arc_data" / "evaluation"

def grid_to_str(g): return "\n".join(" ".join(str(c) for c in row) for row in g)
def str_to_grid(s):
    rows = []
    for line in s.strip().splitlines():
        line = line.strip()
        if not line: continue
        try: rows.append([int(c) for c in line.split()])
        except ValueError: return None
    if not rows: return None
    w = len(rows[0])
    if any(len(r) != w for r in rows): return None
    return rows

def format_prompt(task):
    parts = ["You are solving an ARC puzzle. Each grid is a 2D array of digits 0-9 (colors).",
             "Find the transformation rule from the input/output examples, then apply it to the test input.", ""]
    for i, ex in enumerate(task["train"], 1):
        parts.extend([f"Example {i} Input:", grid_to_str(ex["input"]),
                     f"Example {i} Output:", grid_to_str(ex["output"]), ""])
    test = task["test"][0]
    parts.extend(["Test Input:", grid_to_str(test["input"]), "\nTest Output:"])
    return "\n".join(parts)

SYSTEM = "You are an expert ARC puzzle solver. Output only the test output grid as digits separated by spaces, one row per line. No explanations."

def load_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base_id = "Qwen/Qwen2.5-7B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Loading {base_id} on {device} ({dtype})... (first run downloads ~14GB)")
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype, device_map=device)
    if (ADAPTER_DIR / "adapter_config.json").exists():
        print(f"Applying adapter from {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    else:
        print(f"WARN: no adapter at {ADAPTER_DIR}; running BASE model")
    model.eval()
    return tok, model, device

def predict_hf(task, tok, model, device, max_new_tokens=1024):
    import torch
    prompt = format_prompt(task)
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    completion = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    return str_to_grid(completion), completion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="path to one ARC task json")
    ap.add_argument("--eval-n", type=int, help="run on first N eval tasks from arc_data/evaluation/")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    tok, model, device = load_hf()

    if args.task:
        task = json.loads(Path(args.task).read_text())
        t0 = time.time()
        pred, raw = predict_hf(task, tok, model, device)
        print(f"\n--- Prediction ({time.time()-t0:.1f}s) ---")
        print(grid_to_str(pred) if pred else f"[parse fail]\n{raw[:500]}")
        if "test" in task and "output" in task["test"][0]:
            truth = task["test"][0]["output"]
            print(f"\nCorrect: {pred == truth}")

    elif args.eval_n:
        if not ARC_DIR.exists():
            print(f"Need ARC eval data at {ARC_DIR}")
            print("Download with: python3 -c \"import urllib.request,zipfile,io; from pathlib import Path; "
                  "z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip').read())); "
                  "z.extractall('/tmp/arc'); import shutil; shutil.copytree('/tmp/arc/ARC-AGI-master/data', "
                  f"'{ARC_DIR.parent}', dirs_exist_ok=True)\"")
            sys.exit(1)
        files = sorted(ARC_DIR.glob("*.json"))[:args.eval_n]
        correct = 0
        for i, fp in enumerate(files):
            task = json.loads(fp.read_text())
            pred, _ = predict_hf(task, tok, model, device)
            ok = pred == task["test"][0]["output"]
            correct += int(ok)
            print(f"[{i+1}/{len(files)}] {fp.stem}: {'✓' if ok else '✗'} ({correct}/{i+1} = {correct/(i+1):.0%})")
        print(f"\nFINAL: {correct}/{len(files)} = {correct/len(files):.1%}")

    elif args.interactive:
        print("Paste a 'train' list of {input,output} pairs and a 'test' input as JSON. Ctrl-D to submit.")
        data = json.loads(sys.stdin.read())
        if "test" not in data: data["test"] = [{"input": data.pop("input")}]
        pred, raw = predict_hf(data, tok, model, device)
        print(grid_to_str(pred) if pred else f"[parse fail]\n{raw}")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
