"""
Agent-RL orchestrator: pre-curates unsolved ARC eval tasks into chunks,
verifies returned solve() functions in a sandbox, aggregates across agents,
reports vs deterministic baseline.

Workflow:
  1. agent_framework.py curate       → writes chunks/chunk_{0..3}.json
  2. (subagents are spawned via the Agent tool — see /Users/fullfocus/arc-agi/
     laptop_baseline/AGENT_INSTRUCTIONS.md)
  3. Each agent writes solutions to outputs/agent_{i}.json
  4. agent_framework.py verify       → reads outputs/*.json, verifies, dumps results
  5. agent_framework.py report       → compares vs deterministic baseline
"""
import argparse, copy, json, signal, sys, time
from pathlib import Path

HERE = Path(__file__).parent
EVAL = HERE / "ARC-AGI-master" / "data" / "evaluation"
CHUNK_DIR = HERE / "agent_chunks"
OUT_DIR = HERE / "agent_outputs"
RESULTS = HERE / "results_agent.json"
DET_RESULTS = HERE / "results_det.json"


# -------- sandbox / verifier --------
class _TO(Exception): pass
def _alrm(*_): raise _TO()

def run_solve(code, grid, timeout=3):
    """Compile + exec the code, then call solve(grid). Returns (out, err)."""
    try:
        compiled = compile(code, "<solve>", "exec")
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"
    import numpy as np
    box = {"__builtins__": {
        "abs":abs,"all":all,"any":any,"bool":bool,"dict":dict,"enumerate":enumerate,
        "filter":filter,"float":float,"int":int,"isinstance":isinstance,"len":len,
        "list":list,"map":map,"max":max,"min":min,"range":range,"reversed":reversed,
        "round":round,"set":set,"sorted":sorted,"str":str,"sum":sum,"tuple":tuple,
        "zip":zip,"ValueError":ValueError,"KeyError":KeyError,"IndexError":IndexError,
        "TypeError":TypeError,"Exception":Exception,"print":lambda *a,**k:None,
        "__import__":__import__,
    }, "np": np, "copy": copy}
    try:
        signal.signal(signal.SIGALRM, _alrm)
        signal.alarm(timeout)
        exec(compiled, box)
        if "solve" not in box:
            signal.alarm(0); return None, "no solve()"
        out = box["solve"](copy.deepcopy(grid))
        signal.alarm(0)
    except _TO:
        return None, "timeout"
    except Exception as e:
        signal.alarm(0); return None, f"{type(e).__name__}: {e}"
    # normalize
    try:
        import numpy as np
        if isinstance(out, np.ndarray): out = out.tolist()
        if not isinstance(out, list) or not out: return None, "not 2D list"
        if not all(isinstance(r, list) for r in out): return None, "not 2D"
        out = [[int(c) for c in r] for r in out]
        if any(c < 0 or c > 9 for r in out for c in r):
            return None, "out of range"
    except Exception as e:
        return None, f"normalize: {e}"
    return out, None


def verify(code, task):
    """Run code on every train pair + test input. Returns dict with status."""
    train_pass = 0
    for ex in task["train"]:
        out, err = run_solve(code, ex["input"])
        if err: return {"status": f"train_err:{err}", "train_pass": train_pass}
        if out != ex["output"]: return {"status": "train_fail", "train_pass": train_pass}
        train_pass += 1
    truth = task["test"][0].get("output")
    out, err = run_solve(code, task["test"][0]["input"])
    if err: return {"status": f"test_err:{err}", "train_pass": train_pass}
    if truth is not None and out == truth:
        return {"status": "solved", "train_pass": train_pass, "pred": out}
    return {"status": "wrong_test", "train_pass": train_pass, "pred": out}


# -------- curate --------
def curate(max_dim=12, n_chunks=4, per_chunk=6, exclude=None):
    """Select small unsolved eval tasks and chunk them."""
    CHUNK_DIR.mkdir(exist_ok=True)
    exclude = set(exclude or [])
    files = sorted(EVAL.glob("*.json"))
    candidates = []
    for fp in files:
        if fp.stem in exclude: continue
        t = json.loads(fp.read_text())
        try:
            ih, iw = len(t["train"][0]["input"]), len(t["train"][0]["input"][0])
            oh, ow = len(t["train"][0]["output"]), len(t["train"][0]["output"][0])
        except Exception:
            continue
        if max(ih, iw, oh, ow) <= max_dim:
            candidates.append((fp.stem, t))
    print(f"[curate] {len(candidates)} small unsolved eval tasks (max_dim={max_dim})")
    # Take first n_chunks * per_chunk
    selected = candidates[: n_chunks * per_chunk]
    for i in range(n_chunks):
        chunk = selected[i*per_chunk:(i+1)*per_chunk]
        out = []
        for name, t in chunk:
            # strip test output so agent can't cheat
            stripped = {"train": t["train"],
                        "test": [{"input": t["test"][0]["input"]}]}
            out.append({"task_id": name, "task": stripped})
        (CHUNK_DIR / f"chunk_{i}.json").write_text(json.dumps(out, indent=2))
        print(f"[curate] chunk_{i}.json  ({len(chunk)} tasks: {[c[0] for c in chunk]})")
    return [c[0] for c in selected]


# -------- aggregate / verify --------
def aggregate_and_verify():
    OUT_DIR.mkdir(exist_ok=True)
    all_returns = []
    for fp in sorted(OUT_DIR.glob("agent_*.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            print(f"[verify] skip malformed {fp.name}: {e}"); continue
        for item in data:
            item["agent"] = fp.stem
            all_returns.append(item)
    print(f"[verify] {len(all_returns)} returned solutions from {len(list(OUT_DIR.glob('agent_*.json')))} agents")

    # Load full tasks (with truth) for verification
    verified = []
    by_task = {}  # task_id -> best verification result
    for item in all_returns:
        tid = item["task_id"]
        code = item.get("code")
        if not code:
            continue
        tp = EVAL / f"{tid}.json"
        if not tp.exists():
            print(f"[verify] {tid} not found in eval"); continue
        task = json.loads(tp.read_text())
        result = verify(code, task)
        result["task_id"] = tid
        result["agent"] = item["agent"]
        result["code"] = code
        if result["status"] == "solved":
            verified.append(result)
            by_task[tid] = result
        else:
            by_task.setdefault(tid, result)

    # union by task_id
    solved_ids = sorted({r["task_id"] for r in verified})
    print(f"[verify] verified solves: {len(solved_ids)}")
    for r in verified:
        print(f"  ✓ {r['task_id']}  (agent={r['agent']})")

    summary = {
        "n_returned": len(all_returns),
        "n_unique_tasks": len(by_task),
        "n_solved": len(solved_ids),
        "solved_ids": solved_ids,
        "verifications": [
            {"task_id": k, "status": v["status"], "agent": v.get("agent")}
            for k, v in by_task.items()
        ],
    }
    RESULTS.write_text(json.dumps(summary, indent=2))
    print(f"[verify] -> {RESULTS}")
    return summary


def report():
    if not RESULTS.exists():
        print("no agent results yet"); return
    a = json.loads(RESULTS.read_text())
    d = json.loads(DET_RESULTS.read_text()) if DET_RESULTS.exists() else {}
    det_solved = {n for n,_ in d.get("eval",{}).get("solved",[])}
    agent_solved = set(a["solved_ids"])
    union = det_solved | agent_solved
    print()
    print("="*60)
    print("  ARC-AGI EVAL · AGENT-RL FRAMEWORK · COMPARISON")
    print("="*60)
    print(f"  Deterministic baseline:  {len(det_solved)} solved")
    if det_solved: print("    " + ", ".join(sorted(det_solved)))
    print(f"  Agent-RL framework:      {len(agent_solved)} solved")
    if agent_solved: print("    " + ", ".join(sorted(agent_solved)))
    print(f"  Union (ensemble):        {len(union)} solved   "
          f"(Δ = +{len(agent_solved - det_solved)} new)")
    print("="*60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curate","verify","report"])
    ap.add_argument("--max-dim", type=int, default=12)
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--per-chunk", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "curate":
        # Exclude already-solved-deterministic so agents focus on new ground
        excl = set()
        if DET_RESULTS.exists():
            d = json.loads(DET_RESULTS.read_text())
            for name, _ in d.get("eval", {}).get("solved", []):
                excl.add(name)
        curate(args.max_dim, args.n_chunks, args.per_chunk, exclude=excl)
    elif args.cmd == "verify":
        aggregate_and_verify()
    elif args.cmd == "report":
        report()
