"""
Deterministic ARC solver. No LLM. Tries a library of common transforms against
train pairs; if all train pairs match under transform T, apply T to test input.

Transform library covers the "trivially easy" subset of ARC:
  - D4 group (8 rotations/reflections)
  - Transpose
  - Color permutations consistent across all train pairs
  - Constant scale up/down by integer factor
  - Crop to bounding box of non-zero cells
  - Repeat-tile (output is input tiled k×k)
  - Pixelwise color recoloring driven by histogram match
  - Composition: color-recolor ∘ D4

This won't crack hard ARC tasks but it WILL solve the easy ones — which is
exactly the goal here: prove the local pipeline can solve at least one task.
"""
import argparse, json, time
from pathlib import Path
from itertools import permutations

HERE = Path(__file__).parent
EVAL_DIR = HERE / "ARC-AGI-master" / "data" / "evaluation"
TRAIN_DIR = HERE / "ARC-AGI-master" / "data" / "training"


# ---------- grid ops ----------
def rot90(g):  return [list(r) for r in zip(*g[::-1])]
def fliplr(g): return [r[::-1] for r in g]
def flipud(g): return g[::-1]
def transpose(g): return [list(r) for r in zip(*g)]

D4 = [
    ("id",        lambda g: [row[:] for row in g]),
    ("rot90",     lambda g: rot90(g)),
    ("rot180",    lambda g: rot90(rot90(g))),
    ("rot270",    lambda g: rot90(rot90(rot90(g)))),
    ("fliplr",    lambda g: fliplr(g)),
    ("flipud",    lambda g: flipud(g)),
    ("transp",    lambda g: transpose(g)),
    ("antitransp",lambda g: rot90(rot90(transpose(g)))),
]

def shape(g): return (len(g), len(g[0]) if g else 0)

def colors(g):
    s = set()
    for r in g:
        for c in r: s.add(c)
    return s

def bbox(g, bg=0):
    """Bounding box of non-bg cells, or None."""
    h, w = shape(g)
    rs, cs = [], []
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg:
                rs.append(r); cs.append(c)
    if not rs: return None
    return (min(rs), max(rs), min(cs), max(cs))

def crop_to_content(g, bg=0):
    bb = bbox(g, bg)
    if bb is None: return g
    r0, r1, c0, c1 = bb
    return [row[c0:c1+1] for row in g[r0:r1+1]]

def scale_up(g, k):
    return [[g[r][c] for c in range(len(g[0])) for _ in range(k)]
            for r in range(len(g)) for _ in range(k)]

def scale_down(g, k):
    h, w = shape(g)
    if h % k or w % k: return None
    out = []
    for r in range(0, h, k):
        row = []
        for c in range(0, w, k):
            block = {g[r+dr][c+dc] for dr in range(k) for dc in range(k)}
            if len(block) != 1: return None
            row.append(next(iter(block)))
        out.append(row)
    return out

def tile_kxk(g, kh, kw):
    return [[g[r % len(g)][c % len(g[0])]
             for c in range(len(g[0]) * kw)]
            for r in range(len(g) * kh)]

def apply_color_map(g, m):
    return [[m.get(c, c) for c in row] for row in g]


# ---------- candidate generators ----------
def candidates_d4(pairs):
    """Each D4 op that maps every train input -> train output exactly."""
    out = []
    for name, op in D4:
        if all(op(ex["input"]) == ex["output"] for ex in pairs):
            out.append((name, op))
    return out

def candidates_d4_then_recolor(pairs):
    """For each D4 op, derive a per-color map from the first pair and verify
    it holds across all pairs."""
    out = []
    for name, op in D4:
        # learn map from first pair
        if not pairs: continue
        x = op(pairs[0]["input"])
        y = pairs[0]["output"]
        if shape(x) != shape(y): continue
        m = {}
        ok = True
        for r in range(len(x)):
            for c in range(len(x[0])):
                a, b = x[r][c], y[r][c]
                if a in m and m[a] != b: ok = False; break
                m[a] = b
            if not ok: break
        if not ok: continue
        # verify
        if all(apply_color_map(op(ex["input"]), m) == ex["output"] for ex in pairs):
            out.append((f"{name}+recolor", lambda g, op=op, m=m: apply_color_map(op(g), m)))
    return out

def candidates_scale(pairs):
    out = []
    for k in (2, 3, 4, 5):
        if all(scale_up(ex["input"], k) == ex["output"] for ex in pairs):
            out.append((f"scale_up_{k}", lambda g, k=k: scale_up(g, k)))
        ok = True
        for ex in pairs:
            sd = scale_down(ex["input"], k)
            if sd is None or sd != ex["output"]: ok = False; break
        if ok: out.append((f"scale_down_{k}", lambda g, k=k: scale_down(g, k)))
    return out

def candidates_tile(pairs):
    out = []
    for kh in (1,2,3):
        for kw in (1,2,3):
            if kh==1 and kw==1: continue
            if all(tile_kxk(ex["input"], kh, kw) == ex["output"] for ex in pairs):
                out.append((f"tile_{kh}x{kw}", lambda g, kh=kh, kw=kw: tile_kxk(g, kh, kw)))
    return out

def candidates_crop(pairs):
    out = []
    for bg in range(10):
        if all(crop_to_content(ex["input"], bg) == ex["output"] for ex in pairs):
            out.append((f"crop_bg{bg}", lambda g, bg=bg: crop_to_content(g, bg)))
    return out

def candidates_color_perm(pairs):
    """A single global color permutation applied to input -> output (no shape change)."""
    out = []
    if not pairs: return out
    if any(shape(ex["input"]) != shape(ex["output"]) for ex in pairs): return out
    # learn map from union of first pair
    m = {}
    ok = True
    for ex in pairs:
        for r in range(len(ex["input"])):
            for c in range(len(ex["input"][0])):
                a, b = ex["input"][r][c], ex["output"][r][c]
                if a in m and m[a] != b: ok = False; break
                m[a] = b
            if not ok: break
        if not ok: break
    if ok:
        if all(apply_color_map(ex["input"], m) == ex["output"] for ex in pairs):
            out.append(("colorperm", lambda g, m=m: apply_color_map(g, m)))
    return out


def hstack(a, b):
    if len(a) != len(b): return None
    return [a[i] + b[i] for i in range(len(a))]

def vstack(a, b):
    if (a and b and len(a[0]) != len(b[0])): return None
    return [r[:] for r in a] + [r[:] for r in b]


def candidates_concat(pairs):
    """Output is input concatenated with op(input), for op in D4."""
    out = []
    for name, op in D4:
        # hstack(input, op(input))
        def try_make(combine, op=op, combine_name=""):
            def f(g):
                return combine(g, op(g))
            return f
        # h-stack right
        ok = True
        for ex in pairs:
            cand = hstack(ex["input"], op(ex["input"]))
            if cand is None or cand != ex["output"]: ok = False; break
        if ok: out.append((f"hstack_{name}_right", try_make(hstack, op)))
        # h-stack left (op(input) | input)
        ok = True
        for ex in pairs:
            cand = hstack(op(ex["input"]), ex["input"])
            if cand is None or cand != ex["output"]: ok = False; break
        if ok:
            def make_l(op=op):
                def f(g): return hstack(op(g), g)
                return f
            out.append((f"hstack_{name}_left", make_l()))
        # v-stack below
        ok = True
        for ex in pairs:
            cand = vstack(ex["input"], op(ex["input"]))
            if cand is None or cand != ex["output"]: ok = False; break
        if ok:
            def make_vb(op=op):
                def f(g): return vstack(g, op(g))
                return f
            out.append((f"vstack_{name}_below", make_vb()))
        # v-stack above
        ok = True
        for ex in pairs:
            cand = vstack(op(ex["input"]), ex["input"])
            if cand is None or cand != ex["output"]: ok = False; break
        if ok:
            def make_va(op=op):
                def f(g): return vstack(op(g), g)
                return f
            out.append((f"vstack_{name}_above", make_va()))
    return out


def candidates_quad(pairs):
    """Output is 2×2 quadrant tiling of {input, op1(input), op2(input), op3(input)}.
    Try the common 4-fold-symmetric arrangement: input | fliplr; flipud | rot180.
    """
    def quad(g):
        a = g
        b = fliplr(g)
        c = flipud(g)
        d = fliplr(flipud(g))
        return vstack(hstack(a, b), hstack(c, d))
    out = []
    if all(quad(ex["input"]) == ex["output"] for ex in pairs):
        out.append(("quad_symmetric", quad))
    return out


def candidates_crop_any_bg(pairs):
    """Already covered by candidates_crop, but expose a 'most-common' variant."""
    return []  # superseded


def candidates_constant_color(pairs):
    """Output is a constant grid of a single color whose size/value is derived from input."""
    out = []
    # case 1: output is k×k grid of the count of non-bg cells in input (color = bg color of input? no — colour of non-bg)
    # Too specific; skip — handled below by counts variant
    # case 2: output is N×N grid where N=count_of_unique_non_zero_colors and color=most common
    def f(g):
        unique = {c for r in g for c in r if c != 0}
        n = len(unique)
        if n == 0: return None
        # most common non-zero color
        from collections import Counter
        cnt = Counter(c for r in g for c in r if c != 0)
        col = cnt.most_common(1)[0][0]
        return [[col]*n for _ in range(n)]
    ok = all((f(ex["input"]) is not None and f(ex["input"]) == ex["output"]) for ex in pairs)
    if ok: out.append(("count_unique_x_mostcommon", f))
    return out


def candidates_majority_color_fill(pairs):
    """Output shape == input shape, every cell == majority color of input."""
    from collections import Counter
    def f(g):
        cnt = Counter(c for r in g for c in r)
        col = cnt.most_common(1)[0][0]
        return [[col]*len(g[0]) for _ in g]
    out = []
    if all(f(ex["input"]) == ex["output"] for ex in pairs):
        out.append(("majority_fill", f))
    # minority (rarest non-zero)
    def f2(g):
        cnt = Counter(c for r in g for c in r if c != 0)
        if not cnt: return None
        col = min(cnt, key=cnt.get)
        return [[col]*len(g[0]) for _ in g]
    ok = all((f2(ex["input"]) is not None and f2(ex["input"]) == ex["output"]) for ex in pairs)
    if ok: out.append(("minority_nonzero_fill", f2))
    return out


def candidates_symmetrize(pairs):
    """Output completes the input to be symmetric under some D4 op by taking the
    pixel-wise non-zero union of input and op(input). If input is already symmetric,
    this is identity — uninteresting; we require the input to NOT be symmetric."""
    out = []
    def make(op, name):
        def f(g):
            o = op(g)
            if len(o) != len(g) or len(o[0]) != len(g[0]): return None
            return [[(g[r][c] if g[r][c] != 0 else o[r][c]) for c in range(len(g[0]))]
                    for r in range(len(g))]
        return f
    for name, op in D4[1:]:  # skip identity
        f = make(op, name)
        ok = True
        for ex in pairs:
            got = f(ex["input"])
            if got is None or got != ex["output"]: ok = False; break
        if ok: out.append((f"symmetrize_{name}", f))
    return out


def candidates_crop_then_d4(pairs):
    """Try crop_to_content with each possible bg, then any D4 op."""
    out = []
    for bg in range(10):
        for d4_name, d4_op in D4:
            ok = True
            for ex in pairs:
                cropped = crop_to_content(ex["input"], bg)
                if d4_op(cropped) != ex["output"]:
                    ok = False; break
            if ok:
                def make(bg=bg, d4_op=d4_op):
                    def f(g): return d4_op(crop_to_content(g, bg))
                    return f
                out.append((f"crop_bg{bg}+{d4_name}", make()))
    return out


def candidates_crop_then_scale(pairs):
    out = []
    for bg in range(10):
        for k in (2, 3, 4):
            ok = True
            for ex in pairs:
                cropped = crop_to_content(ex["input"], bg)
                if scale_up(cropped, k) != ex["output"]:
                    ok = False; break
            if ok:
                def make(bg=bg, k=k):
                    def f(g): return scale_up(crop_to_content(g, bg), k)
                    return f
                out.append((f"crop_bg{bg}+scale_up_{k}", make()))
    return out


def candidates_extract_largest_object(pairs):
    """Output is the bounding box of the largest connected non-bg component."""
    def largest_bbox(g, bg=0):
        h, w = shape(g)
        seen = [[False]*w for _ in range(h)]
        best = None
        best_size = -1
        from collections import deque
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg and not seen[r][c]:
                    q = deque([(r,c)])
                    seen[r][c] = True
                    cells = []
                    while q:
                        y, x = q.popleft()
                        cells.append((y, x))
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and g[ny][nx] != bg:
                                seen[ny][nx] = True
                                q.append((ny, nx))
                    if len(cells) > best_size:
                        best_size = len(cells)
                        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
                        best = (min(ys), max(ys), min(xs), max(xs))
        if best is None: return g
        r0, r1, c0, c1 = best
        return [row[c0:c1+1] for row in g[r0:r1+1]]
    out = []
    for bg in range(10):
        f = lambda g, bg=bg: largest_bbox(g, bg)
        if all(f(ex["input"]) == ex["output"] for ex in pairs):
            out.append((f"largest_obj_bg{bg}", f))
    return out


def candidates_remove_color(pairs):
    """Output is input with one specific color set to bg (0)."""
    out = []
    for col in range(1, 10):
        def f(g, col=col):
            return [[0 if c == col else c for c in r] for r in g]
        if all(f(ex["input"]) == ex["output"] for ex in pairs):
            out.append((f"zero_color_{col}", f))
    return out


def candidates_keep_only_color(pairs):
    """Output is input with everything except one color zeroed."""
    out = []
    for col in range(1, 10):
        def f(g, col=col):
            return [[c if c == col else 0 for c in r] for r in g]
        if all(f(ex["input"]) == ex["output"] for ex in pairs):
            out.append((f"keep_only_{col}", f))
    return out


def candidates_fill_enclosed(pairs):
    """For each pair, every cell strictly interior to a rectangular ring of color C
    gets recolored to a specific color (which we infer from train pairs)."""
    # too custom; try: "every 0 cell that is fully enclosed (cannot reach edge via 0-only
    # 4-connectivity) becomes color X"
    out = []
    for fill in range(1, 10):
        def f(g, fill=fill):
            h, w = shape(g)
            # cells that can reach edge via 0-only path
            from collections import deque
            reach = [[False]*w for _ in range(h)]
            q = deque()
            for r in range(h):
                for c in range(w):
                    if g[r][c] == 0 and (r==0 or r==h-1 or c==0 or c==w-1):
                        reach[r][c] = True; q.append((r,c))
            while q:
                y, x = q.popleft()
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w and not reach[ny][nx] and g[ny][nx] == 0:
                        reach[ny][nx] = True; q.append((ny,nx))
            return [[fill if (g[r][c]==0 and not reach[r][c]) else g[r][c]
                     for c in range(w)] for r in range(h)]
        if all(f(ex["input"]) == ex["output"] for ex in pairs):
            out.append((f"fill_enclosed_{fill}", f))
    return out


def candidates_alt_tile(pairs):
    """Output is input tiled kh×kw with rows/cols alternating between input and a flip."""
    flips = {"id": lambda x: x, "fliplr": fliplr, "flipud": flipud,
             "rot180": lambda g: fliplr(flipud(g))}
    def tile_alt(g, kh, kw, fr_name, fc_name):
        h, w = shape(g)
        fr = flips[fr_name]; fc = flips[fc_name]
        out = [[0]*(w*kw) for _ in range(h*kh)]
        for i in range(kh):
            for j in range(kw):
                cell = g
                if i % 2 == 1: cell = fr(cell)
                if j % 2 == 1: cell = fc(cell)
                for dr in range(h):
                    for dc in range(w):
                        out[i*h+dr][j*w+dc] = cell[dr][dc]
        return out
    out = []
    for kh in (2, 3, 4):
        for kw in (2, 3, 4):
            for fr in ("id","fliplr","flipud","rot180"):
                for fc in ("id","fliplr","flipud","rot180"):
                    if fr == "id" and fc == "id" and kh*kw == 1: continue
                    ok = True
                    for ex in pairs:
                        ih, iw = shape(ex["input"])
                        oh, ow = shape(ex["output"])
                        if oh != ih*kh or ow != iw*kw: ok=False; break
                        if tile_alt(ex["input"], kh, kw, fr, fc) != ex["output"]:
                            ok = False; break
                    if ok:
                        def make(kh=kh, kw=kw, fr=fr, fc=fc):
                            def f(g): return tile_alt(g, kh, kw, fr, fc)
                            return f
                        out.append((f"alt_tile_{kh}x{kw}_r{fr}_c{fc}", make()))
    return out


ALL = [candidates_d4, candidates_d4_then_recolor, candidates_scale,
       candidates_tile, candidates_crop, candidates_color_perm,
       candidates_concat, candidates_quad,
       candidates_constant_color, candidates_majority_color_fill,
       candidates_symmetrize,
       candidates_crop_then_d4, candidates_crop_then_scale,
       candidates_extract_largest_object,
       candidates_remove_color, candidates_keep_only_color,
       candidates_fill_enclosed,
       candidates_alt_tile]


def solve(task):
    pairs = task["train"]
    for gen in ALL:
        cands = gen(pairs)
        for name, op in cands:
            try:
                pred = op(task["test"][0]["input"])
                return name, pred
            except Exception:
                continue
    return None, None


def run(directory, n, label):
    files = sorted(directory.glob("*.json"))[:n]
    correct = 0
    solved_names = []
    by_rule = {}
    t0 = time.time()
    for i, fp in enumerate(files):
        task = json.loads(fp.read_text())
        truth = task["test"][0]["output"]
        rule, pred = solve(task)
        ok = (pred == truth)
        if ok:
            correct += 1
            solved_names.append((fp.stem, rule))
            by_rule[rule] = by_rule.get(rule, 0) + 1
        marker = "✓" if ok else (("?" if rule else "·"))
        if ok or i < 20 or i % 25 == 0:
            print(f"[{i+1:3d}/{len(files)}] {fp.stem}  {marker} rule={rule or '—'}  acc={correct/(i+1):.1%}")
    total = time.time() - t0
    print(f"\n{'='*60}\n  {label}: {correct}/{len(files)} = {correct/len(files):.1%}  ({total:.1f}s)\n{'='*60}")
    if solved_names:
        print(f"  SOLVED ({len(solved_names)}):")
        for name, rule in solved_names:
            print(f"    {name}  via  {rule}")
        print(f"  by rule:")
        for r, n in sorted(by_rule.items(), key=lambda x:-x[1]):
            print(f"    {r:25s}  {n}")
    return correct, solved_names, by_rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--include-training", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_det.json"))
    args = ap.parse_args()

    print(f"running deterministic solver on EVAL set (n={args.n})...")
    eval_c, eval_solved, eval_rules = run(EVAL_DIR, args.n, "EVAL")
    summary = {"eval": {"correct": eval_c, "n": min(args.n, len(list(EVAL_DIR.glob('*.json')))),
                        "solved": eval_solved, "by_rule": eval_rules}}

    if args.include_training:
        print(f"\nrunning deterministic solver on TRAINING set (n={args.n})...")
        tr_c, tr_solved, tr_rules = run(TRAIN_DIR, args.n, "TRAIN")
        summary["train"] = {"correct": tr_c, "n": min(args.n, len(list(TRAIN_DIR.glob('*.json')))),
                            "solved": tr_solved, "by_rule": tr_rules}

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
