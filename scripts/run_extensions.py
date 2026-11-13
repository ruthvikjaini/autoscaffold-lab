"""Extensions (all resume-safe, results appended to results/extensions.json).

E1 scaffold dependence: eval saved checkpoints (autoscaffold + grpo,
steps 40/80/120, plus the warm-start) with the oracle class strategy
vs without it.
E2 Props. 1/3: measured mixed-group fraction vs the theoretical u(p)
curve, and the link between block mixed fraction and next-block gain
(computed from results/grid.json; no new training).
E3 fixed-sample teacher (interactive=False, SkillRL-style), seeds 0-1.
E4 transfer: train on add/sub/prec/mod only; scaffold-free eval on
mix3 vs the grpo control trained the same way, seeds 0-1.
E5 instance-only (class scaffolds disabled), seeds 0-1.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from common import run_arm
from autoscaffold.evaluate import evaluate, evaluate_group
from autoscaffold.model import Transformer
from autoscaffold.tasks import sample_problems
from autoscaffold.vocab import CLASSES, STRATEGIES, VOCAB_SIZE


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def e1(out):
    """Scaffold dependence from saved checkpoints."""
    heldout = sample_problems(48, seed=500)
    res = {"e1": {}}
    for arm in ("grpo", "autoscaffold"):
        for seed in (0, 1, 2):
            row = {}
            # step 0 = warm-start
            for step in (0, 40, 80, 120):
                path = (f"checkpoints/sft_seed{seed}.npz" if step == 0
                        else f"checkpoints/{arm}_seed{seed}_step{step}.npz")
                if not os.path.exists(path):
                    continue
                model = Transformer.load(path, VOCAB_SIZE, max_len=64, d=64)
                rng = np.random.default_rng(0)
                plain = evaluate(model, heldout, group=8, rng=rng,
                                 greedy=True)
                # with the oracle class strategy on every problem
                accs = {}
                for cls in CLASSES:
                    probs = [p for p in heldout if p["class"] == cls]
                    accs[cls] = evaluate_group(
                        model, probs, group=8, rng=rng,
                        strategy=STRATEGIES[cls])
                row[str(step)] = {
                    "plain_greedy": plain["greedy"],
                    "plain_sampled": plain["sampled"],
                    "scaffolded_sampled": float(np.mean(list(accs.values()))),
                    "scaffolded_per_class": accs}
            res["e1"][f"{arm}/seed{seed}"] = row
    out.update(res)
    return res


def e2(out, grid):
    """Prop. 1/3 check from the grid's per-block aggregates."""
    rows = []
    for key, run in grid.items():
        if "/" not in key or "blocks" not in run:
            continue
        blocks = run["blocks"]
        curve = {c["step"]: c for c in run["curve"]}
        for i, b in enumerate(blocks):
            for cls in CLASSES:
                if cls not in b:
                    continue
                p = b[cls]["p"]
                u_theory = 1 - (1 - p) ** 8 - p ** 8
                rows.append({"arm": key, "block": b["block"], "class": cls,
                             "p": p, "mixed_measured": b[cls]["mixed_frac"],
                             "u_theory": u_theory,
                             "n_all_fail": b[cls]["n_all_fail"]})
        # next-block gain vs block mixed fraction. The eval curve is
        # sampled every 20 steps but blocks are every 10, so pair each
        # consecutive curve interval (s1, s2] with the mean mixed
        # fraction of the blocks whose steps fall in (s1, s2].
        curve_steps = sorted(curve)
        for s1, s2 in zip(curve_steps, curve_steps[1:]):
            mids = [b for b in blocks if s1 < b["block"] * 10 <= s2]
            if not mids:
                continue
            gain = curve[s2]["greedy"] - curve[s1]["greedy"]
            mixed = float(np.mean(
                [m[c]["mixed_frac"] for m in mids for c in CLASSES
                 if c in m]))
            rows.append({"arm": key, "interval": [s1, s2],
                         "gain_next_interval": gain,
                         "mixed_frac_interval": mixed})
    # summary: mean |measured - theory| and correlation
    mr = [r for r in rows if "mixed_measured" in r]
    mae = float(np.mean([abs(r["mixed_measured"] - r["u_theory"])
                         for r in mr])) if mr else None
    gr = [r for r in rows if "gain_next_interval" in r]
    if len(gr) > 2:
        x = np.array([r["mixed_frac_interval"] for r in gr])
        y = np.array([r["gain_next_interval"] for r in gr])
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else None
    else:
        corr = None
    out["e2"] = {"rows": rows, "mae_mixed_vs_theory": mae,
                 "corr_mixed_gain": corr}
    return out["e2"]


def _train(out, key, mode, seed, train_pool, valid, heldout, **kw):
    if key in out:
        print(f"skip {key} (already in extensions)", flush=True)
        return
    ckpt = f"checkpoints/sft_seed{seed}.npz"
    base = Transformer.load(ckpt, VOCAB_SIZE, max_len=64, d=64)
    model, ref = base.copy(), base.copy()
    print(f"=== {key} ===", flush=True)
    results, _ = run_arm(model, ref, train_pool, valid, heldout,
                         mode=mode, updates=120, seed=seed, log=True, **kw)
    out[key] = results
    save_json("results/extensions.json", out)
    print(f"=== {key} done: final greedy "
          f"{results['final']['greedy']:.3f} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+",
                    default=["e1", "e2", "e3", "e4", "e5"])
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    out = load_json("results/extensions.json") \
        if os.path.exists("results/extensions.json") else {}
    grid = load_json("results/grid.json")

    train_pool = sample_problems(600, seed=300)
    valid = sample_problems(12, seed=400)
    heldout = sample_problems(48, seed=500)

    if "e1" in args.which:
        e1(out)
        save_json("results/extensions.json", out)
    if "e2" in args.which:
        e2(out, grid)
        save_json("results/extensions.json", out)
    if "e3" in args.which:
        for seed in (0, 1):
            _train(out, f"e3-fixed-sample/seed{seed}", "autoscaffold",
                   seed, train_pool, valid, heldout, interactive=False)
    if "e4" in args.which:
        pool4 = [p for p in train_pool if p["class"] != "mix3"]
        mix3_held = [p for p in heldout if p["class"] == "mix3"]
        for seed in (0, 1):
            for mode in ("grpo", "autoscaffold"):
                key = f"e4-transfer-{mode}/seed{seed}"
                if key in out:
                    continue
                ckpt = f"checkpoints/sft_seed{seed}.npz"
                base = Transformer.load(ckpt, VOCAB_SIZE, max_len=64, d=64)
                model, ref = base.copy(), base.copy()
                print(f"=== {key} ===", flush=True)
                results, _ = run_arm(model, ref, pool4, valid, heldout,
                                     mode=mode, updates=120, seed=seed,
                                     log=True)
                # mix3-only held-out eval (never trained on, never
                # scaffolded for)
                rng = np.random.default_rng(seed)
                results["mix3_heldout"] = evaluate(
                    model, mix3_held, group=8, rng=rng, greedy=True)
                out[key] = results
                save_json("results/extensions.json", out)
    if "e5" in args.which:
        for seed in (0, 1):
            _train(out, f"e5-instance-only/seed{seed}", "instance-only",
                   seed, train_pool, valid, heldout)

    print("extensions complete -> results/extensions.json")


if __name__ == "__main__":
    main()
