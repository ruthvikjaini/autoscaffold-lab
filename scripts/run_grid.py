"""Main grid: 5 arms x 3 seeds x 120 GRPO steps.

Arms: grpo (no scaffold), autoscaffold (full teacher protocol),
fixed (oracle strategy, q=0.5, never updated), novalid (teacher
without held-out validation), qmax1 (teacher with q_max = 1.0).
One SFT warm-start per seed, shared across arms. Resume-safe: finished
(arm, seed) pairs are loaded from the output file and skipped.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from common import run_arm
from autoscaffold.evaluate import evaluate
from autoscaffold.model import Transformer
from autoscaffold.sft import train_sft
from autoscaffold.tasks import sample_problems
from autoscaffold.vocab import VOCAB_SIZE

ARMS = {
    "grpo": {"mode": "grpo"},
    "autoscaffold": {"mode": "autoscaffold"},
    "fixed": {"mode": "fixed"},
    "novalid": {"mode": "autoscaffold", "validate": False},
    "qmax1": {"mode": "autoscaffold", "q_max": 1.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--sft-steps", type=int, default=2500)
    ap.add_argument("--out", default="results/grid.json")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = json.load(f)

    train_pool = sample_problems(600, seed=300)
    valid = sample_problems(12, seed=400)  # V_C: 12 tasks/class
    heldout = sample_problems(48, seed=500)

    for seed in args.seeds:
        ckpt = f"checkpoints/sft_seed{seed}.npz"
        if not os.path.exists(ckpt):
            rng = np.random.default_rng(seed)
            train = sample_problems(2000, seed=100)
            model = Transformer(VOCAB_SIZE, max_len=64, d=64, seed=seed)
            t0 = time.time()
            train_sft(model, train, steps=args.sft_steps, batch=32,
                      lr=3e-3, rng=rng, log_every=500)
            model.save(ckpt)
            print(f"seed {seed}: SFT {args.sft_steps} steps in "
                  f"{time.time()-t0:.0f}s -> {ckpt}", flush=True)
        else:
            print(f"seed {seed}: warm-start loaded from {ckpt}", flush=True)
        base = Transformer.load(ckpt, VOCAB_SIZE, max_len=64, d=64)
        ev0 = evaluate(base, heldout, group=8,
                       rng=np.random.default_rng(seed), greedy=True)
        print(f"seed {seed}: warm-start greedy {ev0['greedy']:.3f} "
              f"sampled {ev0['sampled']:.3f} "
              f"per-class {json.dumps(ev0['per_class'])}", flush=True)
        done.setdefault("warm_start", {})[str(seed)] = ev0

        for arm in args.arms:
            key = f"{arm}/seed{seed}"
            if key in done:
                print(f"skip {key} (already in {args.out})", flush=True)
                continue
            cfg = ARMS[arm]
            print(f"=== {key} ===", flush=True)
            model = base.copy()
            ref = base.copy()
            t0 = time.time()
            results, teacher = run_arm(
                model, ref, train_pool, valid, heldout,
                mode=cfg["mode"], updates=args.updates, seed=seed,
                q_max=cfg.get("q_max", 0.5),
                validate=cfg.get("validate", True),
                log=True,
                save_prefix=f"checkpoints/{arm}_seed{seed}")
            results["seconds"] = time.time() - t0
            done[key] = results
            with open(args.out, "w") as f:
                json.dump(done, f)
            print(f"=== {key} done in {results['seconds']:.0f}s: "
                  f"final greedy {results['final']['greedy']:.3f} "
                  f"sampled {results['final']['sampled']:.3f} ===",
                  flush=True)

    print("grid complete ->", args.out)


if __name__ == "__main__":
    main()
