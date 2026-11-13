"""SFT calibration pilot: train the warm-start, report per-class greedy
accuracy, and save the checkpoint for the grid."""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from autoscaffold.evaluate import evaluate
from autoscaffold.model import Transformer
from autoscaffold.sft import train_sft
from autoscaffold.tasks import sample_problems
from autoscaffold.vocab import VOCAB_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/sft_seed0.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train = sample_problems(args.pool, seed=100)
    heldout = sample_problems(48, seed=200)
    model = Transformer(VOCAB_SIZE, max_len=64, d=64, seed=args.seed)
    t0 = time.time()
    hist = train_sft(model, train, steps=args.steps, batch=args.batch,
                     lr=args.lr, rng=rng, log_every=200)
    print(f"SFT {args.steps} steps in {time.time()-t0:.1f}s, "
          f"final loss {hist[-1][1]:.4f}", flush=True)
    ev = evaluate(model, heldout, group=8, rng=rng, greedy=True)
    print("held-out greedy:", ev["greedy"], "sampled:", ev["sampled"],
          "mixed_frac:", ev["mixed_frac"])
    print("per-class sampled:", json.dumps(ev["per_class"]))
    import os
    os.makedirs("checkpoints", exist_ok=True)
    model.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
