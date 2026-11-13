"""E6 follow-up: is the scaffold a strategy or a class cue?

Evaluate step-120 checkpoints (autoscaffold + grpo, 3 seeds) on add and
prec problems with (a) no scaffold, (b) the class's own strategy, and
(c) a mismatched strategy from another class. If the model read the
strategy as advice, (b) should help and (c) should be roughly neutral;
if it reads it as a class cue / positional shift, (c) should collapse
and (b) should not help.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from autoscaffold.evaluate import evaluate_group
from autoscaffold.model import Transformer
from autoscaffold.tasks import sample_problems
from autoscaffold.vocab import STRATEGIES, VOCAB_SIZE


def main():
    heldout = sample_problems(48, seed=500)
    out = {"e6": {}}
    for arm in ("grpo", "autoscaffold"):
        for seed in (0, 1, 2):
            path = f"checkpoints/{arm}_seed{seed}_step120.npz"
            model = Transformer.load(path, VOCAB_SIZE, max_len=64, d=64)
            row = {}
            for cls, wrong in (("add", "mix3"), ("prec", "mod")):
                probs = [p for p in heldout if p["class"] == cls]
                rng = np.random.default_rng(0)
                row[cls] = {
                    "plain": evaluate_group(model, probs, group=8, rng=rng),
                    "own": evaluate_group(model, probs, group=8, rng=rng,
                                          strategy=STRATEGIES[cls]),
                    "wrong": evaluate_group(model, probs, group=8, rng=rng,
                                            strategy=STRATEGIES[wrong]),
                }
            out["e6"][f"{arm}/seed{seed}"] = row
            print(f"{arm}/seed{seed}:", json.dumps(row), flush=True)
    ext = json.load(open("results/extensions.json"))
    ext.update(out)
    with open("results/extensions.json", "w") as f:
        json.dump(ext, f)
    print("E6 complete")


if __name__ == "__main__":
    main()
