"""Cross-foot every number in README.md against results/*.json.

Checks: warm-start rows, final greedy/sampled per arm (3-seed means),
per-class finals, teacher decision counts, extension rows. Exits
non-zero on any mismatch.
"""

import json
import math
import re
import sys

GRID = "results/grid.json"
EXT = "results/extensions.json"
README = "README.md"


def load(path):
    with open(path) as f:
        return json.load(f)


def find(readme, pattern):
    m = re.search(pattern, readme)
    if not m:
        raise SystemExit(f"README pattern not found: {pattern}")
    return m


def check(name, readme_value, computed, tol=0.005):
    if abs(readme_value - computed) > tol:
        raise SystemExit(
            f"CROSSFOOT FAIL {name}: README {readme_value} != "
            f"computed {computed:.4f}")
    print(f"  ok {name}: {computed:.3f}")


def mean(xs):
    return sum(xs) / len(xs)


def main():
    grid = load(GRID)
    readme = open(README).read()
    print("crossfooting README against", GRID)
    arms = ["grpo", "autoscaffold", "fixed", "novalid", "qmax1"]
    for arm in arms:
        finals = [grid[f"{arm}/seed{s}"]["final"] for s in (0, 1, 2)]
        g = mean([f["greedy"] for f in finals])
        s = mean([f["sampled"] for f in finals])
        mg = find(readme, rf"\| {arm} \| ([0-9]+\.[0-9]+) \| ([0-9]+\.[0-9]+) \|")
        check(f"{arm} greedy", float(mg.group(1)), g)
        check(f"{arm} sampled", float(mg.group(2)), s)
        for cls in ("add", "sub", "prec", "mod", "mix3"):
            pc = mean([f["per_class"][cls] for f in finals])
            mp = find(readme,
                      rf"\| {arm} \| {cls} \| ([0-9]+\.[0-9]+) \|")
            check(f"{arm}/{cls}", float(mp.group(1)), pc)
    # warm-start row
    ws = [grid["warm_start"][str(s)] for s in (0, 1, 2)]
    check("warm-start greedy",
          float(find(readme, r"warm-start greedy ([0-9]+\.[0-9]+)").group(1)),
          mean([w["greedy"] for w in ws]))
    # teacher decisions: autoscaffold adopts per seed
    for seed in (0, 1, 2):
        decisions = grid[f"autoscaffold/seed{seed}"]["teacher_decisions"]
        n_adopt = sum(1 for d in decisions for c in d["classes"].values()
                      if c.get("decision") == "adopt")
        m = find(readme, rf"seed {seed} adopts (\d+)")
        if int(m.group(1)) != n_adopt:
            raise SystemExit(
                f"CROSSFOOT FAIL adopts seed {seed}: README {m.group(1)} "
                f"!= {n_adopt}")
        print(f"  ok adopts seed {seed}: {n_adopt}")
    if len(sys.argv) > 1 and sys.argv[1] == "--extensions":
        ext = load(EXT)
        print("crossfooting extensions")
        for key in ("e3-fixed-sample/seed0", "e3-fixed-sample/seed1",
                    "e5-instance-only/seed0", "e5-instance-only/seed1"):
            f = ext[key]["final"]
            m = find(readme, rf"\| {key} \| ([0-9]+\.[0-9]+) \| ([0-9]+\.[0-9]+) \|")
            check(f"{key} greedy", float(m.group(1)), f["greedy"])
            check(f"{key} sampled", float(m.group(2)), f["sampled"])
        for mode in ("grpo", "autoscaffold"):
            for seed in (0, 1):
                key = f"e4-transfer-{mode}/seed{seed}"
                m = find(readme,
                         rf"\| {key} mix3 \| ([0-9]+\.[0-9]+) \|")
                check(f"{key} mix3",
                      float(m.group(1)),
                      ext[key]["mix3_heldout"]["greedy"])
        # E1: scaffolded vs plain sampled at step 120 (3-seed means)
        for arm in ("grpo", "autoscaffold"):
            rows = [ext["e1"][f"{arm}/seed{s}"]["120"] for s in (0, 1, 2)]
            plain = mean([r["plain_sampled"] for r in rows])
            scaff = mean([r["scaffolded_sampled"] for r in rows])
            m = find(readme, rf"E1 step-120 {arm} plain sampled "
                     rf"([0-9]+\.[0-9]+), scaffolded ([0-9]+\.[0-9]+)")
            check(f"e1 {arm} plain", float(m.group(1)), plain)
            check(f"e1 {arm} scaffolded", float(m.group(2)), scaff)
        # E2: MAE and correlation
        m = find(readme, r"E2 MAE ([0-9]+\.[0-9]+), corr (-?[0-9]+\.[0-9]+)")
        check("e2 mae", float(m.group(1)), ext["e2"]["mae_mixed_vs_theory"])
        check("e2 corr", float(m.group(2)), ext["e2"]["corr_mixed_gain"],
              tol=0.01)
        # E6: plain / own / wrong means over both arms and 3 seeds
        for cls in ("add", "prec"):
            vals = {}
            for cond in ("plain", "own", "wrong"):
                vs = [ext["e6"][f"{arm}/seed{s}"][cls][cond]
                      for arm in ("grpo", "autoscaffold") for s in (0, 1, 2)]
                vals[cond] = mean(vs)
            m = find(readme, rf"E6 {cls} plain ([0-9]+\.[0-9]+), own ([0-9]+\.[0-9]+), "
                     rf"wrong ([0-9]+\.[0-9]+)")
            check(f"e6 {cls} plain", float(m.group(1)), vals["plain"])
            check(f"e6 {cls} own", float(m.group(2)), vals["own"])
            check(f"e6 {cls} wrong", float(m.group(3)), vals["wrong"])
    print("CROSSFOOT PASS")


if __name__ == "__main__":
    main()
