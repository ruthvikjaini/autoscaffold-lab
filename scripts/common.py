"""Shared training driver for the grid and extension runs.

One run = one (arm, seed): GRPO from a warm-start, with an optional
teacher updating scaffolds every K steps (paper Algorithm 1).
"""

import numpy as np

from autoscaffold.evaluate import evaluate
from autoscaffold.grpo import grpo_update
from autoscaffold.tasks import prompt_tokens
from autoscaffold.teacher import Teacher
from autoscaffold.vocab import CLASSES, STRATEGIES

K = 10  # teacher update interval (paper Sec. 4.1)


def build_prompts(teacher, problems, rng, mode):
    """Build one prompt per problem for the arm's scaffold policy."""
    prompts, infos = [], []
    for p in problems:
        if mode == "grpo":
            prompts.append(prompt_tokens(p))
            infos.append({"scaffolded": False, "prefix_frac": 0.0})
        elif mode == "fixed":
            # Oracle class strategy, q = 0.5 fixed, never updated, no
            # instance hints (paper's fixed-scaffold baseline).
            z = rng.random() < 0.5
            strat = STRATEGIES[p["class"]] if z else None
            prompts.append(prompt_tokens(p, strategy=strat))
            infos.append({"scaffolded": bool(strat), "prefix_frac": 0.0})
        elif mode == "instance-only":
            frac = teacher.prefix_frac.get(
                (p["class"], tuple(p["expr"])), 0.0)
            prompts.append(prompt_tokens(p, prefix_frac=frac))
            infos.append({"scaffolded": False, "prefix_frac": frac})
        else:  # autoscaffold / novalid / qmax1 / fixed-sample
            prompt, info = teacher.build_prompt(p, rng)
            prompts.append(prompt)
            infos.append(info)
    return prompts, infos


def run_arm(model, ref_model, train_pool, valid, heldout, mode, updates=120,
            batch=12, group=8, lr=3e-4, beta=0.02, seed=0,
            q_max=0.5, validate=True, interactive=True, instance=True,
            eval_every=20, log=None, save_prefix=None, save_steps=(40, 80, 120)):
    """Run one arm. Returns (results dict, teacher or None)."""
    rng = np.random.default_rng(seed)
    teacher = None
    if mode in ("autoscaffold", "novalid", "qmax1", "fixed-sample",
                "instance-only"):
        teacher = Teacher(valid, q_max=q_max, validate=validate,
                          interactive=interactive, instance=instance,
                          enable_class=(mode != "instance-only"))
    results = {"mode": mode, "seed": seed, "curve": [], "blocks": [],
               "teacher_decisions": [], "final": None}
    block_log = []
    for step in range(updates):
        idx = rng.integers(0, len(train_pool), size=batch)
        problems = [train_pool[i] for i in idx]
        prompts, infos = build_prompts(teacher, problems, rng, mode)
        trajs, stats = grpo_update(model, ref_model, problems, prompts,
                                   group=group, lr=lr, beta=beta, rng=rng)
        stats["step"] = step
        # block log: one entry per rollout group
        for gi, p in enumerate(problems):
            gtrajs = [t for t in trajs if t["group"] == gi]
            block_log.append({
                "problem": p, "class": p["class"],
                "mean_reward": float(np.mean([t["reward"] for t in gtrajs])),
                "scaffolded": infos[gi]["scaffolded"], "trajs": gtrajs})
        if (step + 1) % eval_every == 0 or step == updates - 1:
            ev = evaluate(model, heldout, group=8, rng=rng, greedy=True)
            ev["step"] = step + 1
            results["curve"].append(ev)
            if save_prefix and (step + 1) in save_steps:
                model.save(f"{save_prefix}_step{step+1}.npz")
            if log:
                print(f"    step {step+1}/{updates} greedy {ev['greedy']:.3f} "
                      f"sampled {ev['sampled']:.3f} "
                      f"mixed {ev['mixed_frac']:.2f} "
                      f"reward {stats['mean_reward']:.2f} "
                      f"signal {stats['signal_frac']:.2f}", flush=True)
        if teacher is not None and (step + 1) % K == 0 and step < updates - 1:
            rec = teacher.update(model, block_log, block=(step + 1) // K,
                                 rng=rng)
            results["teacher_decisions"].append(rec)
            # per-block aggregates for the Prop. 1/3 analysis
            agg = {"block": (step + 1) // K}
            for cls in CLASSES:
                gs = [g for g in block_log if g["class"] == cls]
                if gs:
                    agg[cls] = {
                        "p": float(np.mean([g["mean_reward"] for g in gs])),
                        "mixed_frac": float(np.mean(
                            [0.0 < g["mean_reward"] < 1.0 for g in gs])),
                        "n_all_fail": sum(1 for g in gs
                                          if g["mean_reward"] == 0.0),
                        "n": len(gs)}
            results["blocks"].append(agg)
            block_log = []
    ev = evaluate(model, heldout, group=8, rng=rng, greedy=True)
    results["final"] = ev
    return results, teacher
