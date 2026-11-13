"""Evaluation: scaffold-free greedy / sampled pass@1, per class and
overall, plus the mixed-group fraction (paper Prop. 1 mechanism).
"""

import numpy as np

from .grpo import rollout_batch
from .tasks import classify, prompt_tokens
from .vocab import CLASSES


def evaluate_group(model, problems, group=8, temperature=1.0, rng=None,
                   strategy=None, prefix_map=None, greedy=False):
    """Mean binary reward over `problems` (group rollouts each).

    `strategy` (a token list) is prepended to every prompt; `prefix_map`
    maps task_key -> revealed fraction. Used both for scaffold-free eval
    (strategy=None) and for the teacher's V_C validation.
    """
    from .teacher import task_key
    rng = rng or np.random.default_rng(0)
    prompts = []
    for p in problems:
        frac = 0.0
        if prefix_map is not None:
            frac = prefix_map.get(task_key(p), 0.0)
        prompts.append(prompt_tokens(p, strategy=strategy, prefix_frac=frac))
    trajs = rollout_batch(model, problems, prompts, group=group,
                          temperature=temperature, rng=rng, greedy=greedy)
    return float(np.mean([t["reward"] for t in trajs]))


def evaluate(model, problems, group=8, temperature=1.0, rng=None,
             greedy=True):
    """Scaffold-free evaluation. Returns a dict with overall + per-class
    greedy and sampled pass@1, and the mixed-group fraction."""
    rng = rng or np.random.default_rng(0)
    prompts = [prompt_tokens(p) for p in problems]
    greedy_acc = None
    if greedy:
        trajs = rollout_batch(model, problems, prompts, group=1,
                              rng=rng, greedy=True)
        greedy_acc = float(np.mean([t["reward"] for t in trajs]))
    trajs = rollout_batch(model, problems, prompts, group=group,
                          temperature=temperature, rng=rng)
    rewards = np.array([t["reward"] for t in trajs])
    groups = np.array([t["group"] for t in trajs])
    sampled = float(rewards.mean())
    mixed = float(np.mean([0.0 < rewards[groups == g].mean() < 1.0
                           for g in np.unique(groups)]))
    per_class = {}
    for cls in CLASSES:
        idx = [i for i, p in enumerate(problems) if p["class"] == cls]
        if not idx:
            continue
        r = np.array([trajs[i * group + j]["reward"]
                      for i in idx for j in range(group)])
        per_class[cls] = float(r.mean())
    return {"greedy": greedy_acc, "sampled": sampled,
            "mixed_frac": mixed, "per_class": per_class}


def failure_breakdown(trajs):
    """Counts of failure modes over a list of rollout dicts."""
    from collections import Counter
    return dict(Counter(t["mode"] for t in trajs if t["reward"] == 0.0))
