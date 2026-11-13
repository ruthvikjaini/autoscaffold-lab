"""Single-turn GRPO on the arithmetic testbed (paper Eq. 3).

Group-standardized advantages on binary verifiable rewards, PPO
clipping (vacuous on-policy with one epoch per rollout batch), and the
per-token k1 KL estimator folded into the advantage,
A_t = adv - beta * (lp - lp_ref) (paper Eq. 4, KL term; the sign is
regression-tested: the KL term must pull the policy toward the
reference, not away from it — see Day 15/17 lessons). If a batch has
no mixed-outcome group (signal_frac == 0) the update is skipped
entirely: with all rewards equal Adam would take a full-size step on
the tiny KL gradient and drift the policy for no signal.
"""

import numpy as np

from .tasks import classify
from .vocab import EOS, PAD


def rollout_batch(model, problems, prompts, group=8, temperature=1.0,
                  rng=None, greedy=False, max_new=24):
    """Sample `group` rollouts per problem on pre-built prompts.

    Returns a list of dicts with problem, prompt, response, reward,
    failure mode, and segment counts. `prompts` is a list of token
    lists, one per problem.
    """
    rng = rng or np.random.default_rng(0)
    seqs = []
    meta = []
    for pi, problem in enumerate(problems):
        for _ in range(group):
            seqs.append(list(prompts[pi]))
            meta.append(pi)
    gens = model.generate(seqs, max_new, temperature=temperature,
                          greedy=greedy, rng=rng)
    out = []
    for gi, pi in enumerate(meta):
        problem = problems[pi]
        resp = gens[gi]
        r, mode, n_ok = classify(problem, resp)
        out.append({"problem": problem, "prompt": prompts[pi],
                    "resp": resp, "reward": r, "mode": mode,
                    "n_seg_ok": n_ok, "group": pi,
                    "scaffolded": None})
    return out


def _padded(trajs):
    """Pad trajectories to arrays in the shifted frame.

    Returns tokens [B, W], resp mask [B, W-1], rewards, groups.
    """
    B = len(trajs)
    W = max(len(t["prompt"]) + len(t["resp"]) for t in trajs)
    tokens = np.full((B, W), PAD, dtype=int)
    resp = np.zeros((B, W - 1), dtype=float)
    for b, tr in enumerate(trajs):
        seq = tr["prompt"] + tr["resp"]
        tokens[b, :len(seq)] = seq
        for j in range(len(tr["prompt"]), len(seq)):
            resp[b, j - 1] = 1.0
    rewards = np.array([t["reward"] for t in trajs])
    groups = np.array([t["group"] for t in trajs])
    return tokens, resp, rewards, groups


def group_advantages(rewards, groups):
    """Group-standardized advantages; 0 for degenerate groups."""
    adv = np.zeros_like(rewards)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        r = rewards[idx]
        mu, sd = r.mean(), r.std()
        if sd > 1e-8:
            adv[idx] = (r - mu) / sd
    return adv


def grpo_update(model, ref_model, problems, prompts, group=8, lr=3e-4,
                clip=0.2, beta=0.02, rng=None, temperature=1.0):
    """One GRPO update on a batch of (problem, prompt) pairs.

    `prompts` are the (possibly scaffolded) prompts for this batch.
    Returns a stats dict; `skipped` is True when no group had a
    mixed outcome.
    """
    rng = rng or np.random.default_rng(0)
    trajs = rollout_batch(model, problems, prompts, group=group,
                          temperature=temperature, rng=rng)
    tokens, resp, rewards, groups = _padded(trajs)
    adv = group_advantages(rewards, groups)
    signal = float(np.mean([rewards[groups == g].std() > 1e-8
                            for g in np.unique(groups)]))
    stats = {"mean_reward": float(rewards.mean()), "signal_frac": signal,
             "skipped": False}
    if signal == 0.0:
        stats["skipped"] = True
        stats["loss_pg"] = 0.0
        stats["mean_kl"] = 0.0
        return trajs, stats

    # On-policy log-probs under the current policy (== rollout logp,
    # no update has happened yet) and under the reference.
    lp = model.logprobs(tokens)
    lp_ref = ref_model.logprobs(tokens)
    kl = lp - lp_ref  # k1 estimator, per token
    # Per-token advantage: trajectory advantage minus beta * KL.
    # lp/lp_ref are detached constants here, so the gradient flows
    # only through logp in the surrogate, exactly as in TRL-style k1.
    A = np.repeat(adv[:, None], lp.shape[1], axis=1) - beta * kl
    A = A * resp

    logits, cache = model.forward(tokens)
    pred = logits[:, :-1, :]
    mx = pred.max(axis=-1, keepdims=True)
    e = np.exp(pred - mx)
    probs = e / e.sum(axis=-1, keepdims=True)
    tgt = tokens[:, 1:]
    B, T1 = tgt.shape
    # PPO ratio is 1 on-policy; the clip is vacuous but kept for form.
    ratio = np.ones_like(lp)
    active = np.where(A > 0, ratio <= 1 + clip,
                      np.where(A < 0, (ratio >= 1 - clip) & (ratio <= 3.0),
                               False))
    w = np.where(active, A * ratio, 0.0)
    n_resp = resp.sum()
    scale = 1.0 / max(n_resp, 1)
    loss_pg = -(w * lp * resp).sum() * scale
    dcoef = w * resp * scale

    d = probs.copy()
    d[np.arange(B)[:, None], np.arange(T1)[None, :], tgt] -= 1.0
    d *= dcoef[:, :, None]
    dlogits = np.zeros_like(logits)
    dlogits[:, :-1, :] = d
    grads = model._backward(dlogits, cache)
    model.adam_step(grads, lr)

    stats["loss_pg"] = float(loss_pg)
    stats["mean_kl"] = float((kl * resp).sum() / max(n_resp, 1))
    return trajs, stats
