"""Tests for GRPO: advantage standardisation, KL-sign regression,
no-signal skip, and the rewarded-logprob property."""

import numpy as np

from autoscaffold.grpo import group_advantages, grpo_update, rollout_batch
from autoscaffold.model import Transformer
from autoscaffold.tasks import make_problem, prompt_tokens, sample_problems
from autoscaffold.vocab import VOCAB_SIZE


def _models(seed=0):
    m = Transformer(VOCAB_SIZE, max_len=64, d=32, seed=seed)
    ref = m.copy()
    return m, ref


def test_group_advantages_standardised():
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    groups = np.array([0] * 4 + [1] * 4)
    adv = group_advantages(rewards, groups)
    # group 0: mean 0.5, std 0.5 -> +1/-1
    assert np.allclose(adv[:4], [1, -1, 1, -1])
    # group 1: degenerate (all 1) -> 0
    assert np.allclose(adv[4:], 0.0)


def _symmetric_kl(m, ref, tokens):
    lp = m.logprobs(tokens)
    lpr = ref.logprobs(tokens)
    return float(np.mean(np.abs(lp - lpr)))


def test_kl_penalty_pulls_toward_reference():
    """Regression test (Day 15/17 lesson): the KL term must act as a
    penalty, not an anti-regulariser. Starting from the reference, 10
    GRPO updates with beta=+0.5 must keep the symmetric KL small,
    while the same updates with beta=-0.5 (the flipped sign) must
    drive it far away. If the formula's sign were flipped, the +0.5
    run would behave like the -0.5 run."""
    rng = np.random.default_rng(0)
    # add-only warm-start (see the KL test): mixed groups, real signal.
    problems = [make_problem("add", rng) for _ in range(200)]
    m, ref = _models()
    from autoscaffold.sft import train_sft
    train_sft(m, problems, steps=300, batch=32, lr=3e-3, rng=rng,
              log_every=0)
    ref = m.copy()
    fresh = [make_problem("add", rng) for _ in range(4)]
    fresh_prompts = [prompt_tokens(p) for p in fresh]
    trajs = rollout_batch(m, fresh, fresh_prompts, group=4, rng=rng)
    seqs = [t["prompt"] + t["resp"] for t in trajs]
    W = max(len(s) for s in seqs)
    from autoscaffold.vocab import PAD
    seqs = np.array([s + [PAD] * (W - len(s)) for s in seqs])

    def run(beta):
        model = ref.copy()
        for _ in range(10):
            _, stats = grpo_update(model, ref, fresh, fresh_prompts,
                                   group=8, lr=3e-4, beta=beta, rng=rng)
            assert stats["signal_frac"] > 0.0, "batch carried no signal"
        return _symmetric_kl(model, ref, seqs)

    k_pos = run(0.5)
    k_neg = run(-0.5)
    assert k_pos < 0.5, k_pos
    assert k_neg > k_pos, (k_pos, k_neg)


def test_grpo_raises_rewarded_logprob():
    """A GRPO update raises the rewarded trajectory's logprob relative
    to the unrewarded one (Day 17 lesson: the old params-moved test
    could not see a sign error)."""
    rng = np.random.default_rng(1)
    # add-only warm-start (see the KL test): mixed groups, real signal.
    problems = [make_problem("add", rng) for _ in range(200)]
    m, ref = _models(seed=1)
    from autoscaffold.sft import train_sft
    train_sft(m, problems, steps=300, batch=32, lr=3e-3, rng=rng,
              log_every=0)
    ref = m.copy()
    fresh = [make_problem("add", rng) for _ in range(4)]
    prompts = [prompt_tokens(p) for p in fresh]
    trajs = rollout_batch(m, fresh, prompts, group=8, rng=rng)
    # keep only groups with a mixed outcome so the signal is real
    mixed = [g for g in np.unique([t["group"] for t in trajs])
             if len({t["reward"] for t in trajs if t["group"] == g}) > 1]
    if not mixed:
        return  # no mixed group this seed; the property is vacuous
    g = mixed[0]
    good = [t for t in trajs if t["group"] == g and t["reward"] == 1.0][0]
    bad = [t for t in trajs if t["group"] == g and t["reward"] == 0.0][0]

    def lp_of(t):
        seq = np.array([t["prompt"] + t["resp"]])
        lp = m.logprobs(seq)[0]
        n_prompt = len(t["prompt"])
        return float(lp[n_prompt - 1:].sum())

    lp_good0, lp_bad0 = lp_of(good), lp_of(bad)
    for _ in range(3):
        grpo_update(m, ref, fresh, prompts, group=8, lr=3e-4, beta=0.0,
                    rng=rng)
    assert lp_of(good) - lp_good0 > lp_of(bad) - lp_bad0


def test_no_signal_update_skipped():
    """When every group is degenerate the update must not move params."""
    rng = np.random.default_rng(2)
    problems = sample_problems(2, seed=2)
    m, ref = _models(seed=2)
    prompts = [prompt_tokens(p) for p in problems]
    before = {k: v.copy() for k, v in m.params.items()}
    # beta=0 and a degenerate batch: use greedy rollouts (all identical)
    trajs, stats = grpo_update(m, ref, problems, prompts, group=4,
                               lr=3e-4, beta=0.0, rng=rng, temperature=1e-6)
    if stats["signal_frac"] == 0.0:
        assert stats["skipped"]
        for k in m.params:
            assert np.array_equal(m.params[k], before[k])
