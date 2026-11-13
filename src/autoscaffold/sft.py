"""SFT warm-start for the autoscaffold student.

Training examples mix three prompt forms so the student learns to use
scaffolds when they appear (and to continue a revealed solution prefix)
without depending on them: plain prompts (majority), class-strategy
prompts (oracle strategy for the class), and instance-prefix prompts
(first segment(s) of the reference derivation revealed).
"""

import numpy as np

from .tasks import prompt_tokens, response_tokens
from .vocab import PAD, STRATEGIES


def sft_example(problem, rng, p_strategy=0.3, p_prefix=0.1):
    """One (tokens, mask) SFT example; mask marks response positions."""
    u = rng.random()
    if u < p_prefix:
        n_seg = len(problem["segments"])
        prefix_frac = 0.5 if n_seg == 2 else 1.0 / 3.0
        prompt = prompt_tokens(problem, strategy=None, prefix_frac=prefix_frac)
        resp = response_tokens(problem, prefix_frac=prefix_frac)
    elif u < p_prefix + p_strategy:
        prompt = prompt_tokens(problem, strategy=STRATEGIES[problem["class"]])
        resp = response_tokens(problem)
    else:
        prompt = prompt_tokens(problem)
        resp = response_tokens(problem)
    tokens = prompt + resp
    mask = [0.0] * len(prompt) + [1.0] * len(resp)
    return tokens, mask


def _batch_arrays(examples):
    W = max(len(t) for t, _ in examples)
    B = len(examples)
    tokens = np.full((B, W), PAD, dtype=int)
    mask = np.zeros((B, W), dtype=float)
    for i, (t, m) in enumerate(examples):
        tokens[i, :len(t)] = t
        mask[i, :len(m)] = m
    return tokens, mask


def train_sft(model, problems, steps=1200, batch=32, lr=3e-3,
              label_smoothing=0.05, rng=None, log_every=200,
              p_strategy=0.3, p_prefix=0.1):
    """Train the warm-start. Returns a list of (step, loss) pairs."""
    rng = rng or np.random.default_rng(0)
    model.reset_optimizer()
    history = []
    for step in range(steps):
        idx = rng.integers(0, len(problems), size=batch)
        examples = [sft_example(problems[i], rng, p_strategy, p_prefix)
                    for i in idx]
        tokens, mask = _batch_arrays(examples)
        loss, grads, _ = model.loss_and_grads(tokens, mask,
                                              label_smoothing=label_smoothing)
        model.adam_step(grads, lr)
        history.append((step, float(loss)))
        if log_every and (step + 1) % log_every == 0:
            print(f"  sft {step+1}/{steps} loss {loss:.4f}", flush=True)
    return history
