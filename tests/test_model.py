"""Tests for the transformer: gradient check, generation, save/load."""

import numpy as np

from autoscaffold.model import Transformer
from autoscaffold.vocab import EOS, PAD, VOCAB_SIZE


def _small_model(seed=0):
    return Transformer(VOCAB_SIZE, max_len=32, d=16, seed=seed)


def test_gradient_finite_differences():
    rng = np.random.default_rng(0)
    m = _small_model()
    B, T = 2, 8
    tokens = rng.integers(0, VOCAB_SIZE, size=(B, T))
    mask = np.zeros((B, T))
    mask[:, 3:] = 1.0
    loss, grads, _ = m.loss_and_grads(tokens, mask)
    # check a few entries of two parameter tensors
    for key in ["Wq", "W2", "Wte"]:
        P = m.params[key]
        G = grads[key]
        idx = [(0, 0), (1, 2), (3, 1)] if P.ndim == 2 else [(0,), (2,)]
        for ix in idx:
            eps = 1e-5
            old = P[ix]
            P[ix] = old + eps
            lp, _, _ = m.loss_and_grads(tokens, mask)
            P[ix] = old - eps
            lm, _, _ = m.loss_and_grads(tokens, mask)
            P[ix] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - G[ix]) < 1e-4, (key, ix, num, G[ix])


def test_loss_decreases_on_sft_batch():
    from autoscaffold.sft import _batch_arrays, sft_example
    from autoscaffold.tasks import sample_problems
    rng = np.random.default_rng(0)
    problems = sample_problems(20, seed=0)
    m = _small_model()
    examples = [sft_example(p, rng) for p in problems]
    tokens, mask = _batch_arrays(examples)
    l0, g0, _ = m.loss_and_grads(tokens, mask)
    for _ in range(30):
        loss, grads, _ = m.loss_and_grads(tokens, mask)
        m.adam_step(grads, 1e-2)
    l1, _, _ = m.loss_and_grads(tokens, mask)
    assert l1 < l0


def test_generate_stops_at_eos():
    m = _small_model()
    seqs = [[1, 2, 3], [4, 5]]
    out = m.generate(seqs, max_new=10, greedy=True,
                     rng=np.random.default_rng(0))
    assert len(out) == 2
    for o in out:
        assert len(o) <= 10
        assert EOS not in o[:-1]  # never generates past EOS


def test_save_load_roundtrip(tmp_path):
    m = _small_model()
    path = str(tmp_path / "m.npz")
    m.save(path)
    m2 = Transformer.load(path, VOCAB_SIZE, max_len=32, d=16)
    for k in m.params:
        assert np.allclose(m.params[k], m2.params[k])
