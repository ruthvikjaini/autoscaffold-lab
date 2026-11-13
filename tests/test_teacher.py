"""Tests for the teacher: diagnosis, validation decisions, q adaptation,
instance prefixes, and the Prop. 1 mixed-group curve."""

import numpy as np

from autoscaffold.tasks import make_problem, sample_problems
from autoscaffold.teacher import Teacher, task_key
from autoscaffold.vocab import CLASSES, EQ, MINUS, MOD, PLUS, TIMES, int_to_tokens


# ---------------------------------------------------------------- mock model
def _reference_segments(expr):
    """Reference derivation segments for an expression token list."""
    if len(expr) == 5:
        a, op1, b, op2, c = expr
        if op1 == PLUS and op2 == PLUS:
            return [a + b, a + b + c]
        if op1 == MINUS and op2 == MINUS:
            return [a - b, a - b - c]
        if op2 == MOD:
            return [a * b, (a * b) % c]
        # prec: multiply first
        if op1 == TIMES:
            return [a * b, a * b + c]
        return [b * c, a + b * c]
    a, op1, b, op2, c, op3, d = expr
    def app(op, x, y):
        return x + y if op == PLUS else (x - y if op == MINUS else x * y)
    s1 = app(op1, a, b)
    s2 = app(op2, s1, c)
    return [s1, s2, app(op3, s2, d)]


class MockModel:
    """Returns the reference response when the prompt contains a scaffold
    (word tokens), a wrong response otherwise — or the reverse, to test
    validation rejecting a harmful scaffold."""

    def __init__(self, scaffold_helps=True):
        self.scaffold_helps = scaffold_helps

    def generate(self, seqs, max_new, temperature=1.0, greedy=False, rng=None):
        out = []
        for seq in seqs:
            has_words = any(t >= 17 for t in seq)
            # strip leading word tokens, take expr up to the first '='
            expr = []
            i = 0
            while i < len(seq) and seq[i] >= 17:
                i += 1
            while i < len(seq) and seq[i] != EQ:
                expr.append(seq[i])
                i += 1
            # count revealed prefix segments (EQ-terminated groups after
            # the first EQ)
            rest = seq[i + 1:]
            n_revealed = sum(1 for t in rest if t == EQ)
            ref = _reference_segments(expr)
            correct = has_words if self.scaffold_helps else (not has_words)
            if correct:
                segs = ref[n_revealed:]
            else:
                segs = [s + 1 for s in ref[n_revealed:]]
            toks = []
            for s in segs:
                toks += int_to_tokens(s) + [EQ]
            toks[-1] = 15  # EOS
            out.append(toks)
        return out


def _block_log(cls="prec", n_groups=4, rewards=(0.0,)):
    """A fake block log: all-fail groups of one class, precedence mode."""
    rng = np.random.default_rng(0)
    problem = make_problem(cls, rng)
    log = []
    for _ in range(n_groups):
        trajs = [{"problem": problem, "prompt": [], "resp": [1, EQ, 2, 15],
                  "reward": r, "mode": "precedence", "n_seg_ok": 0,
                  "group": 0, "scaffolded": False} for r in rewards]
        log.append({"problem": problem, "class": cls,
                    "mean_reward": float(np.mean(rewards)),
                    "scaffolded": False, "trajs": trajs})
    return log


def test_diagnose_picks_dominant_failure():
    trajs = ([{"reward": 0.0, "mode": "precedence"}] * 5
             + [{"reward": 0.0, "mode": "wrong-final"}] * 2
             + [{"reward": 1.0, "mode": "correct"}])
    mode, counts = Teacher.diagnose(trajs)
    assert mode == "precedence"
    assert counts["precedence"] == 5


def test_teacher_adopts_helpful_scaffold():
    valid = sample_problems(4, seed=5)
    teacher = Teacher(valid, validate=True)
    model = MockModel(scaffold_helps=True)
    log = _block_log("prec")
    rec = teacher.update(model, log, block=0, rng=np.random.default_rng(0))
    assert teacher.strategies["prec"] is not None
    assert rec["classes"]["prec"]["decision"] == "adopt"
    assert rec["classes"]["prec"]["diagnosis"] == "precedence"


def test_validation_rejects_harmful_scaffold():
    # Canned V_C eval: weak class (v_none 0.25) but the proposed scaffold
    # is worse (v_new 0.10) — validation must reject it.
    valid = sample_problems(4, seed=5)
    teacher = Teacher(valid, validate=True)

    def fake_eval(model, cls, strategy, n=2, seed=0, greedy=False):
        return 0.10 if strategy is not None else 0.25
    teacher._eval_on_valid = fake_eval
    log = _block_log("prec")
    teacher.update(None, log, block=0, rng=np.random.default_rng(0))
    assert teacher.strategies["prec"] is None
    assert teacher.decisions[0]["classes"]["prec"]["decision"] == \
        "rejected-by-validation"


def test_no_validation_adopts_harmful_scaffold():
    valid = sample_problems(4, seed=5)
    teacher = Teacher(valid, validate=False)

    def fake_eval(model, cls, strategy, n=2, seed=0, greedy=False):
        return 0.10 if strategy is not None else 0.25
    teacher._eval_on_valid = fake_eval
    log = _block_log("prec")
    teacher.update(None, log, block=0, rng=np.random.default_rng(0))
    assert teacher.strategies["prec"] is not None  # deployed unchecked


def test_format_dominant_failure_gets_no_scaffold():
    valid = sample_problems(4, seed=5)
    teacher = Teacher(valid, validate=True)
    model = MockModel(scaffold_helps=True)
    log = _block_log("prec")
    for g in log:
        for t in g["trajs"]:
            t["mode"] = "format"
    teacher.update(model, log, block=0, rng=np.random.default_rng(0))
    assert teacher.strategies["prec"] is None


def test_q_stat_counts():
    valid = sample_problems(2, seed=5)
    teacher = Teacher(valid)
    log = _block_log("prec", n_groups=3, rewards=(1.0, 0.0, 1.0, 0.0,
                                                  1.0, 0.0, 1.0, 0.0))
    stats, _ = teacher.q_stat(log)
    assert stats["prec"]["n"] == 3
    assert stats["prec"]["n_mixed"] == 3
    assert stats["prec"]["n_all_fail"] == 0
    assert stats["add"] is None


def test_instance_prefix_grows_and_shrinks():
    valid = sample_problems(2, seed=5)
    teacher = Teacher(valid)
    rng = np.random.default_rng(0)
    problem = make_problem("prec", rng)
    key = task_key(problem)

    def log_for(reward):
        return [{"problem": problem, "class": "prec",
                 "mean_reward": reward, "scaffolded": False,
                 "trajs": []}]

    teacher._update_instance_prefixes(log_for(0.0))
    assert teacher.prefix_frac[key] == 0.5  # 2-step: one segment revealed
    teacher._update_instance_prefixes(log_for(0.0))
    assert teacher.prefix_frac[key] == 0.5  # capped: never reveals the answer
    teacher._update_instance_prefixes(log_for(1.0))
    assert key not in teacher.prefix_frac  # solved: prefix withdrawn


def test_mixed_group_probability_formula():
    """Prop. 1: u(p) = 1 - (1-p)^G - p^G, measured by simulation."""
    rng = np.random.default_rng(0)
    G = 8
    for p in (0.1, 0.25, 0.5, 0.75):
        draws = rng.random((20000, G)) < p
        mixed = ((draws.sum(axis=1) > 0) & (draws.sum(axis=1) < G)).mean()
        theory = 1 - (1 - p) ** G - p ** G
        assert abs(mixed - theory) < 0.02, (p, mixed, theory)
    # and the within-group variance peaks at p = 1/2 (Prop. 2)
    var = {p: (G - 1) / G * p * (1 - p) for p in (0.1, 0.5, 0.9)}
    assert var[0.5] > var[0.1] and var[0.5] > var[0.9]
