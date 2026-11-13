"""Tests for the task definitions, verifier, and failure taxonomy."""

import numpy as np

from autoscaffold.tasks import (
    classify, make_problem, parse_response, prompt_tokens, response_tokens,
    sample_problems,
)
from autoscaffold.vocab import EOS, EQ, STRATEGIES


def _resp_for(problem, segments):
    from autoscaffold.vocab import int_to_tokens
    toks = []
    for s in segments:
        toks += int_to_tokens(s) + [EQ]
    toks[-1] = EOS
    return toks


def test_correct_response_rewarded():
    rng = np.random.default_rng(0)
    for cls in ["add", "sub", "prec", "mod", "mix3"]:
        p = make_problem(cls, rng)
        resp = response_tokens(p)
        r, mode, n = classify(p, resp)
        assert r == 1.0 and mode == "correct" and n == len(p["segments"])


def test_precedence_detection():
    # a + b*c evaluated left-to-right: ((a+b)*c) is the precedence error.
    p = {"class": "prec", "expr": [2, 10, 3, 12, 4],
         "segments": [12, 14], "answer": 14}  # 2 + (3*4) = 14
    wrong = _resp_for(p, [5, 20])  # (2+3)=5, then 5*4=20
    r, mode, _ = classify(p, wrong)
    assert r == 0.0 and mode == "precedence"


def test_grouping_detection():
    # a - b - c regrouped as a - (b - c).
    p = {"class": "sub", "expr": [8, 11, 3, 11, 2],
         "segments": [5, 3], "answer": 3}
    wrong = _resp_for(p, [1, 7])  # b-c = 1, a-1 = 7
    r, mode, _ = classify(p, wrong)
    assert r == 0.0 and mode == "grouping"


def test_mod_detection():
    p = {"class": "mod", "expr": [3, 12, 4, 13, 5],
         "segments": [12, 2], "answer": 2}  # 3*4=12, 12 mod 5 = 2
    wrong = _resp_for(p, [12, 12])  # forgot the mod
    r, mode, _ = classify(p, wrong)
    assert r == 0.0 and mode == "mod"


def test_skipped_step_detection():
    p = {"class": "mix3", "expr": [2, 10, 3, 12, 4, 10, 1],
         "segments": [5, 20, 21], "answer": 21}
    # two segments, final answer right, intermediate skipped
    wrong = _resp_for(p, [20, 21])
    r, mode, _ = classify(p, wrong)
    assert r == 0.0 and mode == "skipped-step"


def test_format_errors():
    rng = np.random.default_rng(1)
    p = make_problem("add", rng)
    assert classify(p, [1, 2, 3])[1] == "format"  # no EOS
    assert classify(p, [1, EQ, 2, EOS])[1] in ("format", "wrong-intermediate",
                                               "wrong-final")
    assert classify(p, [])[1] == "format"


def test_prefix_prompt_response_consistency():
    rng = np.random.default_rng(2)
    for cls in ["add", "sub", "prec", "mod", "mix3"]:
        p = make_problem(cls, rng)
        n_seg = len(p["segments"])
        frac = 0.5 if n_seg == 2 else 1.0 / 3.0
        prompt = prompt_tokens(p, prefix_frac=frac)
        resp = response_tokens(p, prefix_frac=frac)
        # prompt + response must be the full derivation
        full = prompt_tokens(p) + response_tokens(p)
        assert prompt + resp == full
        # the final answer is never revealed in the prompt
        assert prompt_tokens(p, prefix_frac=1.0) == prompt_tokens(
            p, prefix_frac=(n_seg - 1) / n_seg)


def test_sample_problems_cover_all_classes():
    problems = sample_problems(10, seed=0)
    assert len(problems) == 50
    assert {p["class"] for p in problems} == {
        "add", "sub", "prec", "mod", "mix3"}
    assert all(len(p["segments"]) in (2, 3) for p in problems)
    # strategies exist for every class
    assert set(STRATEGIES) == {"add", "sub", "prec", "mod", "mix3"}
