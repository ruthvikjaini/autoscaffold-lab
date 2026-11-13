"""Task definitions, reference derivations, verifier, failure taxonomy.

Five classes of multi-step arithmetic. The model must emit a derivation
``<s1> = <s2> = <ans> EOS`` (two segments for the 2-step classes, three
for mix3). The verifier returns the binary reward plus a failure-mode
label from a fixed taxonomy, which is what the teacher diagnoses.
"""

import numpy as np

from .vocab import (
    CLASSES, DIGITS, EQ, EOS, MINUS, MOD, PLUS, TIMES, int_to_tokens,
    tokens_to_int,
)

# ---------------------------------------------------------------- problems
# A problem is a dict: {"class": str, "operands": tuple, "ops": tuple,
# "expr": [tokens], "segments": [[tokens], ...], "answer": int}.
# segments are the reference derivation values (s1, s2?, ans).


def make_problem(cls, rng):
    if cls == "add":
        a, b, c = (int(rng.integers(0, 10)) for _ in range(3))
        s1 = a + b
        ans = s1 + c
        expr = [a, PLUS, b, PLUS, c]
        segments = [s1, ans]
    elif cls == "sub":
        a, b, c = (int(rng.integers(0, 10)) for _ in range(3))
        s1 = a - b
        ans = s1 - c
        expr = [a, MINUS, b, MINUS, c]
        segments = [s1, ans]
    elif cls == "prec":
        a, b, c = (int(rng.integers(0, 10)) for _ in range(3))
        if rng.random() < 0.5:
            s1 = b * c
            ans = a + s1
            expr = [a, PLUS, b, TIMES, c]
        else:
            s1 = a * b
            ans = s1 + c
            expr = [a, TIMES, b, PLUS, c]
        segments = [s1, ans]
    elif cls == "mod":
        a, b = (int(rng.integers(0, 10)) for _ in range(2))
        m = int(rng.integers(3, 10))
        s1 = a * b
        ans = s1 % m
        expr = [a, TIMES, b, MOD, m]
        segments = [s1, ans]
    elif cls == "mix3":
        a, b, c, d = (int(rng.integers(0, 10)) for _ in range(4))
        op1, op2, op3 = (rng.choice([PLUS, MINUS, TIMES]) for _ in range(3))
        s1 = _apply(op1, a, b)
        s2 = _apply(op2, s1, c)
        ans = _apply(op3, s2, d)
        expr = [a, op1, b, op2, c, op3, d]
        segments = [s1, s2, ans]
    else:
        raise ValueError(cls)
    return {"class": cls, "operands": expr, "expr": expr,
            "segments": segments, "answer": ans}


def _apply(op, x, y):
    if op == PLUS:
        return x + y
    if op == MINUS:
        return x - y
    return x * y


def sample_problems(n_per_class, seed):
    """Deterministic pool: n_per_class problems per class, shuffled."""
    rng = np.random.default_rng(seed)
    out = []
    for cls in CLASSES:
        for _ in range(n_per_class):
            out.append(make_problem(cls, rng))
    rng.shuffle(out)
    return out


def prompt_tokens(problem, strategy=None, prefix_frac=0.0):
    """Prompt = [strategy words] + expr + '=' [+ revealed prefix segments].

    prefix_frac is the fraction of the reference derivation revealed:
    0.0 reveals nothing; for a 2-step problem 0.5 reveals s1; for a
    3-step problem 1/3 reveals s1 and 2/3 reveals s1, s2. The final
    answer is never revealed (prefix_frac is capped at
    (n_segments - 1) / n_segments).
    """
    toks = list(strategy) if strategy else []
    toks += list(problem["expr"]) + [EQ]
    n_seg = len(problem["segments"])
    max_frac = (n_seg - 1) / n_seg
    frac = min(prefix_frac, max_frac)
    n_reveal = int(round(frac * n_seg))
    n_reveal = min(n_reveal, n_seg - 1)
    for seg in problem["segments"][:n_reveal]:
        toks += int_to_tokens(seg) + [EQ]
    return toks


def response_tokens(problem, prefix_frac=0.0):
    """Reference response for the unrevealed part of the derivation."""
    n_seg = len(problem["segments"])
    max_frac = (n_seg - 1) / n_seg
    frac = min(prefix_frac, max_frac)
    n_reveal = int(round(frac * n_seg))
    n_reveal = min(n_reveal, n_seg - 1)
    toks = []
    for seg in problem["segments"][n_reveal:]:
        toks += int_to_tokens(seg) + [EQ]
    toks[-1] = EOS  # replace the trailing '=' with EOS
    return toks


# ---------------------------------------------------------------- verifier
def parse_response(resp):
    """Split a generated response on '=' into integer segments.

    Returns (segments, ok): ok is False on malformed input (empty
    segments, non-digit tokens, missing EOS, trailing tokens after EOS).
    """
    if not resp or resp[-1] != EOS:
        return None, False
    body = resp[:-1]
    if not body:
        return None, False
    parts, cur = [], []
    for t in body:
        if t == EQ:
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    parts.append(cur)
    segments = []
    for p in parts:
        v = tokens_to_int(p)
        if v is None:
            return None, False
        segments.append(v)
    return segments, True


def _alt_segments(problem):
    """Segments of the plausible wrong derivations the taxonomy checks."""
    cls = problem["class"]
    expr = problem["expr"]
    if cls == "prec":
        a, op1, b, op2, c = expr
        # left-to-right evaluation (ignores precedence)
        s1 = _apply(op1, a, b)
        return [s1, _apply(op2, s1, c)]
    if cls == "sub":
        a, _, b, _, c = expr
        return [b - c, a - (b - c)]  # a-(b-c) regrouping
    if cls == "mod":
        a, _, b, _, m = expr
        return [a * b, a * b]  # forgot the mod entirely
    return None


def classify(problem, resp):
    """Classify a generated response.

    Returns (reward, failure_mode, n_segments_correct) where failure
    mode is one of: correct, format, wrong-intermediate, wrong-final,
    precedence, grouping, mod, skipped-step.
    """
    segments, ok = parse_response(resp)
    if not ok or len(segments) != len(problem["segments"]):
        # mix3 with two segments and the right final answer skipped a
        # written intermediate rather than failing the format.
        if (problem["class"] == "mix3" and ok and len(segments) == 2
                and segments[-1] == problem["answer"]):
            return 0.0, "skipped-step", 0
        return 0.0, "format", 0
    ref = problem["segments"]
    if segments == ref:
        return 1.0, "correct", len(ref)
    n_ok = sum(1 for s, r in zip(segments, ref) if s == r)
    alt = _alt_segments(problem)
    if alt is not None and segments == alt:
        mode = {"prec": "precedence", "sub": "grouping",
                "mod": "mod"}[problem["class"]]
        return 0.0, mode, n_ok
    if segments[0] != ref[0]:
        return 0.0, "wrong-intermediate", n_ok
    return 0.0, "wrong-final", n_ok


def reward(problem, resp):
    r, _, _ = classify(problem, resp)
    return r
