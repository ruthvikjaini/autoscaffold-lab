"""Vocabulary for the autoscaffold arithmetic testbed.

Arithmetic tokens (0-16) plus a small word vocabulary (17-46) for the
plain-text scaffolds. Scaffold strategies are short fixed word
sequences; the model only ever generates arithmetic tokens (digits,
'=', EOS) in its response, so scaffold words appear in prompts only.
"""

# --- arithmetic tokens ---
DIGITS = list(range(10))  # token id == digit value, ids 0..9
PLUS = 10
MINUS = 11
TIMES = 12
MOD = 13
EQ = 14
EOS = 15
PAD = 16

# --- scaffold word tokens ---
ADD_W = 17
SUBTRACT = 18
MULTIPLY = 19
BEFORE = 20
ADDING = 21
LEFT = 22
TO = 23
RIGHT = 24
WRITE = 25
EACH = 26
RESULT = 27
NEVER = 28
REGROUP = 29
FIND = 30
THE = 31
MULTIPLICATION = 32
FIRST = 33
ITS = 34
THEN = 35
TAKE = 36
REMAINDER = 37
PRODUCT = 38
DO = 39
ONE = 40
OPERATION = 41
AT = 42
A = 43
TIME = 44
NEXT = 45
STEP = 46

VOCAB_SIZE = 47

TOKEN_NAMES = (
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    + ["+", "-", "*", "mod", "=", "EOS", "PAD"]
    + ["add", "subtract", "multiply", "before", "adding", "left", "to",
       "right", "write", "each", "result", "never", "regroup", "find",
       "the", "multiplication", "first", "its", "then", "take",
       "remainder", "product", "do", "one", "operation", "at", "a",
       "time", "next", "step"]
)


def decode(tokens):
    """Render a token list as a readable string (PAD/EOS dropped)."""
    return " ".join(TOKEN_NAMES[t] for t in tokens if t not in (PAD, EOS))


def int_to_tokens(value):
    """Non-negative int -> digit tokens; negative ints get a leading MINUS."""
    if value < 0:
        return [MINUS] + int_to_tokens(-value)
    return [int(c) for c in str(value)]


def tokens_to_int(tokens):
    """Digit tokens (optionally MINUS-led) -> int, or None if malformed."""
    if not tokens:
        return None
    neg = False
    if tokens[0] == MINUS:
        neg = True
        tokens = tokens[1:]
    if not tokens or any(t not in DIGITS for t in tokens):
        return None
    value = 0
    for t in tokens:
        value = value * 10 + t
    return -value if neg else value


# ---------------------------------------------------------------- scaffolds
# The teacher's strategy library: one workflow strategy per class, each
# addressing the class's designed failure modes (see tasks.py).
STRATEGIES = {
    "add": [ADD_W, LEFT, TO, RIGHT, WRITE, EACH, RESULT],
    "sub": [SUBTRACT, LEFT, TO, RIGHT, NEVER, REGROUP, WRITE, EACH, RESULT],
    "prec": [MULTIPLY, BEFORE, ADDING, FIND, THE, MULTIPLICATION, FIRST,
             WRITE, ITS, RESULT, THEN, ADD_W],
    "mod": [MULTIPLY, FIRST, THEN, TAKE, THE, REMAINDER, WRITE, THE,
            PRODUCT, THEN, THE, REMAINDER],
    "mix3": [DO, ONE, OPERATION, AT, A, TIME, WRITE, EACH, RESULT,
             BEFORE, THE, NEXT, STEP],
}

# Failure modes each class strategy is designed to address. The teacher
# only deploys a class strategy when the class's dominant failure is in
# this set. "wrong-final" is included everywhere: the workflow strategy
# covers the second operation too, which is where final-step slips
# happen. A "format" dominant failure gets no scaffold — an arithmetic
# strategy cannot fix malformed output.
STRATEGY_TARGETS = {
    "add": {"wrong-intermediate", "wrong-final"},
    "sub": {"grouping", "wrong-intermediate", "wrong-final"},
    "prec": {"precedence", "wrong-intermediate", "wrong-final"},
    "mod": {"mod", "wrong-intermediate", "wrong-final"},
    "mix3": {"skipped-step", "wrong-intermediate", "wrong-final"},
}

CLASSES = ["add", "sub", "prec", "mod", "mix3"]
