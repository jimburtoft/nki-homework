"""
P2 -- Sliding-window causal attention (medium project; prerequisites: Labs 0-4 and 6)
=====================================================================================

Implement single-head attention where query i attends to keys i-W+1 .. i:

    O = softmax( mask_W( Q K^T / sqrt(d) ) ) V

as ONE NKI kernel. Read SPEC.md for the shapes, the tolerance and how you are
graded. Run the public tests with:

    python grader/grade.py P2
    python grader/grade.py P2 --no-compile    # correctness only (faster)

Do not change the function name or signature. `W` is a Python int (a
compile-time constant), so you can use it in Python control flow.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401  -- must be imported before nki

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402,F401
import nki.language as nl  # noqa: E402

P = 128


@nki.jit
def window_attention(q, k, v, W):
    """q, k, v [T, d] bf16; W int window length (query i sees keys i-W+1 .. i)
    -> out [T, d] bf16"""
    T, d = q.shape
    out = nl.ndarray((T, d), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # YOUR CODE HERE

    return out
