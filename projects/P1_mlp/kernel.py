"""
P1 -- Fused MLP block (easy project; prerequisites: Labs 0-4)
=============================================================

Implement

    Y = gelu_tanh(X @ W1 + b1) @ W2 + b2

as ONE NKI kernel. Read SPEC.md for the shapes, the tolerance and how you are
graded. Run the public tests with:

    python grader/grade.py P1                 # correctness + compiler estimate
    python grader/grade.py P1 --no-compile    # correctness only (faster)
    python grader/grade.py P1 --file my_other_version.py

Do not change the function name or signature. You may add helper functions,
constants, and as many loops and tiles as you like. Everything must run inside
this one kernel: no NumPy pre- or post-processing of the inputs.
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
def mlp_kernel(x, w1, b1, w2, b2):
    """x [T, D] bf16, w1 [D, F] bf16, b1 [1, F] fp32, w2 [F, D] bf16, b2 [1, D] fp32
    -> y [T, D] bf16"""
    T, D = x.shape
    _, F = w1.shape
    y = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # YOUR CODE HERE

    return y
