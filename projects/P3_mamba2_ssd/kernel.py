"""
P3 -- Mamba-2 SSD chunked scan (hard project; prerequisites: Labs 0-6)
======================================================================

Implement the core of a Mamba-2 state-space layer (one head):

    h_t = exp(a_t) * h_{t-1} + B_t^T x_t          (state [N, P])
    y_t = C_t h_t

as ONE NKI kernel, using the chunked "state-space duality" form so that most of
the work runs on the Tensor Engine. Read SPEC.md first. Run the public tests with:

    python grader/grade.py P3
    python grader/grade.py P3 --no-compile    # correctness only (faster)

Do not change the function name or signature.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401  -- must be imported before nki

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402,F401
import nki.language as nl  # noqa: E402


@nki.jit
def ssd_kernel(x, a, B, C):
    """x [L, P] bf16, a [L, 1] fp32 (log-decay, <= 0), B [L, N] bf16, C [L, N] bf16
    -> y [L, P] fp32"""
    L, P = x.shape
    N = B.shape[1]
    y = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.shared_hbm)

    # YOUR CODE HERE

    return y
