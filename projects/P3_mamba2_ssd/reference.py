"""P3 reference implementation and input generator (NumPy only).

Mamba-2 state-space duality (SSD) layer core, one head, scalar decay per step:

    state  h_t = exp(a_t) * h_{t-1} + B_t^T x_t        h_t in R^{N x P}   (outer product B_t x_t^T)
    output y_t = C_t h_t                                y_t in R^{P}

Equivalently, the "dual" (attention-like) form

    y_t = sum_{s <= t}  (C_t . B_s) * exp(sum_{r=s+1..t} a_r) * x_s

    x   [L, P]   bfloat16      inputs (P = head dim)
    a   [L, 1]   float32       log-decay per step, a_t <= 0  (decay factor exp(a_t) in (0, 1])
    B   [L, N]   bfloat16      input projection  (N = state dim)
    C   [L, N]   bfloat16      output projection
    y   [L, P]   float32

The reference uses the recurrent form in float64.
"""
import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16
TOL_REL_L2 = 2e-2


def make_inputs(L, P, N, seed, decay=1.0):
    """decay scales the log-decay magnitude: a_t = -decay * softplus(z).
    Large decay (e.g. 20) makes exp(cumsum(a)) underflow to 0 over a chunk,
    so exp(cumA_t) / exp(cumA_s) is 0/0 for a naive implementation."""
    r = np.random.default_rng(seed)
    x = r.standard_normal((L, P)).astype(BF16)
    z = r.standard_normal((L, 1)) - 2.0
    a = (-decay * np.log1p(np.exp(z))).astype(np.float32)
    B = (r.standard_normal((L, N)) / np.sqrt(N)).astype(BF16)
    C = (r.standard_normal((L, N)) / np.sqrt(N)).astype(BF16)
    return x, a, B, C


def reference(x, a, B, C):
    f = np.float64
    L, P = x.shape
    N = B.shape[1]
    x, a, B, C = x.astype(f), a.astype(f)[:, 0], B.astype(f), C.astype(f)
    h = np.zeros((N, P))
    y = np.zeros((L, P))
    for t in range(L):
        h = np.exp(a[t]) * h + np.outer(B[t], x[t])
        y[t] = C[t] @ h
    return y.astype(np.float32)
