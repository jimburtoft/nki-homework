"""P2 reference implementation and input generator (NumPy only).

Single-head sliding-window causal attention:

    S = (Q K^T) * scale,  scale = 1/sqrt(d)
    S[i, j] = -inf  unless  0 <= i - j < W        (query i sees itself and the W-1 keys before it)
    O = softmax(S, axis=-1) @ V

    q, k, v  [T, d]  bfloat16
    out      [T, d]  bfloat16

The reference computes in float64 from the bf16 inputs.
"""
import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16
TOL_REL_L2 = 2e-2


def make_inputs(T, d, W, seed, qscale=1.0):
    """qscale multiplies q; qscale ~ 25 makes S reach ~100 (tests softmax stability)."""
    r = np.random.default_rng(seed)
    q = (qscale * r.standard_normal((T, d))).astype(BF16)
    k = r.standard_normal((T, d)).astype(BF16)
    v = r.standard_normal((T, d)).astype(BF16)
    return q, k, v, int(W)


def reference(q, k, v, W):
    f = np.float64
    T, d = q.shape
    s = (q.astype(f) @ k.astype(f).T) / np.sqrt(d)
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    keep = (i - j >= 0) & (i - j < W)
    s = np.where(keep, s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return (p @ v.astype(f)).astype(BF16)
