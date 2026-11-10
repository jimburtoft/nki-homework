"""P1 reference implementation and input generator (NumPy only).

Y = gelu_tanh(X @ W1 + b1) @ W2 + b2

    X  [T, D]  bfloat16      tokens x model dim
    W1 [D, F]  bfloat16      up projection
    b1 [1, F]  float32
    W2 [F, D]  bfloat16      down projection
    b2 [1, D]  float32
    Y  [T, D]  bfloat16

The reference computes in float64 from the bf16 inputs. The hidden activation H
is rounded to bfloat16 before the second matmul, as a bf16 kernel would do;
either rounding choice fits within the tolerance.
"""
import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16
TOL_REL_L2 = 2e-2


def make_inputs(T, D, F, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((T, D)).astype(BF16)
    w1 = (r.standard_normal((D, F)) / np.sqrt(D)).astype(BF16)
    b1 = (0.1 * r.standard_normal((1, F))).astype(np.float32)
    w2 = (r.standard_normal((F, D)) / np.sqrt(F)).astype(BF16)
    b2 = (0.1 * r.standard_normal((1, D))).astype(np.float32)
    return x, w1, b1, w2, b2


def gelu_tanh(z):
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))


def reference(x, w1, b1, w2, b2):
    f = np.float64
    h = gelu_tanh(x.astype(f) @ w1.astype(f) + b1.astype(f))
    h = h.astype(BF16).astype(f)
    y = h @ w2.astype(f) + b2.astype(f)
    return y.astype(BF16)
