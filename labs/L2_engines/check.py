"""Self-check for Lab 2. Run:  python labs/L2_engines/check.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import nkihw  # noqa: E402,F401
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from nkihw.labkit import Exercise, run_lab  # noqa: E402

BF16 = ml_dtypes.bfloat16


def f32(shape, seed, scale=1.0):
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


def rmsnorm_ref(x, w):
    x32 = x.astype(np.float64)
    y = x32 / np.sqrt((x32 * x32).mean(axis=1, keepdims=True) + 1e-6) * w.astype(np.float64)
    return y.astype(BF16)


def softmax_ref(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


EXERCISES = [
    Exercise(
        "2a",
        "row_stats",
        lambda: (f32((128, 1000), 20),),
        lambda x: (x.max(axis=1, keepdims=True), x.astype(np.float64).sum(axis=1, keepdims=True).astype(np.float32)),
        {"atol": 1e-4, "rtol": 1e-5},
    ),
    Exercise(
        "2b",
        "rmsnorm",
        lambda: (f32((512, 2048), 21).astype(BF16), (1 + 0.1 * f32((1, 2048), 22)).astype(BF16)),
        rmsnorm_ref,
        {"rel_l2": 1e-2},
    ),
    Exercise("2c", "softmax", lambda: (f32((256, 512), 23),), softmax_ref, {"atol": 1e-6, "rtol": 1e-3}, note="case 1: normal range"),
    Exercise("2c", "softmax", lambda: (f32((128, 512), 24, scale=100.0),), softmax_ref, {"atol": 1e-6, "rtol": 1e-3}, note="case 2: |x|~100"),
]

if __name__ == "__main__":
    sys.exit(run_lab("Lab 2: engines and reductions", EXERCISES, os.path.join(HERE, "lab.py")))
