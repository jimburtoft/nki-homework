"""Self-check for Lab 3. Run:  python labs/L3_matmul/check.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import nkihw  # noqa: E402,F401
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from nkihw.labkit import Exercise, run_lab  # noqa: E402

BF16 = ml_dtypes.bfloat16


def bf(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(BF16)


def f64(x):
    return x.astype(np.float64)


def mm_t(at, b):
    return (f64(at).T @ f64(b)).astype(np.float32)


TOL = {"rel_l2": 1e-4}

EXERCISES = [
    Exercise("3a", "matmul_single", lambda: (bf((128, 128), 30), bf((128, 512), 31)), mm_t, TOL),
    Exercise("3b", "matmul_tiled", lambda: (bf((1024, 256), 32), bf((1024, 1024), 33)), mm_t, TOL),
    Exercise("3c", "matmul_reuse", lambda: (bf((1024, 256), 32), bf((1024, 1024), 33)), mm_t, TOL, note="goal: beat 3b's estimate", max_est_us=29.0),
    Exercise("3d", "matmul_natural", lambda: (bf((128, 512), 34), bf((512, 512), 35)), lambda a, b: (f64(a) @ f64(b)).astype(np.float32), TOL),
    Exercise(
        "3e",
        "matmul_bias_relu",
        lambda: (bf((256, 128), 36), bf((256, 512), 37), np.random.default_rng(38).standard_normal((128, 1)).astype(np.float32)),
        lambda at, b, bias: np.maximum(f64(at).T @ f64(b) + bias, 0).astype(BF16),
        {"rel_l2": 1e-2},
    ),
]

if __name__ == "__main__":
    sys.exit(run_lab("Lab 3: Tensor Engine", EXERCISES, os.path.join(HERE, "lab.py")))
