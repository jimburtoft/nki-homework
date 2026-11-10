"""Self-check for Lab 1. Run:  python labs/L1_tiling/check.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import nkihw  # noqa: E402,F401
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from nkihw.labkit import Exercise, run_lab  # noqa: E402

BF16 = ml_dtypes.bfloat16


def f32(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def affine(x):
    return x * np.float32(0.5) + np.float32(2.0)


EXERCISES = [
    Exercise("1a", "tiled_affine", lambda: (f32((1024, 512), 10),), affine, {"atol": 1e-6, "rtol": 1e-6}),
    Exercise("1b", "tiled_affine_any_m", lambda: (f32((300, 512), 11),), affine, {"atol": 1e-6, "rtol": 1e-6}, note="M=300"),
    Exercise("1b", "tiled_affine_any_m", lambda: (f32((129, 256), 12),), affine, {"atol": 1e-6, "rtol": 1e-6}, note="M=129"),
    Exercise("1b", "tiled_affine_any_m", lambda: (f32((64, 256), 13),), affine, {"atol": 1e-6, "rtol": 1e-6}, note="M=64"),
    Exercise("1c", "square_wide", lambda: (f32((128, 65536), 14),), lambda x: x * x, {"atol": 1e-6, "rtol": 1e-6}),
    Exercise(
        "1d",
        "exp_bf16",
        lambda: (f32((256, 1024), 15).astype(BF16),),
        lambda x: np.exp(x.astype(np.float32)).astype(BF16),
        {"rel_l2": 1e-2},
    ),
]

if __name__ == "__main__":
    sys.exit(run_lab("Lab 1: tiling", EXERCISES, os.path.join(HERE, "lab.py")))
