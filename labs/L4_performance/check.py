"""Self-check for Lab 4. Run:  python labs/L4_performance/check.py

The goals are compiler estimates for neuronx-cc 2.28 (Neuron SDK 2.33). Other
compiler versions may give different numbers.
"""
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


def gelu_tanh(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


EXERCISES = [
    Exercise("4a", "double_big_dma", lambda: (f32((1024, 4096), 40),), lambda x: 2 * x, {"exact": True}, max_est_us=120),
    Exercise("4b", "exp_overlap", lambda: (f32((2048, 2048), 41),), lambda x: np.exp(x.astype(np.float64)), {"atol": 1e-5, "rtol": 1e-4}, max_est_us=130),
    Exercise(
        "4c",
        "scaled_gelu",
        lambda: (f32((2048, 2048), 42),),
        lambda x: gelu_tanh(3 * x.astype(np.float64)),
        {"atol": 1e-4, "rtol": 1e-3},
        max_est_us=130,
    ),
    Exercise(
        "4d",
        "proj_relu",
        lambda: (f32((128, 4096), 43).astype(BF16), f32((128, 512), 44).astype(BF16)),
        lambda xt, w: np.maximum(xt.astype(np.float64).T @ w.astype(np.float64), 0).astype(BF16),
        {"rel_l2": 1e-2},
        max_est_us=45,
    ),
]

if __name__ == "__main__":
    sys.exit(run_lab("Lab 4: performance", EXERCISES, os.path.join(HERE, "lab.py")))
