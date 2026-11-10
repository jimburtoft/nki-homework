"""Self-check for Lab 0. Run:  python labs/L0_first_kernel/check.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import nkihw  # noqa: E402,F401
import numpy as np  # noqa: E402
from nkihw.labkit import Exercise, run_lab  # noqa: E402


def rng(seed=0):
    return np.random.default_rng(seed)


def x512():
    return (rng(1).standard_normal((128, 512)).astype(np.float32),)


EXERCISES = [
    Exercise("0a", "copy_kernel", x512, lambda x: x, {"exact": True}),
    Exercise("0b", "scale_shift_kernel", x512, lambda x: 3 * x + 1, {"atol": 1e-5, "rtol": 1e-6}),
    Exercise(
        "0c",
        "add_kernel",
        lambda: (rng(2).standard_normal((128, 512)).astype(np.float32), rng(3).standard_normal((128, 512)).astype(np.float32)),
        lambda a, b: a + b,
        {"exact": True},
    ),
    Exercise(
        "0d",
        "bias_relu_kernel",
        lambda: (rng(4).standard_normal((128, 512)).astype(np.float32), rng(5).standard_normal((128, 1)).astype(np.float32)),
        lambda x, b: np.maximum(2 * x + b, 0),
        {"atol": 1e-5, "rtol": 1e-5},
    ),
    Exercise("0e", "split_rows_kernel", x512, lambda x: 2 * x, {"exact": True}, note="passes in the simulator; make it compile"),
]

if __name__ == "__main__":
    sys.exit(run_lab("Lab 0: first kernels", EXERCISES, os.path.join(HERE, "lab.py")))
