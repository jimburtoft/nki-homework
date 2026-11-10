"""Self-check for Lab 7 (optional; trn3 target, simulator only).

Run:  python labs/L7_trn3/check.py
"""
import os
import sys

os.environ["NKIHW_TARGET"] = "trn3"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import nkihw  # noqa: E402,F401
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from nkihw.labkit import Exercise, run_lab  # noqa: E402

BF16 = ml_dtypes.bfloat16


def bf(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(BF16)


def mm_t(at, b):
    return at.astype(np.float64).T @ b.astype(np.float64)


EXERCISES = [
    Exercise("7a", "matmul_bf16_psum", lambda: (bf((128, 128), 70), bf((128, 512), 71)), lambda a, b: mm_t(a, b).astype(BF16), {"rel_l2": 1e-2}),
    Exercise("7b", "matmul_wide", lambda: (bf((128, 128), 72), bf((128, 4096), 73)), lambda a, b: mm_t(a, b).astype(np.float32), {"rel_l2": 1e-5}),
    # MXFP8 vs the exact bf16 product: ~3.8e-2 relative error for Gaussian data.
    Exercise("7c", "matmul_mxfp8", lambda: (bf((512, 128), 74), bf((512, 512), 75)), lambda a, b: mm_t(a, b).astype(np.float32), {"rel_l2": 6e-2}),
]

if __name__ == "__main__":
    # The compiler path is trn2-oriented in this course; for Lab 7 we report simulator results only.
    argv = sys.argv[1:] + (["--no-compile"] if "--compile" not in sys.argv else [])
    argv = [a for a in argv if a != "--compile"]
    sys.exit(run_lab("Lab 7: trn3 exploration (ungraded)", EXERCISES, os.path.join(HERE, "lab.py"), argv))
