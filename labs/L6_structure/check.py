"""Self-check for Lab 6. Run:  python labs/L6_structure/check.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import nkihw  # noqa: E402,F401
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from nkihw.labkit import Exercise, run_lab  # noqa: E402

BF16 = ml_dtypes.bfloat16
NEG = -30000.0
W = 48


def rng(s):
    return np.random.default_rng(s)


def window_ref(s):
    i = np.arange(128)[:, None]
    j = np.arange(128)[None, :]
    keep = (j <= i) & (i - j < W)
    return np.where(keep, s, np.float32(NEG)).astype(np.float32)


def causal_av_ref(qt, kt, v):
    s = qt.astype(np.float64).T @ kt.astype(np.float64)
    s = np.tril(s).astype(BF16).astype(np.float64)  # the kernel feeds S to the 2nd matmul in bf16
    return (s @ v.astype(np.float64)).astype(np.float32)


def recur_ref(a, b):
    h = np.zeros(a.shape[0])
    out = np.zeros(a.shape)
    for t in range(a.shape[1]):
        h = a[:, t].astype(np.float64) * h + b[:, t]
        out[:, t] = h
    return out.astype(np.float32)


EXERCISES = [
    Exercise("6a", "window_mask", lambda: (rng(60).standard_normal((128, 128)).astype(np.float32),), window_ref, {"exact": True}),
    Exercise(
        "6b",
        "causal_av",
        lambda: tuple(rng(s).standard_normal(sh).astype(BF16) for s, sh in ((61, (128, 1024)), (62, (128, 1024)), (66, (1024, 128)))),
        causal_av_ref,
        {"rel_l2": 1e-2},
        max_est_us=17.0,
    ),
    Exercise(
        "6c",
        "linear_recurrence",
        lambda: (rng(63).uniform(0.9, 1.0, (128, 1024)).astype(np.float32), rng(64).standard_normal((128, 1024)).astype(np.float32)),
        recur_ref,
        {"rel_l2": 1e-5},
    ),
    Exercise(
        "6d",
        "prefix_sum",
        lambda: (rng(65).standard_normal((1024, 256)).astype(np.float32),),
        lambda x: np.cumsum(x.astype(np.float64), axis=0).astype(np.float32),
        {"rel_l2": 1e-5},
    ),
]

if __name__ == "__main__":
    sys.exit(run_lab("Lab 6: structure", EXERCISES, os.path.join(HERE, "lab.py")))
