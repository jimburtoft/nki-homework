"""Self-check for Lab 5. Run:  python labs/L5_precision/check.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import nkihw  # noqa: E402,F401
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
from nkihw.labkit import Exercise, run_lab  # noqa: E402

BF16 = ml_dtypes.bfloat16
FP8 = ml_dtypes.float8_e4m3


def rng(seed):
    return np.random.default_rng(seed)


def long_k(K, seed):
    def f():
        r = rng(seed)
        # positive-mean data: partial sums grow, so bf16 accumulation error grows with K
        return (r.standard_normal((K, 128)) + 0.5).astype(BF16), (r.standard_normal((K, 512)) + 0.5).astype(BF16)
    return f


def mm_t(at, b):
    return (at.astype(np.float64).T @ b.astype(np.float64)).astype(np.float32)


def fp8_ref(at, b):
    exact = at.astype(np.float64).T @ b.astype(np.float64)
    q = at.astype(FP8).astype(np.float64).T @ b.astype(FP8).astype(np.float64)
    print(f"   info    fp8 vs exact bf16 product: rel_l2 = {np.linalg.norm(q - exact) / np.linalg.norm(exact):.3e}")
    return q.astype(np.float32)


def sr_inputs():
    r = rng(53)
    x = (1.0 + r.random((128, 512)) * 2**-6).astype(np.float32)
    bits = r.integers(0, 1 << 16, (128, 512)).astype(np.uint32)
    return x, bits


def sr_ref(x, bits):
    return ((x.view(np.uint32) + bits) & np.uint32(0xFFFF0000)).view(np.float32).astype(BF16)


def acc_ref(x):
    return np.full(x.shape, 1.5, dtype=np.float64)  # 1 + 256 * 2**-9


def mean_near(out, ref):
    from nkihw import CheckResult

    o = np.asarray(out).astype(np.float64)
    if not np.isfinite(o).all():
        return CheckResult(False, "output contains NaN/inf (unwritten output?)")
    m, target = float(o.mean()), float(ref.mean())
    ok = abs(m - target) <= 0.01
    return CheckResult(ok, f"mean = {m:.4f} (target {target}, tolerance 0.01); per-element std = {o.std():.4f}")


EXERCISES = [
    Exercise("5a", "matmul_long_k", long_k(512, 50), mm_t, {"rel_l2": 1e-5}, note="K=512"),
    Exercise("5a", "matmul_long_k", long_k(8192, 51), mm_t, {"rel_l2": 1e-5}, note="K=8192"),
    # 1e-3, not tighter: trn2 hardware differs from the simulator by ~1e-4 here (fp8 product
    # accumulation order). A wrong layout gives ~1.0; plain fp8 vs bf16 is ~4e-2.
    Exercise("5b", "matmul_fp8_double_row", lambda: (rng(52).standard_normal((256, 128)).astype(BF16), rng(54).standard_normal((256, 512)).astype(BF16)), fp8_ref, {"rel_l2": 1e-3}),
    Exercise(
        "5c",
        "four_small_matmuls",
        lambda: (rng(55).standard_normal((4, 32, 128)).astype(BF16), rng(56).standard_normal((4, 32, 512)).astype(BF16)),
        lambda a, b: np.einsum("gkm,gkn->gmn", a.astype(np.float64), b.astype(np.float64)).astype(np.float32),
        {"rel_l2": 1e-5},
    ),
    Exercise("5d", "sr_round", sr_inputs, sr_ref, {"exact": True}),
    Exercise("5e", "sr_accumulate", lambda: (np.zeros((128, 512), np.float32),), acc_ref, custom=mean_near, note="exact answer 1.5"),
]

if __name__ == "__main__":
    sys.exit(run_lab("Lab 5: precision", EXERCISES, os.path.join(HERE, "lab.py")))
