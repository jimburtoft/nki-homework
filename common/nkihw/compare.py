"""Comparing a kernel output to a reference."""
from dataclasses import dataclass

import numpy as np


@dataclass
class CheckResult:
    ok: bool
    msg: str
    rel_l2: float = float("nan")
    max_abs: float = float("nan")

    def __bool__(self):
        return self.ok

    def __str__(self):
        return ("PASS " if self.ok else "FAIL ") + self.msg


def check(out, ref, *, rel_l2=None, atol=None, rtol=0.0, exact=False, name="output") -> CheckResult:
    """Compare ``out`` to ``ref``.

    The simulator fills memory a kernel never wrote with NaN, so an output with
    NaNs almost always means "part of the output was never stored".

    Modes (pick one; ``exact`` wins, then ``rel_l2``, then ``atol``/``rtol``):
      exact   -- bit-for-bit equal
      rel_l2  -- ||out - ref|| / ||ref|| <= rel_l2
      atol    -- |out - ref| <= atol + rtol * |ref| elementwise
    """
    if out is None:
        return CheckResult(False, f"{name}: kernel returned None (did you forget `return`?)")
    out = np.asarray(out)
    ref = np.asarray(ref)
    if out.shape != ref.shape:
        return CheckResult(False, f"{name}: shape {out.shape} != expected {ref.shape}")
    o = out.astype(np.float64)
    r = ref.astype(np.float64)
    nan = ~np.isfinite(o)
    if nan.any():
        rows = np.unique(np.nonzero(nan)[0])
        frac = nan.sum() / o.size
        why = ("Unwritten output reads as NaN in the simulator." if frac > 0.99 else
               "Either part of the output was never written (reads as NaN), or the math overflowed (inf, inf-inf, 0*inf).")
        return CheckResult(
            False,
            f"{name}: {int(nan.sum())} of {o.size} values are NaN/inf (rows {rows[:6].tolist()}"
            f"{'...' if rows.size > 6 else ''}). {why}",
        )
    diff = np.abs(o - r)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = float(np.linalg.norm(r)) or 1.0
    rl2 = float(np.linalg.norm(o - r)) / denom
    stats = f"rel_l2={rl2:.3e} max_abs={max_abs:.3e}"
    if exact:
        ok = np.array_equal(out, ref)
        if not ok:
            idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
            return CheckResult(False, f"{name}: not bit-exact; worst at {idx}: got {o[idx]} want {r[idx]}", rl2, max_abs)
        return CheckResult(True, f"{name}: bit-exact", rl2, max_abs)
    if rel_l2 is not None:
        ok = rl2 <= rel_l2
        return CheckResult(ok, f"{name}: {stats} (limit rel_l2 {rel_l2:g})", rl2, max_abs)
    if atol is None:
        atol = 1e-5
    bad = diff > atol + rtol * np.abs(r)
    if bad.any():
        idx = np.unravel_index(int(np.argmax(diff - rtol * np.abs(r))), diff.shape)
        return CheckResult(
            False,
            f"{name}: {int(bad.sum())} values out of tolerance (atol={atol:g}, rtol={rtol:g}); "
            f"worst at {idx}: got {o[idx]:.6g} want {r[idx]:.6g}; {stats}",
            rl2,
            max_abs,
        )
    return CheckResult(True, f"{name}: {stats}", rl2, max_abs)
