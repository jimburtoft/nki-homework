"""Self-test for the nkihw harness. Run: python common/selftest.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nkihw  # noqa: E402  (must precede nki)

import numpy as np  # noqa: E402
import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402


@nki.jit
def good(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    u = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=t, op0=nl.multiply, operand0=2.0)
    nisa.dma_copy(dst=out, src=u)
    return out


@nki.jit
def too_big(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((128, 64 * 1024), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[:, 0:x.shape[1]], src=x)
    nisa.dma_copy(dst=out, src=t[:, 0:x.shape[1]])
    return out


@nki.jit
def bad_partition(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    u = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u[16:80, :], data=t[16:80, :], op0=nl.multiply, operand0=2.0)
    nisa.tensor_scalar(dst=u[0:16, :], data=t[0:16, :], op0=nl.multiply, operand0=2.0)
    nisa.tensor_scalar(dst=u[80:128, :], data=t[80:128, :], op0=nl.multiply, operand0=2.0)
    nisa.dma_copy(dst=out, src=u)
    return out


@nki.jit
def no_return(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)


@nki.jit
def stub(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    return out


def main():
    x = np.random.default_rng(0).standard_normal((128, 512)).astype(np.float32)
    fails = 0

    def expect(cond, what):
        nonlocal fails
        print(("ok   " if cond else "BAD  ") + what)
        fails += 0 if cond else 1

    r = nkihw.check(nkihw.simulate(good, x), 2 * x, exact=True)
    expect(r.ok, f"simulate good: {r}")
    r = nkihw.check(nkihw.simulate(stub, x), 2 * x)
    expect(not r.ok and "NaN" in r.msg, f"stub fails with NaN message: {r}")
    rep = nkihw.compile_kernel(good, x)
    expect(rep.ok and rep.est_ns and rep.est_ns > 0, f"compile good: {rep.summary()}")
    for k in (too_big, bad_partition, no_return):
        rep = nkihw.compile_kernel(k, x)
        expect(not rep.ok, f"compile {k.func.__name__} rejected: {rep.summary()}")
    print("SELFTEST", "PASS" if fails == 0 else f"FAIL ({fails})")
    return fails


if __name__ == "__main__":
    sys.exit(main())
