"""Lab 5 bonus -- hardware stochastic rounding (needs a trn2 instance).

    python labs/L5_precision/bonus_hw_sr.py                                     # round-to-nearest-even
    NEURON_RT_STOCHASTIC_ROUNDING_EN=1 python labs/L5_precision/bonus_hw_sr.py  # stochastic rounding

The kernel adds 2**-9 to a bfloat16 accumulator (starting at 1.0) 256 times
with plain bf16 adds (no software SR). The exact result is 1.5.
The CPU simulator ignores NEURON_RT_STOCHASTIC_ROUNDING_EN, so this script
runs on the simulator only to show the RNE result for comparison.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
import numpy as np  # noqa: E402

N_STEPS, STEP = 256, 2.0 ** -9


@nki.jit
def plain_bf16_accumulate(x):
    M, F = x.shape
    out = nl.ndarray((M, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    acc = nl.ndarray((M, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=1.0)
    for _ in range(N_STEPS):
        nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=STEP)
    nisa.dma_copy(dst=out, src=acc)
    return out


def main():
    x = np.zeros((128, 512), np.float32)
    sr = os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
    if nkihw.has_device():
        out = nkihw.run_device(plain_bf16_accumulate, x).astype(np.float64)
        t = nkihw.bench_device(plain_bf16_accumulate, x)
        where = f"trn2 device, NEURON_RT_STOCHASTIC_ROUNDING_EN={sr}, {t['mean_us']:.1f} us"
    else:
        out = nkihw.simulate(plain_bf16_accumulate, x).astype(np.float64)
        where = "CPU simulator (always RNE; the SR variable is ignored)"
    print(f"{where}\n  mean = {out.mean():.4f}   std = {out.std():.4f}   (exact: 1.5)")


if __name__ == "__main__":
    main()
