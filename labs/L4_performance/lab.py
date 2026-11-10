"""
Lab 4 -- Performance: reading the compiler's cost estimate
==========================================================

Goal: learn the three habits that make most NKI kernels fast, and measure each
one with the compiler's own estimate. Time: about 40 minutes.

Background
----------
check.py compiles each kernel for trn2 and prints the compiler's
post-scheduling ESTIMATE of one execution ("compiler estimate N us"). It is a
model of the hardware, not a measurement:
  * use it to compare two versions of YOUR kernel;
  * do not quote it as a real runtime (only trn2 hardware gives you that).
Every exercise in this lab has a performance goal on that estimate.

The three habits:

1. Move data in large DMAs.
   Every DMA has a fixed cost on top of its bytes. Moving [128, 64] at a time
   costs far more than moving [128, 4096] at a time, even though the total
   bytes are the same. Aim for at least 2-4 KiB per partition per transfer.

2. Let the engines overlap.
   The DMA queues and the four engines run IN PARALLEL. The scheduler can
   load tile i+1 while computing tile i and storing tile i-1, but ONLY if
   those tiles live in different buffers. Allocate a tile INSIDE the loop body
   (`t = nl.ndarray(...)` each iteration) and the compiler can give each
   iteration its own buffer. Allocate it ONCE outside the loop and every
   iteration must wait for the previous one to finish with that buffer.

3. Do not round-trip through HBM.
   HBM is far away; SBUF is next to the engines. If one step's output is the
   next step's input, keep it on chip and fuse the steps into one pass over
   the data.

Run:  python labs/L4_performance/check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

P = 128


# ---------------------------------------------------------------------------
# Exercise 4a: DMA size
#   x: [1024, 4096] float32.  Return 2 * x.
#   The starter code is correct but moves [128, 64] chunks. Make it meet the
#   goal printed by check.py.
# ---------------------------------------------------------------------------
@nki.jit
def double_big_dma(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    CH = 64  # columns per DMA
    for i in range(M // P):
        for j in range(N // CH):
            t = nl.ndarray((P, CH), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=x[i * P:(i + 1) * P, j * CH:(j + 1) * CH])
            y = nl.ndarray((P, CH), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y, data=t, op0=nl.multiply, operand0=2.0)
            nisa.dma_copy(dst=out[i * P:(i + 1) * P, j * CH:(j + 1) * CH], src=y)
    return out


# ---------------------------------------------------------------------------
# Exercise 4b: let iterations overlap
#   x: [2048, 2048] float32.  Return exp(x).
#   The starter code is correct, but its tiles are allocated once outside the
#   loop. Change only WHERE the buffers are allocated to meet the goal.
# ---------------------------------------------------------------------------
@nki.jit
def exp_overlap(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
    y = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
    for i in range(M // P):
        nisa.dma_copy(dst=t, src=x[i * P:(i + 1) * P, :])
        nisa.activation(dst=y, op=nl.exp, data=t)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=y)
    return out


# ---------------------------------------------------------------------------
# Exercise 4c: fuse two passes
#   x: [2048, 2048] float32.  Return gelu_tanh(3 * x).
#   The starter code writes 3*x to an HBM scratch tensor and reads it back in a
#   second pass. Fuse it into one pass to meet the goal. (Hint: nisa.activation
#   has a `scale` argument.)
# ---------------------------------------------------------------------------
@nki.jit
def scaled_gelu(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    scratch = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in range(M // P):
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[i * P:(i + 1) * P, :])
        y = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y, data=t, op0=nl.multiply, operand0=3.0)
        nisa.dma_copy(dst=scratch[i * P:(i + 1) * P, :], src=y)
    for i in range(M // P):
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=scratch[i * P:(i + 1) * P, :])
        y = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=y, op=nl.gelu_apprx_tanh, data=t)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=y)
    return out


# ---------------------------------------------------------------------------
# Exercise 4d: keep weights resident (open-ended)
#   xt: [128, T] bfloat16 (X^T, T=4096), w: [128, 512] bfloat16.
#   Return relu(X @ W) as bfloat16 [T, 512], i.e. out[t, :] = relu(sum_d xt[d, t] * w[d, :]).
#
#   The starter code is correct but reloads W from HBM for every token tile and
#   uses a separate instruction for relu. Meet the goal any way you like.
# ---------------------------------------------------------------------------
@nki.jit
def proj_relu(xt, w):
    D, T = xt.shape
    _, N = w.shape
    out = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for m in range(T // P):
        x_t = nl.ndarray((D, P), dtype=xt.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_t, src=xt[:, m * P:(m + 1) * P])
        w_t = nl.ndarray((D, N), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_t, src=w)
        acc = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=acc, stationary=x_t, moving=w_t)
        h = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=h, src=acc)
        y = nl.ndarray((P, N), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y, data=h, op0=nl.maximum, operand0=0.0)
        nisa.dma_copy(dst=out[m * P:(m + 1) * P, :], src=y)
    return out
