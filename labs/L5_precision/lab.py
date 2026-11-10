"""
Lab 5 -- Precision on Trainium: accumulation, fp8, and rounding
===============================================================

Goal: understand where precision is lost in a kernel, and how the Trainium
features below trade precision against speed. Time: about 60 minutes.
Needed for P2 and P3; optional for P1.

    * fp32 PSUM accumulation (always on, and free)
    * fp8 inputs and the fp8 "double row" Tensor Engine mode (2x the K per instruction)
    * small-K matmuls and Tensor Engine row tiling
    * stochastic rounding (in software here; in hardware as a trn2 bonus)

Background
----------
1. The Tensor Engine multiplies bf16/fp8 inputs and ACCUMULATES IN FP32. On
   trn2 the PSUM destination of nc_matmul MUST be float32 (try bfloat16 and
   read the error). Accumulating over K in PSUM is therefore much more
   accurate than adding partial products into a bf16 SBUF tile.

2. fp8 (float8_e4m3: 4 exponent bits, 3 mantissa bits) halves the bytes of
   bf16. With perf_mode=nisa.matmul_perf_mode.double_row, one nc_matmul
   contracts over 2*128 = 256 rows of K: the stationary and moving tiles have
   shape [128, 2, M] and [128, 2, N], and contraction index k = p * 2 + j
   pairs partition p with slot j. Read the compiler estimate: about 2x the
   MACs in about the same time.
   Converting to fp8 is an engine instruction: nisa.tensor_copy(dst=fp8_tile, src=bf16_tile).

3. A matmul with K < 128 leaves part of the 128x128 array idle. Row tiling
   (tile_position=(row, 0), tile_size=(K, 128) with K in {32, 64}) tells the
   hardware which rows of the array the matmul uses, so independent small-K
   matmuls placed at different rows can share the array.

4. Rounding. Converting fp32 -> bf16 keeps the top 16 bits. Round-to-nearest-
   even (RNE, the default) is unbiased on average for random data, but it is
   badly biased when you add many SMALL values to a LARGE accumulator: every
   addition rounds back down, and the sum never grows. Stochastic rounding (SR)
   rounds up with probability equal to the discarded fraction, so the
   EXPECTED result is exact.
       fp32 -> bf16 with SR, as bit operations on the uint32 view:
           bits = bits(x) + (random 16-bit integer);  bits &= 0xFFFF0000
   On Trainium hardware SR is a hardware rounding mode, enabled for the whole
   process with  NEURON_RT_STOCHASTIC_ROUNDING_EN=1. The CPU simulator ignores
   that variable (it always uses RNE), so in this lab you implement SR
   yourself with the bit trick, and (bonus) compare against hardware SR.

API used here
    tile.view(nl.uint32)                 reinterpret the bits of a float32 tile (no copy)
    nisa.rng(dst=uint32_tile)            fill with random 32-bit integers
    ops nl.bitwise_and, nl.add on uint32 tiles
    nisa.nc_matmul(..., perf_mode=nisa.matmul_perf_mode.double_row)
    nisa.nc_matmul(..., tile_position=(row, 0), tile_size=(K, 128))

Run:  python labs/L5_precision/check.py
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
# Exercise 5a: where to accumulate
#   at: [K, 128] bf16 (A^T), b: [K, 512] bf16, K up to 8192.
#   Return A @ B (float32 [128, 512]) to within rel_l2 1e-5.
#
#   The starter code is correct in structure, but it adds each k-tile's partial
#   product into a bfloat16 SBUF accumulator. Run check.py and note how the
#   error grows with K (K=512 vs K=8192). Fix it. (Hint: what does a second
#   nc_matmul into the SAME psum tile do?)
# ---------------------------------------------------------------------------
@nki.jit
def matmul_long_k(at, b):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    run = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=run, value=0.0)
    for k in range(K // P):
        a_t = nl.ndarray((P, M), dtype=at.dtype, buffer=nl.sbuf)
        b_t = nl.ndarray((P, N), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=at[k * P:(k + 1) * P, :])
        nisa.dma_copy(dst=b_t, src=b[k * P:(k + 1) * P, :])
        acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=acc, stationary=a_t, moving=b_t)
        nisa.tensor_tensor(dst=run, data1=run, data2=acc, op=nl.add)
    o = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=run)
    nisa.dma_copy(dst=out, src=o)
    return out


# ---------------------------------------------------------------------------
# Exercise 5b: fp8 double-row matmul
#   at: [256, 128] bf16 (A^T), b: [256, 512] bf16.
#   Return A @ B as float32, computed with ONE nc_matmul in fp8 double_row mode.
#
#   check.py compares against the product of the fp8-ROUNDED inputs, so a
#   correct kernel matches it to ~1e-6. It also prints how far that is from the
#   exact bf16 product: that is the accuracy cost of fp8.
#
#   Steps:
#     1. DMA at into a [128, 2, M] bf16 tile so that tile[p, j, :] = at[2*p + j, :].
#        A reshaped view does this: at.ap(pattern=[[2*M, P], [M, 2], [1, M]]).
#        Do the same for b.
#     2. tensor_copy both tiles to float8_e4m3 tiles of the same shape.
#     3. one nc_matmul(..., perf_mode=nisa.matmul_perf_mode.double_row).
# ---------------------------------------------------------------------------
@nki.jit
def matmul_fp8_double_row(at, b):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 5c: four small-K matmuls with row tiling
#   at: [4, 32, 128] bf16, b: [4, 32, 512] bf16.
#   Return out[g] = at[g].T @ b[g] for g = 0..3, float32 [4, 128, 512].
#
#   Place group g's K=32 rows at SBUF partitions [32g, 32g+32) of one [128, *]
#   tile, and issue each matmul with tile_position=(32*g, 0), tile_size=(32, 128).
#   (The partition-start rule from Lab 0 is what makes 32 the smallest tile.)
# ---------------------------------------------------------------------------
@nki.jit
def four_small_matmuls(at, b):
    G, K, M = at.shape
    _, _, N = b.shape
    out = nl.ndarray((G, M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 5d: stochastic rounding, given the random bits
#   x: [128, F] float32, r: [128, F] uint32 with values in [0, 65536).
#   Return x rounded to bfloat16 with stochastic rounding, using r as the random
#   bits:  bits = (bits(x) + r) & 0xFFFF0000, then reinterpret as float32 and
#   convert to bf16 (the low 16 bits are 0, so this conversion is exact).
#   check.py requires a BIT-EXACT match.
# ---------------------------------------------------------------------------
@nki.jit
def sr_round(x, r):
    M, F = x.shape
    out = nl.ndarray((M, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 5e: why SR matters -- a bf16 accumulator
#   x: [128, 512] float32 (only its shape is used).
#   Start a bfloat16 accumulator at 1.0 and add 2**-9 to it N_STEPS = 256 times.
#   The exact answer is 1.5.
#     * With plain bf16 adds (RNE) the accumulator never moves. (bf16 spacing
#       at 1.0 is 2**-7, and 2**-9 is below half a step.)
#     * Implement each add as: fp32 sum -> stochastic round to bf16 (as in 5d,
#       but generate the random bits with nisa.rng and keep the low 16 bits).
#   check.py requires the MEAN over all 65536 elements to be within 0.01 of 1.5.
# ---------------------------------------------------------------------------
N_STEPS = 256
STEP = 2.0 ** -9


@nki.jit
def sr_accumulate(x):
    M, F = x.shape
    out = nl.ndarray((M, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    acc = nl.ndarray((M, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=1.0)
    for _ in range(N_STEPS):
        nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=STEP)  # RNE: stays at 1.0
    nisa.dma_copy(dst=out, src=acc)
    return out


# ---------------------------------------------------------------------------
# BONUS (requires trn2 hardware): hardware stochastic rounding
#   On a trn2 instance, run the PLAIN bf16 version of 5e (one tensor_scalar add
#   per step, no software SR) on the device twice:
#       python labs/L5_precision/bonus_hw_sr.py                   # RNE
#       NEURON_RT_STOCHASTIC_ROUNDING_EN=1 python labs/L5_precision/bonus_hw_sr.py
#   and explain the two results. Which is faster, software SR (5e) or hardware SR?
# ---------------------------------------------------------------------------
