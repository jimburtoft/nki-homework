"""
Lab 3 -- The Tensor Engine: tiled matrix multiplication
=======================================================

Goal: write a correct, tiled GEMM on the 128x128 systolic array, handle data
layout (transposes), and fuse work into the PSUM read-out.
Time: about 60 minutes.

Background
----------
    nisa.nc_matmul(dst=psum_tile, stationary=S, moving=Mv)
        computes  dst[m, n] (+)= sum_k S[k, m] * Mv[k, n]      i.e. dst = S.T @ Mv

* The CONTRACTION dimension k is the PARTITION dimension of BOTH inputs, up to 128.
* `stationary` S is [K<=128, M<=128]. It is loaded into the 128x128 array and
  stays there.
* `moving` Mv is [K<=128, N<=512] and streams through the array.
* `dst` is a [M, N] tile in PSUM, float32 on trn2. The array always
  accumulates in fp32, even for bf16/fp8 inputs.
* Calling nc_matmul again into the SAME psum tile ACCUMULATES. That is how you
  sum over K > 128: loop over K tiles into one PSUM tile.
* Engines cannot DMA out of PSUM. Read it with an engine instruction
  (tensor_copy, activation, tensor_scalar, ...) into SBUF, which can also
  apply a function and change the dtype ("evicting" PSUM). PSUM is small:
  8 banks of [128 x 2 KiB], i.e. at most 8 live [128, 512] fp32 accumulators.

So C = A @ B with A:[M,K], B:[K,N] needs A^T as the stationary operand:
        C[m-tile, n-tile] = sum over k-tiles of  AT[k-tile, m-tile].T @ B[k-tile, n-tile]

Data layout matters. If A arrives as [M, K] (rows = M), you must transpose
128x128 blocks to get [K, M]. Two ways to do it on the hardware:
    nisa.dma_transpose(dst=sbuf[K,M], src=hbm[M,K])   the DMA engine transposes in flight (2-byte dtypes)
    nisa.nc_transpose(dst=psum[K,M], data=sbuf[M,K])  the Tensor Engine transposes, result in PSUM

Run:  python labs/L3_matmul/check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

P = 128  # partition / contraction tile (nl.tile_size.pmax)
TM = 128  # stationary free size (nl.tile_size.gemm_stationary_fmax)
TN = 512  # moving free size    (nl.tile_size.gemm_moving_fmax)


# ---------------------------------------------------------------------------
# Exercise 3a: one matmul instruction
#   at: [128, 128] bfloat16 (this is A^T), b: [128, 512] bfloat16.
#   Return A @ B = at.T @ b as float32 [128, 512].
# ---------------------------------------------------------------------------
@nki.jit
def matmul_single(at, b):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 3b: tiled GEMM
#   at: [K, M] bfloat16 (A^T), b: [K, N] bfloat16; K, M multiples of 128,
#   N a multiple of 512. check.py uses K=1024, M=256, N=1024.
#   Return A @ B as float32 [M, N].
#
#   Loop over (m-tile, n-tile); for each, accumulate over k-tiles into one PSUM
#   tile, then evict it to SBUF and store.
# ---------------------------------------------------------------------------
@nki.jit
def matmul_tiled(at, b):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 3c: reuse what you load
#   Same inputs and output as 3b. The 3b loop nest reloads every A tile once
#   per n-tile and every B tile once per m-tile. Restructure the kernel so each
#   input element is DMA'd from HBM exactly ONCE. Hints:
#     - load all of B's k-tiles for an n-tile column once ([128, K/128, 512]
#       fits easily in SBUF for K=1024), or the whole of A^T, then loop;
#     - a tile may have more than 2 dims: nl.ndarray((P, KT, TN), ...) and
#       index it as t[:, kt, :].
#   check.py prints the compiler estimate; compare it with 3b's.
# ---------------------------------------------------------------------------
@nki.jit
def matmul_reuse(at, b):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 3d: natural layout -- transpose on chip
#   a: [M, K] bfloat16 (NOT transposed), b: [K, N] bfloat16.
#   M = 128, K = 512, N = 512 in check.py.  Return a @ b as float32 [M, N].
#
#   You need A^T tiles [128(k), 128(m)]. Produce them on chip with either
#   nisa.dma_transpose (HBM -> SBUF) or nisa.nc_transpose (SBUF -> PSUM, then
#   copy to SBUF). Either is accepted; try both and compare the estimates.
# ---------------------------------------------------------------------------
@nki.jit
def matmul_natural(a, b):
    M, K = a.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 3e: fuse the epilogue into the PSUM read-out
#   at: [K, M] bf16 (A^T), b: [K, N] bf16, bias: [M, 1] float32.
#   K=256, M=128, N=512 in check.py.
#   Return relu(A @ B + bias) as bfloat16 [M, N]   (bias is per ROW of C).
#
#   Evicting PSUM needs an engine instruction anyway, so make it do useful work:
#   a SINGLE nisa.activation can read the PSUM tile, add the per-partition bias,
#   apply relu, and write bf16 to SBUF.
# ---------------------------------------------------------------------------
@nki.jit
def matmul_bias_relu(at, b, bias):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out
