"""
Lab 7 (optional, ungraded) -- A look at Trainium3 in the simulator
==================================================================

Goal: try three features of the next-generation NeuronCore (NeuronCore-v4,
"trn3") and compare them with what you did on trn2. None of this is graded, and
none of it is allowed in the final projects (they are graded on trn2).
Time: about 30-45 minutes.

This lab runs with the trn3 target. check.py sets NKIHW_TARGET=trn3 before
importing nkihw, which pins NEURON_PLATFORM_TARGET_OVERRIDE=trn3. Results come
from the simulator only; there is no trn3 hardware in this course.

Features
--------
1. bf16 PSUM output. On trn2 the nc_matmul destination must be fp32 (Lab 5).
   On trn3 it may be bf16; the array still accumulates in fp32 and rounds once
   on write.
2. Wide moving tiles. On trn2 the moving free size is at most 512 (one PSUM
   bank). On trn3 it can be up to 4096 for an fp32 destination (all 8 banks),
   so one instruction can do 8x the work.
3. MX (microscaling) formats. MXFP8 stores fp8 values plus one shared 8-bit
   power-of-two scale per group of 32 values (OCP Microscaling spec). The scale
   gives each small group its own range, so MXFP8 is far more accurate than
   plain fp8 for data with outliers.
       nisa.quantize_mx(dst=fp8x4_tile, src=bf16_tile, dst_scale=uint8_tile)
       nisa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale)
   The x4 dtypes (nl.float8_e4m3fn_x4) pack 4 fp8 values in one 32-bit element,
   and a scaling group is 8 partitions x 4 packed elements. One nc_matmul_mx
   contracts over 4 x 128 = 512 rows of K.
   Layout: for an unquantized [512 (K), M] input viewed as 4 slabs of [128, M],
   SBUF must hold tile[p, m, j] = input[j*128 + p, m] (the 4 elements packed
   together come from the 4 slabs). One strided DMA does it:
       src.reshape((4, 128, M)).ap(pattern=[[M, 128], [1, M], [128*M, 4]], offset=0)
   (Based on the public Neuron SDK "MXFP matmul" deep dive.)

Questions to answer in your notes
---------------------------------
  Q1. 7a: what error does trn2 give for a bf16 PSUM tile? What is the accuracy
      cost of bf16 PSUM output on trn3, and what does it save?
  Q2. 7b: compare the instruction count and compiler estimate with Lab 3's
      512-wide version. Why does the estimate not drop by 8x?
  Q3. 7c: compare MXFP8 accuracy with Lab 5's plain fp8 double_row matmul
      (both are 8-bit). Then multiply one input row by 1000 (an outlier) and
      compare again.
  Q4. Which of these would most change your P1, P2 or P3 design on trn3?

Run:  python labs/L7_trn3/check.py            (simulator only)
      python labs/L7_trn3/check.py --compile  (also compile for trn3 and print the estimate)
"""
import os
import sys

os.environ["NKIHW_TARGET"] = "trn3"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

P = 128


# ---------------------------------------------------------------------------
# Exercise 7a: bf16 PSUM
#   at: [128, 128] bf16, b: [128, 512] bf16. Return at.T @ b as bfloat16,
#   using a bfloat16 PSUM tile as the nc_matmul destination.
# ---------------------------------------------------------------------------
@nki.jit
def matmul_bf16_psum(at, b):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 7b: one wide matmul
#   at: [128, 128] bf16, b: [128, 4096] bf16. Return at.T @ b as float32
#   with a SINGLE nc_matmul (moving free size 4096).
# ---------------------------------------------------------------------------
@nki.jit
def matmul_wide(at, b):
    K, M = at.shape
    _, N = b.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 7c: MXFP8 matmul
#   st: [512, 128] bf16 (K, M), mv: [512, 512] bf16 (K, N).
#   Return st.T @ mv as float32, computed with on-chip quantize_mx and ONE
#   nc_matmul_mx (float8_e4m3fn_x4 data, uint8 scales).
#   Steps: strided DMA into [128, M, 4] / [128, N, 4] tiles (see the header),
#   view them as [128, 4M] / [128, 4N] with .reshape, quantize_mx each into
#   [128, M] / [128, N] x4 tiles with [128, M] / [128, N] uint8 scale tiles,
#   then nc_matmul_mx into a [M, N] fp32 PSUM tile.
# ---------------------------------------------------------------------------
@nki.jit
def matmul_mxfp8(st, mv):
    K4, M = st.shape
    _, N = mv.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out
