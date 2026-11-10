"""
Lab 6 -- Structure: masks, block sparsity, and scans
====================================================

Goal: exploit structure in the data -- zeros you can skip, and sequential
recurrences you can restructure -- using GpSimd predicates, the Tensor Engine,
and the Vector engine's scan instruction. Time: about 60 minutes.
Needed for P2 and P3.

Background
----------
1. Predicates without loading a mask. nisa.affine_select computes, for every
   element, an affine function of its partition index p and free index j:
       value = offset + channel_multiplier * p + step * j
   and keeps the input where cmp_op(value, 0) holds, else writes a constant.
       causal (keep j <= p):  pattern=[[-1, F]], offset=0, channel_multiplier=1, cmp_op=nl.greater_equal
   nisa.iota(dst, pattern, offset, channel_multiplier) writes `value` itself (int32);
   useful for building index-dependent constants.

2. Block sparsity. If a whole 128x128 block of a matrix is known to be zero
   (above the diagonal of a causal mask, or outside a sliding window), do not
   compute it at all. This is the largest speedup in causal attention and in
   Mamba-2's chunked scan. Trainium2 also has hardware M:N STRUCTURED sparsity
   in the Tensor Engine (e.g. 4 nonzeros in every 16 along K for 4x throughput);
   it is not yet exposed in NKI 0.7, so Exercise 6e is a pen-and-paper question.

3. Scans. A linear recurrence h[t] = a[t] * h[t-1] + b[t] is sequential in t.
   Two hardware-friendly ways to compute it:
     * nisa.tensor_tensor_scan(dst, data0=a, data1=b, initial=h0, op0=nl.multiply, op1=nl.add)
       runs it along the FREE dimension, one independent recurrence per partition,
       in ONE Vector-engine instruction.
     * For a plain prefix sum (a = 1), y = L @ x where L is lower-triangular ones:
       a MATMUL. The Tensor Engine does it in parallel. For long sequences, split
       into chunks: matmul inside each chunk, then carry each chunk's total into
       the next. That is exactly the structure of Mamba-2 (P3).

Run:  python labs/L6_structure/check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

P = 128
NEG = -30000.0  # "minus infinity" that stays finite in bf16/fp32 arithmetic


# ---------------------------------------------------------------------------
# Exercise 6a: causal and sliding-window masks
#   s: [128, 128] float32 (row i = query i, column j = key j, same block).
#   Return s where (j <= i) and (i - j < W), else NEG.  W = 48 in check.py.
#   Use nisa.affine_select twice (or once per condition); no loaded mask.
# ---------------------------------------------------------------------------
W_MASK = 48


@nki.jit
def window_mask(s):
    M, F = s.shape
    out = nl.ndarray((M, F), dtype=s.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 6b: skip zero blocks -- causal "attention without softmax"
#   qt, kt: [128 (d), T] bf16 (Q^T and K^T), v: [T, 128] bf16.  T = 1024 in check.py.
#   Return O = causal(Q K^T) @ V as float32 [T, 128], where causal() zeroes the
#   entries with key > query.
#
#   Layout trick: compute S^T blocks directly, nc_matmul(stationary=K^T block,
#   moving=Q^T block) gives S^T[key, query]. That block is exactly the
#   stationary operand of the second matmul, O_i += S^T_ji (contract over key) @ V_j.
#
#   The starter code computes every (query-block i, key-block j) pair and masks.
#   Blocks with j > i are entirely zero. Skip them, and only mask the diagonal
#   blocks. Meet the goal printed by check.py.
# ---------------------------------------------------------------------------
@nki.jit
def causal_av(qt, kt, v):
    D, T = qt.shape
    out = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.shared_hbm)
    q_s = nl.ndarray((D, T), dtype=qt.dtype, buffer=nl.sbuf)
    k_s = nl.ndarray((D, T), dtype=kt.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_s, src=qt)
    nisa.dma_copy(dst=k_s, src=kt)
    NB = T // P
    v_s = nl.ndarray((P, NB, D), dtype=v.dtype, buffer=nl.sbuf)
    for j in range(NB):
        nisa.dma_copy(dst=v_s[:, j, :], src=v[j * P:(j + 1) * P, :])
    for i in range(NB):
        o_acc = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.psum)
        for j in range(NB):
            st = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=st, stationary=k_s[:, j * P:(j + 1) * P], moving=q_s[:, i * P:(i + 1) * P])
            sb = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sb, src=st)
            # S^T[key j*P+p, query i*P+c]: keep (i*P + c) - (j*P + p) >= 0, else 0
            m = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.affine_select(dst=m, pattern=[[1, P]], offset=(i - j) * P, channel_multiplier=-1,
                               on_true_tile=sb, on_false_value=0.0, cmp_op=nl.greater_equal)
            nisa.nc_matmul(dst=o_acc, stationary=m, moving=v_s[:, j, :])
        o = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o, src=o_acc)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=o)
    return out


# ---------------------------------------------------------------------------
# Exercise 6c: a hardware linear recurrence
#   a, b: [128, L] float32 (L = 1024). 128 independent sequences, one per partition.
#   Return h with h[:, t] = a[:, t] * h[:, t-1] + b[:, t], h[:, -1] = 0.
#   Use ONE nisa.tensor_tensor_scan.
# ---------------------------------------------------------------------------
@nki.jit
def linear_recurrence(a, b):
    M, L = a.shape
    out = nl.ndarray((M, L), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 6d: prefix sum along a long sequence, with the Tensor Engine
#   x: [L, N] float32, L = 1024 (sequence along ROWS), N = 256 channels.
#   Return the inclusive prefix sum along rows: y[t, :] = sum_{s<=t} x[s, :].
#
#   Plan (chunk size 128):
#     * Build U [128, 128] with U[s, t] = 1 if s <= t else 0 (iota/memset + affine_select).
#       Then for one chunk X_c [128 (s), N], U.T @ X_c is the within-chunk prefix sum:
#       nc_matmul(stationary=U, moving=X_c).
#     * Carry: add the running total of all previous chunks to every row. A ones
#       matrix J [128, 128] as the stationary operand gives J.T @ X_c = column sums
#       broadcast to all 128 rows -- accumulate it into the NEXT chunks' PSUM tiles.
#   Everything stays in PSUM accumulation; no Vector-engine adds are required.
# ---------------------------------------------------------------------------
@nki.jit
def prefix_sum(x):
    L, N = x.shape
    out = nl.ndarray((L, N), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 6e (written, not checked): M:N structured sparsity
#   Trainium2's Tensor Engine supports 4:16, 4:12 and 4:8 structured sparsity in
#   the STATIONARY operand: in every group of N consecutive elements along K, at
#   most 4 are nonzero. Only the nonzeros are stored, plus 4-bit position tags.
#   Answer in your notes:
#     1. For a [128 (K), 128] stationary tile with 4:16 sparsity, how many values
#        and how many tag bits are stored? What is the compression ratio?
#     2. Why does this speed up the matmul by up to 4x, while skipping zero
#        BLOCKS (6b) only helps when the zeros are contiguous?
#     3. Name one model component whose weights you could prune to 4:16 and one
#        where you could not, and say why.
# ---------------------------------------------------------------------------
