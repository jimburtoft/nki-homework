"""
Lab 2 -- Engines, reductions, and per-partition scalars
=======================================================

Goal: implement the normalization and softmax patterns that appear in every
transformer, and learn which engine does what. Time: about 40 minutes.

Background
----------
In every exercise here, each ROW of the input lives on one PARTITION, so a
reduction along a row is a reduction along the FREE dimension. That is cheap:
each partition reduces its own data, all 128 in parallel.

    nisa.tensor_reduce(dst=[P,1], op=nl.add | nl.maximum, data=[P,F], axis=1, negate=False)
        Vector engine. Reduce along the free axis. negate=True writes -result.

    nisa.activation(dst, op, data, bias=None, scale=1.0)
        Scalar engine: dst = op(data * scale + bias).
        `bias` and `scale` may be Python floats or [P,1] tiles (one value per partition).

    nisa.activation_reduce(dst, op, data, reduce_op=nl.add, reduce_res=[P,1], bias=None, scale=1.0)
        Like activation, but ALSO reduces the result along the free axis into
        reduce_res in the SAME instruction. exp + row-sum in one pass: this is
        the softmax trick.

    nisa.tensor_scalar(dst, data, op0, operand0, op1=None, operand1=None)
        Vector engine. operand0/operand1 may be floats or [P,1] tiles.

    nisa.scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1)
        dst = op1(op0(data, operand0), operand1), where operand0 is a float or a
        [P,1] tile and operand1 is a full tile. One instruction for "x * rstd * w".

    nisa.reciprocal(dst, data)       1/x on the Vector engine.

Broadcasting a row vector to all 128 partitions:
    A weight vector w of shape [1, D] in HBM can be DMA'd to every partition
    by reading it with partition stride 0:
        w_tile = nl.ndarray((P, D), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_tile, src=w.ap(pattern=[[0, P], [1, D]]))
    `.ap(pattern=[[stride, count], ...])` is an explicit access pattern over the
    tensor's elements: the FIRST pair is the partition dimension, the rest are
    free dimensions, and strides are in ELEMENTS. Above: partition stride 0
    (repeat the same row P times), then D contiguous elements.
    A second common pattern: spread a row vector b of shape [1, F] over
    partitions as columns, so that tile[p, j] = b[0, j*P + p] (one value per
    partition for each 128-chunk j):
        b_tile = nl.ndarray((P, F // P), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b.ap(pattern=[[1, P], [P, F // P]]))
    Then b_tile[:, j:j+1] is a [P, 1] per-partition operand.

Rules of thumb for engine choice (trn2):
    Scalar (ACT)  transcendental functions (exp, gelu, rsqrt, ...) and fused scale+bias
    Vector (DVE)  tensor-tensor arithmetic, reductions, per-partition scalars
    The engines run IN PARALLEL. Spreading the work across them is how fused
    kernels get fast; check.py's compiler estimate lets you measure this.

Run:  python labs/L2_engines/check.py
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
# Exercise 2a: row statistics
#   x: [128, F] float32.  Return a TUPLE (row_max, row_sum), each [128, 1] float32.
# ---------------------------------------------------------------------------
@nki.jit
def row_stats(x):
    M, F = x.shape
    row_max = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    row_sum = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return row_max, row_sum


# ---------------------------------------------------------------------------
# Exercise 2b: RMSNorm
#   x: [T, D] bfloat16 (T a multiple of 128), w: [1, D] bfloat16, eps = 1e-6.
#   y[t, :] = x[t, :] / sqrt(mean(x[t, :]**2) + eps) * w      -> bfloat16 [T, D]
#
#   Suggested plan per 128-row tile:
#     1. activation_reduce(op=nl.square, reduce_op=nl.add) -> sum of squares [P,1]
#     2. activation(op=nl.rsqrt, scale=1/D, bias=eps)       -> rstd [P,1]
#     3. scalar_tensor_tensor: (x * rstd) * w                -> y
#   Load w once (outside the loop) with the stride-0 broadcast shown above.
# ---------------------------------------------------------------------------
EPS = 1e-6


@nki.jit
def rmsnorm(x, w):
    T, D = x.shape
    out = nl.ndarray((T, D), dtype=x.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 2c: numerically stable softmax over rows
#   x: [T, F] float32 (T a multiple of 128).
#   y = exp(x - rowmax) / rowsum(exp(x - rowmax))        -> float32 [T, F]
#
#   Target: 4 compute instructions per tile (one per step below).
#     1. tensor_reduce max with negate=True   -> -max  [P,1]
#     2. activation_reduce exp with bias=-max  -> e, and sum(e) in the same instruction
#     3. reciprocal(sum)                       -> 1/sum [P,1]
#     4. tensor_scalar multiply by 1/sum       -> y
#   check.py tests inputs with large values (|x| ~ 100) where the unstable
#   version overflows.
# ---------------------------------------------------------------------------
@nki.jit
def softmax(x):
    T, F = x.shape
    out = nl.ndarray((T, F), dtype=x.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out
