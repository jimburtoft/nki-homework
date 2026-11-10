"""
Lab 1 -- Tiling: working within 128 partitions and finite SBUF
===============================================================

Goal: process tensors of any shape by splitting them into tiles that fit the
hardware. Time: about 30-40 minutes.

Background
----------
* The partition dimension of an on-chip tile is at most 128
  (`nl.tile_size.pmax`). A [1024, 512] tensor is processed as 8 tiles of
  [128, 512].
* Shapes are not always multiples of 128. The last tile may be partial: you
  allocate a full [128, N] tile and use only the first `rows` partitions,
  `tile[0:rows, :]`. The rule from Lab 0 still holds: an instruction must START
  at partition 0/32/64/96. Starting at 0 and using fewer rows is always legal.
* SBUF holds about 208 KiB per partition on trn2. A tile of [128, F] float32
  uses 4*F bytes of EVERY partition, so F can be at most roughly 50,000
  float32 values in total for everything that is live at once. Large free
  dimensions must also be tiled. The simulator does NOT check this limit, but
  the compiler does.
* Python `for` loops in a kernel are unrolled at trace time. Use plain
  `range(...)`; the loop variable is an ordinary int, so `min(...)` and
  `if` work on it.
* Data types: bfloat16 halves memory traffic. Engines compute internally in
  float32 and convert on write: the dtype of the `dst` tile decides the output
  type. `nisa.tensor_copy(dst=bf16_tile, src=fp32_tile)` is a cast.
  (`dma_copy` moves bytes; it does not convert between dtypes.)

Run:  python labs/L1_tiling/check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

P = 128  # nl.tile_size.pmax


# ---------------------------------------------------------------------------
# Exercise 1a: row tiling
#   x: [M, N] float32, M a multiple of 128 (check.py uses M=1024, N=512).
#   Return x * 0.5 + 2.0, processing one [128, N] tile per loop iteration.
# ---------------------------------------------------------------------------
@nki.jit
def tiled_affine(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 1b: a partial last tile
#   Same computation, but M may be any positive integer (check.py uses M=300,
#   M=129, and M=64).
# ---------------------------------------------------------------------------
@nki.jit
def tiled_affine_any_m(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 1c: free-dimension tiling
#   x: [128, 65536] float32 -- 256 KiB per partition, more than SBUF holds.
#   Return x * x (elementwise square).
#
#   First try the obvious one-tile version and run check.py: the simulator
#   passes, but the compiler rejects it. Then tile the free dimension in chunks
#   of FCHUNK elements.
# ---------------------------------------------------------------------------
FCHUNK = 8192


@nki.jit
def square_wide(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    y = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=t, data2=t, op=nl.multiply)
    nisa.dma_copy(dst=out, src=y)
    return out


# ---------------------------------------------------------------------------
# Exercise 1d: mixed precision
#   x: [M, N] bfloat16 (M a multiple of 128; check.py uses [256, 1024]).
#   Return exp(x) as bfloat16.
#   Use nisa.activation(op=nl.exp, ...). Engines compute in float32 internally;
#   the dst tile's dtype decides what is written. Write the result straight
#   into a bfloat16 tile -- no separate cast instruction needed.
# ---------------------------------------------------------------------------
@nki.jit
def exp_bf16(x):
    M, N = x.shape
    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out
