# P3: Mamba-2 SSD chunked scan (hard)

**Prerequisites:** Labs 0-6. **Expected effort:** 6-10 hours.

## Background

Mamba-2 replaces attention with a linear recurrence over a matrix-valued state.
For one head, with head dimension `P` and state dimension `N`:

```
h_t = exp(a_t) * h_{t-1} + B_t^T x_t        h_t: [N, P]   (B_t^T x_t is an outer product)
y_t = C_t h_t                                y_t: [P]
```

`a_t <= 0` is a learned log-decay for each step, so `exp(a_t)` is in (0, 1].

Computed step by step, this is a chain of `L` dependent tiny updates, which is a
poor fit for a 128x128 systolic array. The Mamba-2 paper (Dao and Gu, 2024)
shows that the same output can be written in a **dual, attention-like form**:

```
y_t = sum_{s <= t}  (C_t . B_s) * exp(A_t - A_s) * x_s,      A_t = a_1 + ... + a_t
```

Split the sequence into chunks of `Q = 128` steps. Then:

- **Inside a chunk** the dual form is a masked matrix product, much like
  causal attention without softmax:
  `Y_intra = (G o Lmat) X`, where `G[t, s] = C_t . B_s` and
  `Lmat[t, s] = exp(A_t - A_s)` for `s <= t`, else 0.
- **Between chunks**, carry the state forward. Chunk `c` adds
  `exp(A_t) C_t S_{c-1}`, where `S_{c-1}` is the state after the previous chunk
  and `A_t` restarts at each chunk. The state update is itself a matmul:
  `S_c = exp(A_last) S_{c-1} + B_c^T (w o X_c)`, with `w_s = exp(A_last - A_s)`.

Every piece is a matmul, an elementwise op, or a prefix sum. You built each of
these in Labs 3-6.

## The task

Implement `ssd_kernel(x, a, B, C)` in `kernel.py`.

| Tensor | Shape | dtype |
|---|---|---|
| `x` | `[L, P]` | bfloat16 |
| `a` | `[L, 1]` | float32, all values <= 0 |
| `B`, `C` | `[L, N]` | bfloat16 |
| output `y` | `[L, P]` | float32 |

### Shapes your kernel must support

- `L` is a multiple of 128, with 128 <= L <= 8192.
- `P` is 64 or 128.
- `N` is 64 or 128.
- The decays can be **large**. Over one chunk, `A_t` can reach -500 or below, so
  `exp(-A_s)` overflows fp32. Hidden tests include such inputs.

### Correctness

`||y - y_ref|| / ||y_ref|| <= 2e-2`, with no NaN. `y_ref` is the recurrent form
in float64 (`reference.py`).

## How it is graded

```
python grader/grade.py P3
```

| Component | What is measured |
|---|---|
| Correctness | every public and hidden case passes in the simulator |
| Legality | every passing case compiles for trn2 |
| Performance | compiler estimate on the `long` case (L=4096, P=128, N=128) |
| Write-up | see below |

## Write-up (2 pages)

1. **Derivation.** Derive the chunked form from the recurrence: the intra-chunk
   term, the inter-chunk term, and the state update.
2. **Numerics.** Which quantities can overflow or underflow, and how does your
   kernel avoid it?
3. **Mapping to hardware.** Which computations run on which engine? Which
   operand is stationary in each matmul? Where is the remaining sequential
   dependency between chunks, and how did you shorten it?
4. **Measurements.** Give the compiler estimate for each version you built. What
   limits your fastest version: Tensor Engine, Vector/Scalar engines, or DMA?
   How do you know?

## Hints

- **Prefix sums.** Lab 6 (6d) computes a prefix sum with an upper-triangular
  ones matrix. Two related tricks are useful here:
  - A ones matrix as the stationary operand broadcasts a column sum to all
    partitions.
  - `upper * a` (a per-partition scalar) as the moving operand gives `A_t` on
    every partition.
- **Order of operations.** Form `A_t - A_s` *before* `exp`, using
  `activation(exp, data=..., bias=-A_s)`. Computing `exp(A_t) * exp(-A_s)` is
  mathematically equal but numerically wrong.
- **Masking.** Mask with `affine_select` to a large negative value *before*
  `exp`, so that masked entries become exactly 0.
- **Layout.** If you compute `G^T` (keys `s` on partitions) directly with
  `nc_matmul(stationary=B^T, moving=C^T)`, then `(G o Lmat)^T` is already the
  stationary operand that the next matmul with `X` needs.
- **The carry.** It is a linear recurrence over chunks. Lab 6 (6c) has a single
  instruction for that.

## Bonus (optional)

- Multi-head: support `x [H, L, P]` and the matching `a`, `B`, `C`, and explain
  how you would schedule several heads across the engines.
- Try chunk size 64 with Tensor Engine row tiling (Lab 5, 5c). Is it faster? Why
  or why not?
