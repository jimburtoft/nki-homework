# P2: Sliding-window causal attention (medium)

**Prerequisites:** Labs 0-4 and 6. Lab 5 is optional and only needed for the
fp8 bonus. **Expected effort:** 4-6 hours.

## The task

Implement one head of sliding-window causal attention as a single NKI kernel
in `kernel.py`:

```
S[i, j] = (q_i . k_j) / sqrt(d)           if 0 <= i - j < W
        = -infinity                         otherwise
O = softmax(S, axis=-1) @ V
```

Query `i` attends to itself and the `W - 1` keys before it. This is the
attention used by models such as Mistral and Gemma for their local layers.

| Tensor | Shape | dtype |
|---|---|---|
| `q`, `k`, `v` | `[T, d]` | bfloat16 |
| `W` | Python `int` | compile-time constant |
| output | `[T, d]` | bfloat16 |

### Shapes your kernel must support

- `T` is a multiple of 128, with 128 <= T <= 4096.
- `d` is 64 or 128.
- `W` is any integer with 1 <= W <= T. W is **not** necessarily a multiple of 128.
- The public tests are in `grader/projects.py`. Hidden tests use other shapes
  inside these bounds.

### Correctness

Your output must satisfy `||O - O_ref|| / ||O_ref|| <= 2e-2` and contain no
NaN. `O_ref` comes from `reference.py`.

### Rules

- All computation happens inside `window_attention`.
- The kernel must compile for trn2. Scores for long sequences do **not** fit in
  SBUF all at once. The compiler checks SBUF capacity, but the simulator does not.

## How it is graded

```
python grader/grade.py P2
```

| Component | What is measured |
|---|---|
| Correctness | every public and hidden case passes in the simulator |
| Legality | every passing case compiles for trn2 |
| Performance | compiler-estimated latency on the `long` case (T=4096, d=128, W=512) |
| Write-up | see below |

## Write-up (1-2 pages)

1. How much work does your kernel do as a function of T and W? Give the matmul
   MACs and the number of 128x128 blocks visited, compared with full causal
   attention.
2. How do you handle a window edge that falls inside a 128-block?
3. What did you transpose, and how: DMA, Tensor Engine, or by choosing which
   operand is stationary? Give the compiler estimate for each choice you tried.
4. Is your softmax numerically safe for large logits? Explain why. Hidden tests
   include inputs scaled so that `|S|` reaches about 100.

## Hints

- Lab 6 (6a, 6b) contains both of the key ideas: masks from `affine_select`,
  and skipping blocks that are known to be zero.
- Most of the speedup comes from not visiting key blocks that lie entirely
  outside the window.
- `exp` with a per-row `bias = -max` and a fused row-sum (`activation_reduce`)
  is Lab 2's softmax. Normalize *after* the `P @ V` matmul: that is one
  multiply per output element instead of one per score.
- `P @ V` contracts over keys, so the probabilities must have **keys on
  partitions** when they are used as the stationary operand.

## Bonus (optional, not graded by `grade.py`)

- **Online softmax.** Process key blocks one at a time with a running max and
  rescaling (as in FlashAttention). Is it faster than the two-pass version for
  the window sizes here? Why or why not?
- **fp8.** Compute `P @ V` with fp8 `double_row` (Lab 5). What happens to the
  accuracy and to the estimate?
