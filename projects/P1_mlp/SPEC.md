# P1: Fused MLP block (easy)

**Prerequisites:** Labs 0-4. **Expected effort:** 2-4 hours.

## The task

Implement the feed-forward block of a transformer as a single NKI kernel in
`kernel.py`:

```
H = gelu_tanh(X @ W1 + b1)
Y = H @ W2 + b2
```

| Tensor | Shape | dtype |
|---|---|---|
| `x` (X) | `[T, D]` | bfloat16 |
| `w1` (W1) | `[D, F]` | bfloat16 |
| `b1` | `[1, F]` | float32 |
| `w2` (W2) | `[F, D]` | bfloat16 |
| `b2` | `[1, D]` | float32 |
| output Y | `[T, D]` | bfloat16 |

`gelu_tanh(z) = 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3)))`, which is
`nl.gelu_apprx_tanh`.

### Shapes your kernel must support

- T, D and F are multiples of 128.
- 128 <= T <= 2048, 128 <= D <= 1024, 128 <= F <= 8192, and D x F <= 4,194,304.
- D and T are **not** necessarily multiples of 512.
- The public tests are in `grader/projects.py`. Your instructor grades with
  additional hidden shapes inside these bounds, so handle the general case.

### Correctness

Your output must satisfy `||Y - Y_ref|| / ||Y_ref|| <= 2e-2`, where `Y_ref` comes
from `reference.py` (float64 math on the bf16 inputs). The output must contain
no NaN. Accumulate matmuls in fp32 PSUM, as the hardware does by default; bf16
intermediates are fine.

### Rules

- All computation happens inside `mlp_kernel`. Do not preprocess the inputs in
  NumPy, for example by pre-transposing `x`.
- The kernel must compile for trn2. `grade.py` runs the real compiler, which
  checks SBUF/PSUM capacity and hardware access rules.
- You may use any `nki.isa` / `nki.language` API.

## How it is graded

```
python grader/grade.py P1
```

| Component | What is measured |
|---|---|
| Correctness | every case (public and hidden) passes the tolerance in the CPU simulator |
| Legality | every passing case compiles for trn2 |
| Performance | compiler-estimated latency on the `medium` case (T=1024, D=1024, F=4096); full marks at or below the threshold printed by `grade.py`, scaled on a log scale up to the zero-credit threshold |
| Write-up | see below |

The compiler estimate is the compiler's own model of the kernel's runtime. It is
not a hardware measurement, but it responds to the same things: instruction
count, DMA sizes, engine overlap and data reuse. If your class has trn2
hardware, `grade.py P1 --device` also measures the real runtime.

Your instructor sets the weights. A typical split is correctness 50%, legality
10%, performance 25%, write-up 15%.

## Write-up (1 page)

1. Draw (or describe) your dataflow. Which operand is stationary in each matmul,
   and how did you get `x` into the layout the Tensor Engine needs?
2. Give your SBUF and PSUM budgets per partition for the `medium` shape.
3. List each optimization you tried, with the compiler estimate before and after.
   Include the ones that did not help.

## Hints

- Lab 3 (exercise 3d) shows how to get `x^T` tiles on chip. Lab 3 (3e) shows how
  to fuse a bias and activation into the PSUM read-out.
- Think about which layout of `H` makes the second matmul easy. Is it `H [T, F]`
  or `H^T [F, T]`? Which one lets `b1` be a per-partition bias?
- The weights are the same for every token tile.
- Get a simple, correct version working first and commit it. Then optimize.

## Bonus (optional, not graded by `grade.py`)

- **fp8.** Once you have done Lab 5, try fp8 inputs to the Tensor Engine
  (`perf_mode=double_row`). Does it stay within the tolerance? What happens to
  the estimate?
- **Stochastic rounding, trn2 hardware only.** Run with
  `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` and compare your bf16 output's mean error
  with and without it. The simulator always uses round-to-nearest-even and
  ignores this variable.
