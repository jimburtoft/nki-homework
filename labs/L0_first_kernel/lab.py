"""
Lab 0 -- Your first NKI kernels
===============================

Goal: learn the NeuronCore memory model and the basic shape of every NKI kernel.
Time: about 30 minutes.

Background
----------
A Trainium2 NeuronCore has three kinds of memory that a kernel touches directly:

    HBM   device DRAM, large (GBs), where kernel inputs and outputs live
    SBUF  on-chip scratchpad, organized as 128 PARTITIONS x ~208 KiB each,
          where every compute engine reads and writes its data
    PSUM  small on-chip accumulator memory (128 partitions x 16 KiB),
          written by the Tensor Engine (Lab 3)

and four compute engines that all read and write SBUF:

    Tensor engine (PE)   matrix multiply on a 128 x 128 systolic array (Lab 3)
    Vector engine (DVE)  elementwise and reduction math between tensors
    Scalar engine (ACT)  elementwise math with a fast lookup table for exp, gelu, ...
    GpSimd engine        general-purpose cores, the most flexible and the slowest

Engines cannot read HBM. Every kernel therefore follows the same pattern:

    1. allocate the output in HBM            nl.ndarray(..., buffer=nl.shared_hbm)
    2. DMA inputs HBM -> SBUF                nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor)
    3. compute SBUF -> SBUF on an engine     nisa.tensor_scalar / tensor_tensor / activation ...
    4. DMA results SBUF -> HBM               nisa.dma_copy(dst=hbm_out, src=sbuf_tile)
    5. return the output tensor              (the hardware REQUIRES the return)

The first dimension of an SBUF tile is the PARTITION dimension and can be at
most 128. The remaining ("free") dimensions are stored within each partition.
An engine instruction processes all partitions in parallel, so a [128, 512]
tile is processed as 128 lanes x 512 elements.

Every tensor operation in this lab is an `nki.isa` (nisa) instruction. Each
instruction writes into a `dst` tile that you allocate first with
`nl.ndarray(shape, dtype=..., buffer=nl.sbuf)`.

How to run
----------
    python labs/L0_first_kernel/check.py          # simulate + compile every exercise
    python labs/L0_first_kernel/check.py -e 0b    # one exercise only
    python labs/L0_first_kernel/check.py --no-compile

check.py runs your kernel on the CPU simulator, compares it against a NumPy
reference, and then compiles it for trn2 with the real Neuron compiler. The
compiler catches problems the simulator does not (see Exercise 0e) and prints
its own runtime ESTIMATE.

Until you fill in a kernel it returns an output tensor that was never written.
The simulator fills unwritten memory with NaN, so the check reports NaN values.
That is expected.

Useful API (full docs: https://awsdocs-neuron.readthedocs-hosted.com/ -> NKI API reference)
    nl.ndarray(shape, dtype=..., buffer=nl.sbuf | nl.shared_hbm)
    nisa.dma_copy(dst=..., src=...)
    nisa.tensor_scalar(dst=, data=, op0=, operand0=, op1=None, operand1=None)
        dst = op1(op0(data, operand0), operand1)  -- two ops in ONE instruction
    nisa.tensor_tensor(dst=, data1=, data2=, op=)        dst = op(data1, data2)
    nisa.activation(dst=, op=, data=, bias=None, scale=1.0)
        dst = op(data * scale + bias)                    -- runs on the Scalar engine
    ops: nl.add, nl.subtract, nl.multiply, nl.maximum, nl.minimum,
         nl.relu, nl.exp, nl.gelu_apprx_tanh, ...
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import nkihw  # noqa: E402,F401  -- must be imported before nki (pins the trn2 target)

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402


# ---------------------------------------------------------------------------
# Exercise 0a: copy
#   x: [128, 512] float32 in HBM.  Return a copy of x.
#   Steps 1 and 5 are done for you. Allocate an SBUF tile, DMA x into it, and DMA
#   it back out to `out`.
# ---------------------------------------------------------------------------
@nki.jit
def copy_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 0b: scale and shift in ONE instruction
#   x: [128, 512] float32.  Return 3 * x + 1.
#   Use a single nisa.tensor_scalar with both op0/operand0 and op1/operand1.
#   (Fewer instructions = less time. The compiler estimate printed by
#   check.py lets you see this.)
# ---------------------------------------------------------------------------
@nki.jit
def scale_shift_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 0c: elementwise add of two tensors
#   a, b: [128, 512] float32.  Return a + b.
#   Note: NKI tensors do not support Python operators (`a + b` is an error).
#   Use nisa.tensor_tensor.
# ---------------------------------------------------------------------------
@nki.jit
def add_kernel(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 0d: a fused activation on the Scalar engine
#   x: [128, 512] float32, b: [128, 1] float32 (one value per PARTITION).
#   Return relu(2 * x + b), with b broadcast across each row.
#
#   nisa.activation computes op(data * scale + bias) in ONE instruction, where
#   `bias` may be a [P, 1] tile: one value per partition. Per-partition scalars
#   like this come up constantly in NKI (biases, normalization factors, softmax
#   maxima); remember the pattern.
# ---------------------------------------------------------------------------
@nki.jit
def bias_relu_kernel(x, b):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    # YOUR CODE HERE
    return out


# ---------------------------------------------------------------------------
# Exercise 0e: the simulator is not the hardware
#   This kernel is already complete and PASSES in the simulator. Run
#       python labs/L0_first_kernel/check.py -e 0e
#   and read the compiler's error. Then fix the kernel so it compiles. Keep the
#   two-instruction structure: one instruction for rows 0-63 and one for rows
#   64-127, each computing x * 2.
#
#   Rule to learn: an engine instruction may only START at partition 0, 32, 64
#   or 96. The simulator does not enforce this; the hardware does.
# ---------------------------------------------------------------------------
@nki.jit
def split_rows_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y[0:48, :], data=t[0:48, :], op0=nl.multiply, operand0=2.0)
    nisa.tensor_scalar(dst=y[48:128, :], data=t[48:128, :], op0=nl.multiply, operand0=2.0)
    nisa.dma_copy(dst=out, src=y)
    return out
