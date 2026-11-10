# Using the compiler as a test (on your CPU)

The simulator tells you whether your kernel computes the right numbers. The
**Neuron compiler** tells you whether the kernel can run on real trn2 hardware,
and roughly how fast it would be. Both run on your laptop or any Linux box;
neither needs Trainium.

Every `check.py` and `grade.py` runs both by default. This page explains what
the compiler step does and how to call it yourself.

## What the compiler catches that the simulator does not

| Problem | Simulator | Compiler |
|---|---|---|
| Too much SBUF (more than ~208 KiB per partition) or PSUM (16 KiB per partition) | runs fine | `COMPILE FAILED ... Allocated memory out of bound` with a hint |
| An engine instruction starting at a partition other than 0/32/64/96 | runs fine | `Generated MLIR failed verification` |
| Kernel does not `return` its output | runs fine | `NCC_INKI003 Insufficient number of outputs` |
| An operand in a memory the instruction cannot use (e.g. PSUM where SBUF is required) | usually caught too | caught |

Kernels that pass the simulator but fail the compiler **will not run on
hardware**. The grader gives no legality credit for a case that does not
compile.

## What the compiler tells you about speed

A successful compile prints a line like

```
trn2    compiler estimate 27.59 us, 268.4 M MACs, 86 instructions
          top opcodes: LDWEIGHTS=32, MATMUL=32, EVENT_SEMAPHORE=14, COPY=4, ...
```

- **compiler estimate** is the compiler's own model of how long one execution
  takes on a trn2 NeuronCore, after it has scheduled every instruction onto the
  engines and DMA queues. It is an estimate, not a measurement. Use it to
  compare two versions of your kernel. On the instructor solutions it ranked
  versions in the same order as real trn2 hardware, with absolute times within
  roughly 20-60%.
- **MACs** is the number of multiply-accumulates your matmuls perform. It is
  useful for checking that block skipping actually removed work.
- **instructions / top opcodes** count what the compiler emitted, for example
  `MATMUL`, `LDWEIGHTS` (loading a stationary operand), `ACTIVATE` (the Scalar
  engine) and `TENSOR_TENSOR` (the Vector engine). Fewer, larger instructions
  are usually faster.

## Running it yourself

From the command line:

```bash
python labs/L3_matmul/check.py -e 3b                         # simulate + compile one exercise
python labs/L3_matmul/check.py -e 3b --keep-artifacts out/   # also keep the compiler's files
python grader/grade.py P1                                    # every public case, simulate + compile
python grader/grade.py P1 --no-compile                       # simulator only (faster while debugging)
```

From your own Python script:

```python
import nkihw                      # must come before `import nki`
import numpy as np
from kernel import mlp_kernel     # your @nki.jit kernel

args = ...                        # numpy inputs, same as for the simulator
out = nkihw.simulate(mlp_kernel, *args)          # 1. correct numbers?
rep = nkihw.compile_kernel(mlp_kernel, *args)    # 2. legal on trn2? how fast?
print(rep.summary())
print(rep.ok, rep.est_us, rep.instructions)      # fields you can use in your own experiments

rep = nkihw.compile_kernel(mlp_kernel, *args, keep_artifacts="out/p1")   # keep the files
```

`compile_kernel` sets up everything the offline compile needs: target trn2,
one logical NeuronCore, and a fake trn2.3xlarge instance type so the runtime
check passes on a CPU-only machine. It does not touch any device. A typical
compile takes 2-10 seconds.

## The compiler's own files

With `--keep-artifacts DIR` (or `keep_artifacts=`), the directory contains
the compiler outputs. The useful ones:

| File | What is in it |
|---|---|
| `log-neuron-cc.txt` | the full compiler log. The estimate is the line `Time-aware simulation time: <ns>`. Errors appear as `[NCC_...]` lines |
| `sg00/instruction_stats.txt` | table of emitted instructions by opcode |
| `sg00/dma_stats.txt` | DMA descriptors by queue and transfer size |
| `kernel.neff` | the compiled program; this is what would be loaded onto a NeuronCore |

## Simulator vs compiler vs hardware

| Question | Use |
|---|---|
| Is my output correct? | simulator (`nkihw.simulate`, every `check.py`) |
| Will it run on trn2? | compiler (`nkihw.compile_kernel`, on by default in `check.py` / `grade.py`) |
| Is version B faster than version A? | compiler estimate (CPU) or `bench_device` (trn2) |
| How fast is it, really? | trn2 hardware only (`grade.py --device`, see [HARDWARE.md](HARDWARE.md)) |
| Stochastic rounding (`NEURON_RT_STOCHASTIC_ROUNDING_EN=1`) | trn2 hardware only; the simulator always rounds to nearest even |
