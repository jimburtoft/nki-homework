# NKI Homework: programming the AWS Trainium NeuronCore

Hands-on labs and final projects for learning **NKI** (Neuron Kernel
Interface), the low-level kernel language for AWS Trainium. Students write real
kernels that use the Trainium2 memory hierarchy (HBM, SBUF, PSUM), its four
compute engines, and its 128x128 systolic-array Tensor Engine.

**No Trainium hardware is needed.** Everything runs on a normal Linux x86-64
machine:

- the **NKI CPU simulator** runs your kernel and checks its output;
- the **Neuron compiler** (`neuronx-cc`) compiles it for trn2, checks the rules
  the simulator does not enforce, and gives a **latency estimate** to optimize
  against.

If you do have a trn2 instance, the same scripts also run on the device and
measure real latency.

## Contents

| | Topic | Time | Needed for |
|---|---|---|---|
| [Lab 0](labs/L0_first_kernel/lab.py) | memory model, DMA, engine instructions, the kernel must `return`, simulator vs hardware | ~30 min | all projects |
| [Lab 1](labs/L1_tiling/lab.py) | 128-partition tiling, partial tiles, SBUF capacity, bf16 | ~40 min | all projects |
| [Lab 2](labs/L2_engines/lab.py) | Vector/Scalar engines, reductions, per-partition scalars, RMSNorm, softmax | ~40 min | all projects |
| [Lab 3](labs/L3_matmul/lab.py) | Tensor Engine: 128x128 matmul, fp32 PSUM accumulation, tiled GEMM, transposes, fused epilogues | ~60 min | all projects |
| [Lab 4](labs/L4_performance/lab.py) | performance: DMA size, overlap, fusion, data reuse (compiler estimate) | ~40 min | all projects |
| [Lab 5](labs/L5_precision/lab.py) | precision: fp32 vs bf16 accumulation, fp8 `double_row`, row tiling, stochastic rounding (software; hardware SR as a trn2 bonus) | ~60 min | optional for P1/P2, P3 |
| [Lab 6](labs/L6_structure/lab.py) | masks, block skipping, the hardware scan instruction, prefix sums as matmuls, M:N structured sparsity (written) | ~60 min | P2, P3 |
| Lab 7 | optional trn3 exploration (MX formats) in the simulator, ungraded | *coming* | - |

| Final project | Difficulty | Prerequisites |
|---|---|---|
| [P1: fused MLP block](projects/P1_mlp/SPEC.md) | easy (2-4 h) | Labs 0-4 |
| [P2: sliding-window causal attention](projects/P2_window_attention/SPEC.md) | medium (4-6 h) | Labs 0-4, 6 (5 optional) |
| [P3: Mamba-2 SSD chunked scan](projects/P3_mamba2_ssd/SPEC.md) | hard (6-10 h) | Labs 0-6 |

Each project can be assigned alone. The labs are not graded; each lab's
`check.py` tells you whether your answers are right. Only the final projects are
graded, with `grader/grade.py`.

## Setup (about 5 minutes)

Requirements: Linux x86-64 and **Python 3.12**. The NKI wheel is built for
cp312 / linux_x86_64 only.

```bash
git clone <this repo> nki-homework && cd nki-homework
python3.12 -m venv .venv && source .venv/bin/activate
pip install --index-url https://pip.repos.neuron.amazonaws.com \
            --extra-index-url https://pypi.org/simple \
            "nki==0.7.0" "neuronx-cc==2.28.*" numpy ml_dtypes
python common/selftest.py          # should end with: SELFTEST PASS
```

The package named `nki` on plain PyPI is a placeholder. You must use the Neuron
index as shown above.

On a trn2 instance with the AWS Neuron Deep Learning AMI (SDK 2.33), activate a
venv that already has `nki` and `neuronx-cc` instead, for example
`source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate`.

## Workflow

```bash
python labs/L0_first_kernel/check.py           # check every exercise in a lab
python labs/L0_first_kernel/check.py -e 0b     # one exercise
python labs/L0_first_kernel/check.py --no-compile   # skip the compiler (faster)

python grader/grade.py P1                      # grade your project (public tests)
python grader/grade.py P1 --device             # on trn2: also run and time on hardware
```

Starter kernels allocate their output and return it **unwritten**. The
simulator fills unwritten memory with NaN, so every check fails until you write
the kernel.

### Simulator, compiler, and hardware

| | CPU simulator | Compiler (`neuronx-cc`) | trn2 hardware |
|---|---|---|---|
| Correct output | yes | - | yes |
| SBUF/PSUM capacity | **not checked** | checked | enforced |
| Partition start 0/32/64/96, kernel must `return` | **not checked** | checked | enforced |
| Performance | no | **estimate** | measured |
| Stochastic rounding (`NEURON_RT_STOCHASTIC_ROUNDING_EN=1`) | ignored (always round-to-nearest-even) | - | yes |

The compiler estimate is good for comparing two versions of your kernel. It is
not a measured runtime.

## For instructors

Solutions (lab answers plus several solution tiers per project), hidden test
cases, grading thresholds and a TA guide are in a separate **private instructor
repository**. To request access, contact the maintainer listed in
[MAINTAINERS.md](MAINTAINERS.md) from your institutional email and give your
course name.

The grader accepts extra hidden test cases (`--cases hidden.json`) and writes
machine-readable results (`--json`). See [docs/INSTRUCTORS.md](docs/INSTRUCTORS.md).

## Versions

Tested with nki 0.7.0 and neuronx-cc 2.28.7631.0 (AWS Neuron SDK 2.33), Python
3.12, target trn2. Compiler estimates and performance goals are calibrated for
this compiler version.

## License

Apache License 2.0. See [LICENSE](LICENSE).
