# Running on trn2 hardware (optional)

Every lab and project can be completed on the CPU simulator alone. If you have a
trn2 instance, you can also run your kernels on a real NeuronCore and measure
them. You need this for the Lab 5 hardware stochastic-rounding bonus and for
`grade.py --device`.

## Setup on a trn2 instance

Use the AWS **Deep Learning AMI Neuron (Ubuntu 24.04), SDK 2.33**. A
trn2.3xlarge is enough.

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
pip install --index-url https://pip.repos.neuron.amazonaws.com nrtpy   # NKI's device runtime bindings
git clone https://github.com/jimburtoft/nki-homework.git && cd nki-homework
python common/selftest.py
```

The SDK 2.33 venvs include `nki` and `neuronx-cc`, but not `nrtpy`. `nrtpy` is
the Python binding to the Neuron Runtime that NKI uses to run a standalone
kernel on the device. Without it, device runs fail with
`nki.runtime is not available on this platform`. `nkihw` loads
`/opt/aws/neuron/lib/libnrt.so.1` for you.

## Commands

```bash
python grader/grade.py P1 --device          # correctness on simulator AND device, plus device latency
python labs/L5_precision/bonus_hw_sr.py     # round-to-nearest-even
NEURON_RT_STOCHASTIC_ROUNDING_EN=1 python labs/L5_precision/bonus_hw_sr.py   # hardware stochastic rounding
```

From your own scripts:

```python
import nkihw
out = nkihw.run_device(my_kernel, *inputs)           # numpy outputs
t   = nkihw.bench_device(my_kernel, *inputs)          # {"mean_us", "min_us", "max_us", "std_us", ...}
```

`bench_device` times the kernel on the device itself (warm-up iterations, then
timed iterations). Each kernel launch has a fixed overhead of roughly 10 us, so
very small kernels measure about 14-15 us however little work they do.

## Simulator vs device: what to expect

Measured on trn2.3xlarge with the instructor solutions:

- Every lab and project solution that passes on the simulator also passes on
  the device, within the same tolerances.
- Outputs are often bit-identical. Elementwise math, DMA and masks give the
  same bits. Matmuls, reductions and transcendental functions can differ in the
  last bits.
- The compiler estimate tracks device time well for larger kernels:
  - Lab 4, `exp` over [2048, 2048]: estimate 111 us, device 92 us.
  - Lab 1c: estimate 193 us, device 174 us.
- For tiny kernels the device number is dominated by the launch overhead.
