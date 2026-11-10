# Setup on any Linux machine (no Trainium needed)

Both tools in this homework run on an ordinary x86-64 Linux CPU:

- the **NKI simulator** (`nki.simulate`) runs your kernel in NumPy;
- the **Neuron compiler** (`neuronx-cc`) compiles your kernel for trn2 exactly
  as it would on a Trainium machine, and reports errors and a latency estimate.

## Minimum requirements

| | Requirement | Why |
|---|---|---|
| CPU / OS | **x86-64 Linux** (glibc). No GPU or Trainium needed | the wheels are built for linux_x86_64 only |
| Python | **3.12** exactly | the `nki` wheel is cp312 only |
| System library | **libarchive** (`libarchive.so.13`) | needed by the compiler back end (`walrus_driver`). Without it, every compile fails with `exit code 70` and the log says `libarchive.so.13: cannot open shared object file` |
| Python packages | `nki==0.7.0`, `neuronx-cc==2.28.*`, `numpy`, `ml_dtypes` | all four from the commands below |
| Disk / RAM | ~2.5 GB for the venv; 4 GB RAM is enough (8 GB recommended) | the compiler package is large |

Nothing else is needed: no compiler toolchain, no Neuron driver, and no root
access beyond installing Python and libarchive.

macOS and Windows: use a Linux VM, WSL2 (Ubuntu 24.04), or the Docker image
below.

## Install commands (tested)

Pick your distribution. Each block was tested in a clean container and ends
with `SELFTEST PASS`.

**Ubuntu 24.04**
```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv libarchive13t64
```

**Ubuntu 22.04** (Python 3.12 from the deadsnakes PPA)
```bash
sudo apt-get update && sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv libarchive13
```

**Debian 12** (or the `python:3.12-slim` Docker image, which already has Python 3.12)
```bash
sudo apt-get update && sudo apt-get install -y libarchive13
```

**Amazon Linux 2023 / Fedora / RHEL 9 family**
```bash
sudo dnf install -y python3.12 libarchive
```

Then, on every distribution:

```bash
git clone https://github.com/jimburtoft/nki-homework.git && cd nki-homework
python3.12 -m venv .venv && source .venv/bin/activate
pip install --index-url https://pip.repos.neuron.amazonaws.com \
            --extra-index-url https://pypi.org/simple \
            "nki==0.7.0" "neuronx-cc==2.28.*" numpy ml_dtypes
python common/selftest.py        # last line must be: SELFTEST PASS
```

The selftest runs one kernel in the simulator and compiles it. It then checks
that the compiler rejects three kernels the simulator accepts: an SBUF overflow,
a bad partition start, and a missing `return`. If the simulator lines pass but
`compile good` fails, libarchive is missing.

The package called `nki` on plain PyPI is an empty placeholder. Always install
from the Neuron index, as shown above.

## Docker (any OS with Docker)

```bash
docker build -t nki-homework .
docker run --rm -it -v "$PWD":/hw -w /hw nki-homework python common/selftest.py
docker run --rm -it -v "$PWD":/hw -w /hw nki-homework python labs/L0_first_kernel/check.py
```

The `Dockerfile` in the repository root uses `python:3.12-slim`, installs
`libarchive13`, then the four Python packages. Mounting the repo means your
edits on the host are what gets tested. On Apple Silicon add
`--platform linux/amd64` to both `docker` commands (slower, through emulation).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: No matching distribution found for nki==0.7.0` | You are not on Python 3.12 or not on x86-64 Linux, or the `--index-url` is missing |
| `ImportError: ... WRONG PACKAGE` or `nki` has no `jit` | You installed the PyPI placeholder; reinstall with the Neuron index |
| Every compile fails with `exit code 70`, simulator works | Install libarchive (see the table above) |
| `neuronx-cc: command not found` during a compile | Run from the venv (`source .venv/bin/activate`); `nkihw` adds the venv's `bin/` to `PATH`, but only for the interpreter that has `neuronx-cc` installed |
| `nkihw must be imported before nki` | Put `import nkihw` above `import nki` in your file |
