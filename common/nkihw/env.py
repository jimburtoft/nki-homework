"""Environment pinning. Must run before ``import nki``.

Why this matters:
  * ``@nki.jit`` binds the hardware target when the decorator runs. On a machine
    without Neuron devices NKI otherwise falls back to the newest generation
    (trn3), so a kernel that "works" there may not be legal on trn2.
  * The homework uses one Logical NeuronCore (LNC=1) everywhere, so the simulator,
    the offline compiler, and the device all run the same program.
"""
import os
import sys

TARGET = "trn2"


def has_device() -> bool:
    return os.path.exists("/dev/neuron0")


def configure() -> None:
    if "nki" in sys.modules and os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE") != TARGET:
        raise RuntimeError(
            "nkihw must be imported before nki (the target is fixed when @nki.jit runs). "
            "Put `import nkihw` above `import nki`."
        )
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = TARGET
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    os.environ.setdefault("NKI_LNC1_SHARED_HBM", "0")
    os.environ.setdefault("NKI_DISABLE_COMPILE_CACHE", "1")
    if not has_device():
        # Lets the offline compiler build a trn2 NEFF on a CPU-only machine.
        os.environ.setdefault("NEURON_RT_FAKE_INSTANCE_TYPE", "trn2.3xlarge")
    # The NKI driver runs `neuronx-cc` as a subprocess; make sure the copy that is
    # installed next to this Python interpreter is found even if the venv is not
    # activated.
    bindir = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if bindir not in path.split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + path
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module=r"nki\..*")
