"""nkihw -- harness for the NKI homework.

Import ``nkihw`` BEFORE ``nki``. Importing it pins the hardware target (trn2) and
the Logical NeuronCore setting (LNC=1), so the simulator, the offline compiler and
real hardware all see the same machine:

    import nkihw                      # must come first
    import nki, nki.language as nl, nki.isa as nisa

Public API:
    nkihw.simulate(kernel, *args)          run on the CPU simulator, return numpy output(s)
    nkihw.compile_kernel(kernel, *args)    offline-compile for trn2, return a CompileReport
                                           (device-legality check + compiler cost estimate)
    nkihw.run_device(kernel, *args)        run on a real NeuronCore (trn2 only)
    nkihw.bench_device(kernel, *args)      time on a real NeuronCore (trn2 only)
    nkihw.check(out, ref, ...)             compare an output to a reference
"""
from . import env as _env

_env.configure()

from .compare import CheckResult, check  # noqa: E402
from .run import (  # noqa: E402
    CompileReport,
    bench_device,
    compile_kernel,
    has_device,
    run_device,
    simulate,
)

__all__ = [
    "CheckResult",
    "CompileReport",
    "bench_device",
    "check",
    "compile_kernel",
    "has_device",
    "run_device",
    "simulate",
]
__version__ = "0.1.0"
