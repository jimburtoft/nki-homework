"""Running kernels three ways: CPU simulator, offline compile (cost estimate), device."""
import glob
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .env import TARGET, has_device

_SIM_LINE = re.compile(r"Time-aware simulation time:\s*([0-9]+)")
_ROW = re.compile(r"^\W*([A-Z_][A-Z0-9_()x]*)\W+([0-9]+)\W*$")


def _as_numpy_tuple(out):
    if isinstance(out, (tuple, list)):
        return tuple(np.asarray(o) for o in out)
    return np.asarray(out) if out is not None else None


def simulate(kernel, *args, **kwargs):
    """Run ``kernel`` on the NKI CPU simulator and return numpy output(s)."""
    import nki

    return _as_numpy_tuple(nki.simulate(kernel)(*args, **kwargs))


# ----------------------------------------------------------------------------
# Offline compile
# ----------------------------------------------------------------------------


@dataclass
class CompileReport:
    """Result of compiling a kernel for trn2 with the real Neuron compiler.

    ``est_ns`` is the compiler's own post-scheduling latency estimate for one
    execution of the kernel. It is an ESTIMATE: use it to compare two versions
    of your kernel, not as a measured time. Real timings come from hardware.
    """

    ok: bool
    est_ns: int | None = None
    instructions: dict = field(default_factory=dict)
    mac_count: int | None = None
    error: str | None = None
    hint: str | None = None
    artifacts_dir: str | None = None

    @property
    def est_us(self):
        return None if self.est_ns is None else self.est_ns / 1000.0

    @property
    def total_instructions(self):
        return sum(self.instructions.values())

    def summary(self, top=8) -> str:
        if not self.ok:
            s = "COMPILE FAILED: " + (self.error or "unknown error")
            if self.hint:
                s += "\n  hint: " + self.hint
            return s
        parts = [f"compiler estimate {self.est_us:.2f} us" if self.est_ns is not None else "no estimate"]
        if self.mac_count:
            parts.append(f"{self.mac_count / 1e6:.1f} M MACs")
        parts.append(f"{self.total_instructions} instructions")
        top_ops = sorted(self.instructions.items(), key=lambda kv: -kv[1])[:top]
        return ", ".join(parts) + "\n  top opcodes: " + ", ".join(f"{k}={v}" for k, v in top_ops)


_HINTS = [
    (r"Allocated memory out of bound.*SB", "SBUF capacity exceeded (about 208 KiB usable per partition on trn2). The simulator does not check this; tile smaller or reuse buffers."),
    (r"Allocated memory out of bound.*PSUM|PSUM.*out of bound", "PSUM capacity exceeded (16 KiB per partition = 8 banks of 2 KiB on trn2)."),
    (r"Allocated memory out of bound", "On-chip memory capacity exceeded (SBUF ~208 KiB / PSUM 16 KiB per partition on trn2). The simulator does not check this."),
    (r"start partition|partition offset|multiple of 32|quadrant", "On hardware, an engine access must start at partition 0, 32, 64 or 96."),
    (r"failed verification", "The kernel uses a pattern the hardware cannot execute (often a partition range that does not start at 0/32/64/96, or mismatched operand shapes). The simulator is more permissive than the device."),
    (r"no output|returns? nothing|no return|output.*not.*found|NCC_INKI003|Insufficient number of outputs", "The kernel must `return` its output tensor(s)."),
]


def _hint(text: str):
    for pat, h in _HINTS:
        if re.search(pat, text, flags=re.I | re.S):
            return h
    return None


def _parse_instruction_stats(art: str) -> dict:
    counts = {}
    for f in glob.glob(os.path.join(art, "sg*", "instruction_stats.txt")):
        for line in open(f, errors="replace"):
            m = _ROW.match(line.strip())
            if m and m.group(1) != "Opcode":
                counts[m.group(1)] = counts.get(m.group(1), 0) + int(m.group(2))
    return counts


def _parse_estimate(art: str):
    vals = []
    for f in glob.glob(os.path.join(art, "**", "log-neuron-cc.txt"), recursive=True):
        for line in open(f, errors="replace"):
            m = _SIM_LINE.search(line)
            if m:
                vals.append(int(m.group(1)))
    # One line per NeuronCore program; with LNC=1 there is one. Take the slowest.
    return max(vals) if vals else None


_NCC_CODE = re.compile(r"\[(NCC_[A-Z0-9]+)\]\s*(.*?)(?: - Please open.*)?$")


def _log_errors(art: str) -> str:
    """Return the one or two lines of the compiler log that explain a failure."""
    out = []
    for f in glob.glob(os.path.join(art, "**", "log-neuron-cc.txt"), recursive=True):
        for line in open(f, errors="replace"):
            m = _NCC_CODE.search(line)
            if m:
                msg = f"[{m.group(1)}] {m.group(2).strip()}"
                msg = re.sub(r"#Internal DebugInfo:.*", "", msg).strip()
                if msg not in out:
                    out.append(msg)
            elif "Assertion failure:" in line:
                msg = line.split("Assertion failure:", 1)[1].strip()
                if msg not in out:
                    out.append("assertion: " + msg)
    return "\n".join(out[:3])


def compile_kernel(kernel, *args, keep_artifacts=None, **kwargs) -> CompileReport:
    """Compile ``kernel`` for the target (trn2 unless NKIHW_TARGET=trn3) with neuronx-cc (no device needed).

    This catches errors the simulator does not (on-chip memory capacity,
    hardware access-pattern rules, a missing ``return``) and returns the
    compiler's cost estimate.
    """
    from nki.framework.compiled import StandaloneKernel

    art = keep_artifacts or tempfile.mkdtemp(prefix="nkihw_cc_")
    if keep_artifacts:
        shutil.rmtree(art, ignore_errors=True)
        os.makedirs(art)
    captured = {}

    def _capture(compiled, inputs, outputs):
        captured["c"] = compiled

    prev = os.environ.get("NKI_ARTIFACTS_DIR")
    os.environ["NKI_ARTIFACTS_DIR"] = art
    try:
        try:
            k = kernel._to_subclass(StandaloneKernel, _executor=_capture)
            k(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- report every compile failure to the student
            msg = str(e).strip()
            log = _log_errors(art)
            first = msg.splitlines()[0] if msg else type(e).__name__
            detail = first if not log else first + "\n  " + log.replace("\n", "\n  ")
            if "nki.runtime is not available" in msg and os.path.isfile(os.path.join(art, "kernel.neff")):
                pass  # compiled fine; only the (absent) runtime failed
            else:
                return CompileReport(False, error=detail, hint=_hint(msg + "\n" + log), artifacts_dir=art if keep_artifacts else None)
        c = captured.get("c")
        rep = CompileReport(
            True,
            est_ns=_parse_estimate(art),
            instructions=_parse_instruction_stats(art),
            mac_count=getattr(getattr(c, "bir", None), "mac_count", None) if c else None,
            artifacts_dir=art if keep_artifacts else None,
        )
        return rep
    finally:
        if prev is None:
            os.environ.pop("NKI_ARTIFACTS_DIR", None)
        else:
            os.environ["NKI_ARTIFACTS_DIR"] = prev
        if not keep_artifacts:
            shutil.rmtree(art, ignore_errors=True)


# ----------------------------------------------------------------------------
# Device (trn2)
# ----------------------------------------------------------------------------


def _require_device():
    if not has_device():
        raise RuntimeError(f"No Neuron device found (/dev/neuron0). Device runs need a {TARGET} instance.")


def run_device(kernel, *args, **kwargs):
    """Compile and run ``kernel`` on NeuronCore 0; return numpy output(s)."""
    _require_device()
    return _as_numpy_tuple(kernel(*args, **kwargs))


def bench_device(kernel, *args, warmup=5, iterations=20, **kwargs) -> dict:
    """Time ``kernel`` on NeuronCore 0 with device-side tracing.

    Returns {"mean_us", "min_us", "max_us", "std_us", "iterations"}.
    """
    _require_device()
    from nki.framework.compiled import StandaloneKernel

    res = {}

    def _bench(compiled, inputs, outputs):
        r = compiled.benchmark(warmup=warmup, iterations=iterations, **inputs)
        res["r"] = r
        r.copy_to(compiled.output_specs, outputs)

    k = kernel._to_subclass(StandaloneKernel, _executor=_bench)
    k(*args, **kwargs)
    r = res["r"]
    return {
        "mean_us": r.latency * 1e6,
        "min_us": r.latency_min * 1e6,
        "max_us": r.latency_max * 1e6,
        "std_us": r.latency_std * 1e6,
        "iterations": iterations,
    }
