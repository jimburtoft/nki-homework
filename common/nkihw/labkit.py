"""Shared runner for the self-check scripts in labs/.

Each lab's check.py builds a list of Exercise objects and calls run_lab().
The student's kernels live in that lab's lab.py, and --file points at another copy.
"""
import argparse
import importlib.util
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .compare import check
from .run import compile_kernel, simulate


@dataclass
class Exercise:
    name: str  # e.g. "0b"
    kernel: str  # function name in lab.py
    inputs: Callable[[], tuple]  # -> tuple of numpy arrays (positional kernel args)
    reference: Callable[..., np.ndarray]  # numpy reference, same args
    tol: dict = field(default_factory=lambda: {"atol": 1e-5})
    compile: bool = True  # also offline-compile for trn2 (legality + cost estimate)
    note: str = ""
    max_est_us: float | None = None  # performance goal: compiler estimate must be <= this
    custom: Callable | None = None  # optional fn(out, ref) -> CheckResult, replaces the tolerance check


def load_module(path):
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0] + "_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_lab(title, exercises, default_file, argv=None):
    ap = argparse.ArgumentParser(description=f"Self-check for {title}")
    ap.add_argument("--file", default=default_file, help="lab file to check (default: lab.py next to this script)")
    ap.add_argument("-e", "--exercise", action="append", help="only run these exercises (e.g. -e 1a -e 1b)")
    ap.add_argument("--no-compile", action="store_true", help="skip the offline compile step (faster)")
    args = ap.parse_args(argv)

    mod = load_module(args.file)
    print(f"== {title}  ({args.file})")
    n_pass = n_run = 0
    for ex in exercises:
        if args.exercise and ex.name not in args.exercise:
            continue
        n_run += 1
        k = getattr(mod, ex.kernel, None)
        print(f"\n-- Exercise {ex.name}: {ex.kernel}" + (f"  ({ex.note})" if ex.note else ""))
        if k is None:
            print(f"   FAIL  {ex.kernel} not found in {args.file}")
            continue
        inp = ex.inputs()
        ref = ex.reference(*inp)
        t0 = time.time()
        try:
            out = simulate(k, *inp)
        except Exception as e:  # noqa: BLE001
            print(f"   FAIL  simulator raised {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
            tb = traceback.extract_tb(e.__traceback__)
            for fr in tb:
                if os.path.abspath(fr.filename) == os.path.abspath(args.file):
                    print(f"         at {os.path.basename(fr.filename)}:{fr.lineno}: {fr.line}")
            continue
        refs = ref if isinstance(ref, tuple) else (ref,)
        outs = out if isinstance(out, tuple) else (out,)
        if len(outs) != len(refs):
            print(f"   FAIL  kernel returned {len(outs)} outputs, expected {len(refs)}")
            continue
        if ex.custom is not None:
            results = [ex.custom(o, r) for o, r in zip(outs, refs)]
        else:
            results = [check(o, r, name=f"out{i}" if len(refs) > 1 else "output", **ex.tol) for i, (o, r) in enumerate(zip(outs, refs))]
        ok = all(results)
        for r in results:
            print(f"   sim     {r}  [{time.time() - t0:.1f}s]")
        if ok and ex.compile and not args.no_compile:
            rep = compile_kernel(k, *inp)
            print("   trn2    " + rep.summary().replace("\n", "\n           "))
            ok = rep.ok
            if ok and ex.max_est_us is not None:
                met = rep.est_us is not None and rep.est_us <= ex.max_est_us
                print(f"   goal    estimate <= {ex.max_est_us:g} us: {'met' if met else 'NOT met'}")
                ok = met
        n_pass += ok
        print(f"   {'PASS' if ok else 'FAIL'}")
    print(f"\n== {title}: {n_pass}/{n_run} exercises pass")
    return 0 if n_pass == n_run else 1
