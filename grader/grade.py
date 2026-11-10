#!/usr/bin/env python
"""Autograder for the final projects.

    python grader/grade.py P1                           # public cases, simulator + compiler estimate
    python grader/grade.py P1 --no-compile              # correctness only (fast)
    python grader/grade.py P1 --file path/to/kernel.py  # grade another file
    python grader/grade.py P1 --cases hidden.json       # instructor: extra cases
    python grader/grade.py P1 --device                  # on trn2: also run + time on hardware
    python grader/grade.py P1 --json results.json       # machine-readable results

Scoring (each case is run in a fresh subprocess with a time limit):
  correctness  simulator output within the project tolerance, no NaN
  legality     compiles for trn2 with neuronx-cc (the compiler checks on-chip
               memory capacity and hardware rules the simulator does not)
  performance  compiler-estimated latency on the perf case(s) (or measured
               device latency with --device, which has its own thresholds),
               mapped to [0, 1]: 1 at or below "full", 0 at or above "zero",
               linear in log(time) in between
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from projects import PROJECTS  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_case(project, kernel_file, case, do_compile, device):
    """Runs inside the child process. Returns a result dict."""
    sys.path.insert(0, os.path.join(ROOT, "common"))
    import nkihw  # noqa: F401

    cfg = PROJECTS[project]
    pdir = os.path.join(ROOT, cfg["dir"])
    ref = _load(os.path.join(pdir, "reference.py"), f"{project}_reference")
    res = {"case": case["name"], "shape": case["shape"], "seed": case["seed"], "perf_case": bool(case.get("perf"))}
    try:
        mod = _load(kernel_file, f"{project}_submission")
        kern = getattr(mod, cfg["kernel"])
    except Exception as e:  # noqa: BLE001
        res.update(correct=False, error=f"could not import {cfg['kernel']} from {kernel_file}: {type(e).__name__}: {e}")
        return res
    args = ref.make_inputs(**case["shape"], seed=case["seed"])
    expected = ref.reference(*args)
    t0 = time.time()
    try:
        out = nkihw.simulate(kern, *args)
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()
        res.update(correct=False, error=f"simulator: {type(e).__name__}: {msg[0] if msg else ''}"[:500])
        return res
    res["sim_s"] = round(time.time() - t0, 2)
    chk = nkihw.check(out, expected, rel_l2=ref.TOL_REL_L2)
    res.update(correct=chk.ok, check=chk.msg, rel_l2=chk.rel_l2)
    if chk.ok and do_compile:
        rep = nkihw.compile_kernel(kern, *args)
        res.update(compiled=rep.ok, est_us=rep.est_us, instructions=rep.total_instructions, compile_error=rep.error, compile_hint=rep.hint)
    if chk.ok and device:
        try:
            dout = nkihw.run_device(kern, *args)
            dchk = nkihw.check(dout, expected, rel_l2=ref.TOL_REL_L2)
            res.update(device_correct=dchk.ok, device_check=dchk.msg)
            if dchk.ok and case.get("perf"):
                res["device_us"] = nkihw.bench_device(kern, *args)["mean_us"]
        except Exception as e:  # noqa: BLE001
            res.update(device_correct=False, device_check=f"{type(e).__name__}: {str(e)[:300]}")
    return res


def perf_score(us, full, zero):
    if us is None:
        return 0.0
    if us <= full:
        return 1.0
    if us >= zero:
        return 0.0
    return (math.log(zero) - math.log(us)) / (math.log(zero) - math.log(full))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", choices=sorted(PROJECTS))
    ap.add_argument("--file", help="kernel file (default: projects/<P>/kernel.py)")
    ap.add_argument("--cases", help="JSON file with extra cases (instructor hidden tests)")
    ap.add_argument("--only-extra", action="store_true", help="with --cases: skip the public cases")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", action="store_true", help="also run and time on a trn2 NeuronCore")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per case")
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--_child", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if os.environ.get("NKIHW_TARGET", "trn2") != "trn2":
        print("grade.py grades the trn2 target only; unset NKIHW_TARGET.")
        return 2
    cfg = PROJECTS[a.project]
    kfile = os.path.abspath(a.file or os.path.join(ROOT, cfg["dir"], "kernel.py"))

    if a._child:
        case = json.loads(a._child)
        print("@@RESULT@@" + json.dumps(run_case(a.project, kfile, case, not a.no_compile, a.device)))
        return 0

    cases = [] if a.only_extra else list(cfg["cases"])
    pkey = "perf_device" if a.device else "perf"
    perf_cfg = dict(cfg[pkey])
    if a.cases:
        extra = json.load(open(a.cases))
        cases += extra.get("cases", [])
        perf_cfg.update(extra.get(pkey, {}))

    print(f"== {a.project}: {cfg['title']}\n   file: {kfile}")
    results = []
    for case in cases:
        cmd = [sys.executable, os.path.abspath(__file__), a.project, "--file", kfile, "--_child", json.dumps(case)]
        cmd += ["--no-compile"] if a.no_compile else []
        cmd += ["--device"] if a.device else []
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("@@RESULT@@")]
            if line:
                r = json.loads(line[-1][len("@@RESULT@@"):])
            else:
                tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
                r = {"case": case["name"], "correct": False, "error": "crashed: " + " | ".join(tail)[:500]}
        except subprocess.TimeoutExpired:
            r = {"case": case["name"], "correct": False, "error": f"timed out after {a.timeout}s"}
        r["wall_s"] = round(time.time() - t0, 1)
        results.append(r)
        status = "PASS" if r.get("correct") and r.get("compiled", True) and r.get("device_correct", True) else "FAIL"
        shape = " ".join(f"{k}={v}" for k, v in case["shape"].items())
        print(f"\n-- {case['name']:<8} {shape:<24} {status}")
        if "check" in r:
            print(f"   sim      {r['check']}")
        if r.get("error"):
            print(f"   error    {r['error']}")
        if "compiled" in r:
            if r["compiled"]:
                print(f"   trn2     compiler estimate {r['est_us']:.1f} us ({r['instructions']} instructions)")
            else:
                print(f"   trn2     COMPILE FAILED: {r['compile_error']}")
                if r.get("compile_hint"):
                    print(f"            hint: {r['compile_hint']}")
        if "device_check" in r:
            print(f"   device   {r['device_check']}" + (f"; {r['device_us']:.1f} us" if r.get("device_us") else ""))

    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    legal = sum(1 for r in results if r.get("correct") and r.get("compiled")) if not a.no_compile else None
    perf_rs = [r for r in results if r.get("perf_case")]
    key = "device_us" if a.device else "est_us"
    perf_vals = [r.get(key) if r.get("correct") and r.get("compiled", True) else None for r in perf_rs]
    perf = None
    if not a.no_compile and perf_rs:
        perf = sum(perf_score(v, perf_cfg["full"], perf_cfg["zero"]) for v in perf_vals) / len(perf_vals)
    summary = {
        "project": a.project,
        "file": kfile,
        "cases": n,
        "correct": correct,
        "legal": legal,
        "perf_metric": key,
        "perf_values_us": perf_vals,
        "perf_score": perf,
        "perf_thresholds_us": perf_cfg,
        "results": results,
    }
    print(f"\n== {a.project} summary: correct {correct}/{n}", end="")
    if legal is not None:
        print(f", compiles {legal}/{n}", end="")
    if perf is not None:
        shown = ", ".join("n/a" if v is None else f"{v:.1f} us" for v in perf_vals)
        print(f", performance {perf:.2f} ({'device' if a.device else 'compiler estimate'}: {shown}; "
              f"full marks <= {perf_cfg['full']:g} us)", end="")
    print()
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1)
    return 0 if correct == n and (legal is None or legal == n) else 1


if __name__ == "__main__":
    sys.exit(main())
