#!/usr/bin/env python
"""Convert grade.py --json output into Gradescope's results.json.

    python grader/gradescope/to_gradescope.py results_P1.json /autograder/results/results.json \
        [--weights correctness=50,legality=10,performance=25] [--show-hidden-details]

Write-up points (if any) are graded by hand in Gradescope, so they are not included here.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grade_json")
    ap.add_argument("out")
    ap.add_argument("--weights", default="correctness=50,legality=10,performance=25")
    ap.add_argument("--public-cases", default="", help="comma-separated case names that are public (shown in detail)")
    ap.add_argument("--show-hidden-details", action="store_true")
    a = ap.parse_args()
    w = {k: float(v) for k, v in (kv.split("=") for kv in a.weights.split(","))}
    g = json.load(open(a.grade_json))
    public = set(filter(None, a.public_cases.split(",")))
    res = g.get("results", [])
    n = max(1, len(res))
    tests = []
    for r in res:
        ok = bool(r.get("correct"))
        legal = ok and bool(r.get("compiled", True))
        shown = a.show_hidden_details or r["case"] in public
        name = r["case"] if shown else f"hidden case {r['case']}"
        out = r.get("check") or r.get("error") or ""
        if r.get("compiled") is False:
            out += f"\nCOMPILE FAILED: {r.get('compile_error')}\nhint: {r.get('compile_hint')}"
        tests.append({
            "name": f"{name}: correctness",
            "score": w["correctness"] / n if ok else 0.0,
            "max_score": w["correctness"] / n,
            "output": out if shown else ("passed" if ok else "failed (hidden case; check your kernel against the SPEC bounds)"),
            "visibility": "visible",
        })
        tests.append({
            "name": f"{name}: compiles for trn2",
            "score": w["legality"] / n if legal else 0.0,
            "max_score": w["legality"] / n,
            "visibility": "visible",
        })
    perf = g.get("perf_score") or 0.0
    vals = ", ".join("n/a" if v is None else f"{v:.1f} us" for v in g.get("perf_values_us", []))
    tests.append({
        "name": "performance",
        "score": round(w["performance"] * perf, 3),
        "max_score": w["performance"],
        "output": f"{g.get('perf_metric')}: {vals}; thresholds {g.get('perf_thresholds_us')}",
        "visibility": "visible",
    })
    json.dump({"tests": tests, "stdout_visibility": "visible"}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
