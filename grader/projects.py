"""Project configuration for the grader.

Each project lists its PUBLIC test cases here. Instructors grade with extra
hidden cases by passing `--cases <file.json>`; see docs/INSTRUCTORS.md.

A case is {"name", "shape": {...make_inputs kwargs...}, "seed"}; "perf": true
marks a case used for the performance score.
"""
PROJECTS = {
    "P1": {
        "dir": "projects/P1_mlp",
        "kernel": "mlp_kernel",
        "title": "Fused MLP block (easy)",
        "cases": [
            {"name": "tiny", "shape": {"T": 128, "D": 128, "F": 256}, "seed": 1},
            {"name": "small", "shape": {"T": 256, "D": 256, "F": 512}, "seed": 2},
            {"name": "rect", "shape": {"T": 384, "D": 768, "F": 3072}, "seed": 3},
            {"name": "medium", "shape": {"T": 1024, "D": 1024, "F": 4096}, "seed": 4, "perf": True},
        ],
        # Performance score: compiler estimate (us) on the perf case(s). Full marks
        # at or below `full`, zero at or above `zero`, linear in between (log scale).
        # Values are set from the instructor solutions (neuronx-cc 2.28).
        "perf": {"full": 340.0, "zero": 4500.0},
    },
}
