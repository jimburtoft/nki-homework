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
        # Performance score: compiler estimate (us) on the perf case(s). 1.0 at or below
        # `full`, 0.0 at or above `zero`, interpolated linearly in log(time) between.
        # Calibrated on the instructor solutions (neuronx-cc 2.28). `full` sits at the
        # estimator's Tensor-Engine floor for this shape (~320 us in bf16), so it
        # needs a matmul-bound kernel; partial designs land between 0.7 and 0.97.
        "perf": {"full": 325.0, "zero": 4500.0},
        # With --device (trn2.3xlarge, measured): instructor solutions 1962 / 265 / 270 us.
        "perf_device": {"full": 270.0, "zero": 4000.0},
    },
    "P2": {
        "dir": "projects/P2_window_attention",
        "kernel": "window_attention",
        "title": "Sliding-window causal attention (medium)",
        "cases": [
            {"name": "tiny", "shape": {"T": 128, "d": 64, "W": 128}, "seed": 1},
            {"name": "small", "shape": {"T": 512, "d": 64, "W": 100}, "seed": 2},
            {"name": "mid", "shape": {"T": 2048, "d": 128, "W": 256}, "seed": 3},
            {"name": "long", "shape": {"T": 4096, "d": 128, "W": 512}, "seed": 4, "perf": True},
        ],
        "perf": {"full": 110.0, "zero": 1500.0},
        # Device (trn2.3xlarge): instructor solutions 880 / 210 / 158 us.
        "perf_device": {"full": 160.0, "zero": 1500.0},
    },
    "P3": {
        "dir": "projects/P3_mamba2_ssd",
        "kernel": "ssd_kernel",
        "title": "Mamba-2 SSD chunked scan (hard)",
        "cases": [
            {"name": "one", "shape": {"L": 128, "P": 64, "N": 64}, "seed": 1},
            {"name": "small", "shape": {"L": 512, "P": 64, "N": 128}, "seed": 2},
            {"name": "decay", "shape": {"L": 1024, "P": 128, "N": 64, "decay": 5.0}, "seed": 3},
            {"name": "long", "shape": {"L": 4096, "P": 128, "N": 128}, "seed": 4, "perf": True},
        ],
        "perf": {"full": 80.0, "zero": 250.0},
        # Device (trn2.3xlarge): instructor solutions 144 / 109 / 98 us.
        "perf_device": {"full": 100.0, "zero": 300.0},
    },
}
