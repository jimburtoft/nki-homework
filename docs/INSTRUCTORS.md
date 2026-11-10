# Instructor notes

## Assigning work

- **Shortest path, one session:** Labs 0-4, then P1.
- **Medium:** Labs 0-4, then Lab 6, then P2.
- **Hard:** all labs, then P3.

The three projects are independent. You can assign one, two or all three, for
example P1 as a first homework and P2 or P3 later.

## Grading

```bash
python grader/grade.py P1 --file submissions/alice/kernel.py \
       --cases /path/to/instructor-repo/hidden_tests/P1.json --json alice.json
```

- **Isolation and limits:** each case runs in a fresh subprocess with a per-case
  timeout (`--timeout`, default 900 s).
- **Correctness:** checked on the CPU simulator against `reference.py` with a
  fixed tolerance.
- **Legality:** checked by compiling for trn2 with `neuronx-cc`.
- **Performance:** uses the compiler estimate on the cases marked `"perf": true`.
  Pass `--device` on a trn2 instance to use measured latency instead.
- **Thresholds:** performance thresholds (`full`, `zero`, in microseconds) are in
  `grader/projects.py`. A hidden-case file can override them with a `"perf"` key.
- **Output:** the `--json` output holds every case's result, the performance
  values and the score, for import into your gradebook.

Hidden case files have this format:

```json
{"cases": [{"name": "h1", "shape": {"T": 640, "D": 512, "F": 1024}, "seed": 102}],
 "perf":  {"full": 340.0, "zero": 4500.0}}
```

## Machine resources

- **Memory:** a P1 grading run (4 public plus 5 hidden cases) takes about 2-3
  minutes and under 4 GB of RAM on one core.
- **Parallelism:** grade many submissions in parallel, one process per core.

## Compiler-estimate caveats

- The estimate comes from the Neuron compiler's post-scheduling model of the
  trn2 NeuronCore.
- It tracks real hardware well for ranking versions of the same kernel, but it
  is not a measured runtime.
- Thresholds are calibrated for neuronx-cc 2.28 (Neuron SDK 2.33). If you pin a
  different compiler version, re-run the instructor solutions and re-set the
  thresholds.

## Gradescope

`grader/gradescope/` contains an experimental autograder wrapper. It has a
setup script, `run_autograder`, and a converter from `grade.py --json` output to
Gradescope `results.json`, which masks hidden-case details. See
[grader/gradescope/README.md](../grader/gradescope/README.md). The converter has
been tested locally; the wrapper has not yet been tested on Gradescope itself.
