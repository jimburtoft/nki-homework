# Gradescope autograder (experimental)

These files package `grader/grade.py` as a Gradescope programming-assignment
autograder. They have not been tested on Gradescope yet. Test them with
Gradescope's "Test Autograder" before using them for a class.

## Build the autograder zip

```bash
mkdir -p ag/source && cp -r common grader projects ag/source/
mkdir -p ag/source/hidden && cp /path/to/instructor-repo/hidden_tests/P1.json ag/source/hidden/
cp grader/gradescope/setup.sh grader/gradescope/run_autograder ag/
chmod +x ag/run_autograder ag/setup.sh
(cd ag && zip -r ../autograder_P1.zip .)
```

## Configure the assignment

- Students upload `kernel.py`.
- To grade another project, set the default value of `PROJECT` at the top of
  `run_autograder` before you build the zip.
- Use at least 4 GB of RAM and a 30-minute timeout. A full P1/P2/P3 run with
  hidden cases takes 2-5 minutes.

## Scoring

- Points default to 50 for correctness, 10 for legality and 25 for performance.
- To change the split, edit the `--weights` argument in `run_autograder`.
- Add the write-up as a separate manually graded question.

## Hidden cases

- Results for public cases are shown in full.
- Hidden cases show only "passed" or "failed".
- The hidden-case JSON goes inside the zip, which only the instructor can see.
  Never add it to the public repo.

## Local test without Gradescope

Simulate the directory layout:

```bash
python grader/grade.py P1 --file my_kernel.py --json /tmp/g.json
python grader/gradescope/to_gradescope.py /tmp/g.json /tmp/results.json --public-cases tiny,small,rect,medium
```
