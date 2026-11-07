# Notes for agents

## Environments

- `baseline` runs the latest scipy release, `main` a pinned scipy commit (see `pixi.toml`).
- `local` is an editable install of `../scipy` and rebuilds from whatever branch is checked out there on import. Check that branch before benchmarking with it.

## Where things go

- Every script goes into `scipy_bench/prototypes/`, even one-off experiments, never into a scratchpad. Run it with `pixi run -e <env> python -m scipy_bench.prototypes.<name>`.
- Reports go into `scipy_bench/reports/`, mirroring the scipy module path. Results and plots are regenerable and gitignored.
- The READMEs describe the harness for the whole library. Leave out anything that holds for one module only or is expected behavior.
- Write the READMEs in plain words a newcomer understands on first read. Name concrete things and say what they do, with no metaphors or clever phrasing.

## Benchmarking

- Benchmark only the versions and functions the user asked to compare.
- When the user says the machine is free, run immediately without waiting for it to calm down. When the user runs a test suite, wait for it to pass before benchmarking.
