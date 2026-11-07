# scipy-bench

Benchmarks for the scipy array API port. Each benchmark calls a scipy function with
numpy, torch, jax and cupy arrays, on cpu and gpu, for input sizes from 1 to 10^7.
This repo is separate from scipy, so you can switch branches in your scipy clone and
run the same benchmarks on each branch.

## Benchmark a scipy release or main

Each pixi environment in this repo installs a different scipy.

- `baseline` has the latest scipy release.
- `main` has a fixed commit of scipy main. To benchmark a newer commit, change `rev`
  under `feature.main` in `pixi.toml`.
- `local` has your scipy clone in `../scipy`. See the next section.

`--variant` names the folder the timings are stored in. `report` compares the variants
you pass to it.

```bash
cd ~/repos/scipy-bench
pixi run -e baseline bench run --module spatial/distance --variant baseline
pixi run -e main     bench run --module spatial/distance --variant main
pixi run -e main     bench report --module spatial/distance --variant main baseline
```

## Benchmark your scipy branch

The `local` environment builds your scipy clone in `../scipy`. Switch branches there
and run the same command for each branch.

```bash
cd ~/repos/scipy && git switch main
cd ~/repos/scipy-bench && pixi run -e local bench run --module spatial/distance --variant baseline

cd ~/repos/scipy && git switch my-feature
cd ~/repos/scipy-bench && pixi run -e local bench run --module spatial/distance --variant xp

pixi run -e local bench report --module spatial/distance --variant xp baseline
```

## Options and output

`bench list` shows all benchmark cases. `--module` selects cases by path, so `spatial`
selects everything under `scipy.spatial`, and `--fn` selects cases by name. By default,
`run` times sizes 10^0 to 10^7 with 5 repeats of 100 calls each. `bench run --help`
lists all options.

`run` stores timings in `scipy_bench/results`. `report` writes a summary to
`scipy_bench/reports` and figures to `scipy_bench/plots`. `plot` writes only the
figures. The report compares the first variant against each later one, and a speedup
above 1 means the first variant is faster. The figures draw the first variant dashed.

## Pitfalls

**Run each benchmark several times.** All repeats of one size run in the same process,
so they vary less than separate runs do. Between processes, timings differ by about 1
percent on numpy cpu and 9 percent on jax gpu. Run the same command several times with
`--append` to collect timings from separate processes.

**Delete old timings after you change code.** `run` keeps the stored timings of sizes
it does not measure again, and `--append` adds to the stored timings. Delete
`scipy_bench/results/<module>/<variant>` before you rerun, so old and new timings do
not mix.

**Compare frameworks only roughly.** jax is timed on a jit-compiled function, the
other frameworks on the plain function call. Comparing one framework before and after
a change is reliable.

**Missing sizes failed.** When a size takes too long, runs out of memory or is not
supported, `run` skips it and all larger sizes for that framework and device. The
reason is stored in a `.skip.json` file next to the timings.

**Do not run other work during a benchmark.** A compile or test suite running at the
same time slows the benchmark down, and the stored timings do not show it.

`scipy_bench/README.md` explains how to add benchmarks and how they are timed.
