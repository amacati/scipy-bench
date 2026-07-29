# scipy-bench

Benchmarks for the scipy array API port. Every case is timed on numpy, torch, jax and
cupy, on cpu and gpu, over a sweep of sample sizes.

The harness lives outside the scipy checkout, so it stays put when you switch branches.

## Setup

Once per scipy checkout, to build the `all-frameworks` pixi environment:

```bash
cd ~/repos/scipy
~/repos/scipy-bench/scipy_bench/setup_env.sh
```

Re-run it whenever `pixi.toml` is synced from main, which wipes the environment.

## Commands

All of them run from the scipy checkout, through `spin` so that the numbers come from
the code you are standing on:

```bash
cd ~/repos/scipy
BENCH="pixi run -e all-frameworks spin run --build-dir=build-all-frameworks python ~/repos/scipy-bench/scipy_bench/bench.py"

$BENCH list                                          # what can be measured
$BENCH run    --module spatial/distance --variant xp # measure
$BENCH plot   --module spatial/distance --variant baseline xp
$BENCH report --module spatial/distance --variant baseline xp
```

`--module` takes a path prefix, so `spatial` covers everything under it. `--fn` picks a
single case. `report` draws the figures itself, so you only need `plot` if you want
figures without a report.

Useful options for `run`:

| option | default | |
| --- | --- | --- |
| `--xp` | all four | `numpy`, `torch`, `jax`, `cupy` |
| `--device` | both | `cpu` or `gpu` |
| `--low` / `--high` | 0 / 7 | smallest and largest size, as powers of ten |
| `--repeat` | 5 | measurements per size |
| `--variant` | `current` | which result set to write |
| `--append` | off | add to the stored numbers instead of replacing them |

## Comparing two versions

Measure the reference, measure the change, compare:

```bash
git switch main
$BENCH run --module spatial/distance --variant baseline

git switch my-feature
$BENCH run --module spatial/distance --variant xp

$BENCH report --module spatial/distance --variant baseline xp
```

The order of `--variant` matters. The first is drawn dashed, the second solid, and the
report divides the second into the first.

Results, figures and reports land next to the harness, under `scipy_bench/results`,
`scipy_bench/plots` and `scipy_bench/reports`. None of them are committed.

## Gotchas

**One run proves very little.** Repeats inside a single run share a process and a warm
cache, so they look far more consistent than reality. Running the same thing again in a
fresh process moves the answer by about 1 percent on numpy cpu and about 9 percent on
jax gpu. Anything smaller than that is not a result. Use `--append` and run the command
several times to collect numbers across processes.

**Wipe a variant when the code changed.** `--append` only ever adds, so old and new
timings quietly end up in the same pile. Delete the directory first:
`rm -rf scipy_bench/results/spatial/distance/xp`.

**jax is not measured like the others.** Its cases are timed on a compiled function
while the rest are timed on the plain python call, and jax works in single precision
unless you turn x64 on. Comparing frameworks against each other is therefore rough.
Comparing the same framework before and after a change is fine.

**Missing large sizes mean too slow, not skipped.** A case that would blow the five
minute budget is dropped along with every larger size for that framework and device.
The same happens when a backend runs out of memory or cannot run the function at all.

**Sample size means different things.** Vector length for the scalar metrics, batch
size for `pdist` and `cdist`, matrix side for the distance matrix checks.

**Keep the machine quiet.** A compile or a test suite running alongside will show up in
the timings, and nothing in the stored data says it happened.

## Adding a case

Put the module where it mirrors the scipy one, then register it. Discovery is
automatic.

```python
from scipy_bench import register, to_xp

@register
def my_case(xp, device, n_samples):
    data = None

    def setup():
        nonlocal data
        data = to_xp(xp, np.random.rand(n_samples), device)

    def test():
        return scipy_function(data)

    def jax_test():
        jax.block_until_ready(scipy_function(data))

    return setup, test, jax_test
```

`setup` runs again before every measurement, which is why it rebinds with `nonlocal`
rather than returning the data. `scipy_bench/README.md` explains the timing rules in
full, and is worth reading before trusting a number.
