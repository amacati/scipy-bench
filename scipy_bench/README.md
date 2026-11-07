# scipy_bench

This file explains how to add benchmarks and how they are timed. The top-level README
explains how to run them.

## Layout

Each scipy module has one suite file that holds its benchmark cases. Suites and their
outputs use the same paths as the scipy modules.

```
suites/spatial/distance.py                                   the cases
results/spatial/distance/<variant>/<xp>/<device>/<case>.json  raw timings
plots/spatial/distance/<case>.{png,svg}                       figures
reports/spatial/distance/{summary.md,benchmark.pdf}           summaries
```

## Adding a case

Create the suite file at the path of the scipy module and decorate each case with
`@register`. `bench` finds suite files on its own and uses the function name as the
case name. A case takes the framework, the device and the input size, and returns
three functions. `setup` creates the inputs, `test` is timed for numpy, torch and
cupy, and `jax_test` is timed for jax.

```python
import jax
import numpy as np

from scipy_bench import register, to_xp

@register
def my_case(xp, device, n_samples):
    data = None
    jitted = jax.jit(scipy_function) if xp == "jax" else None

    def setup():
        nonlocal data
        data = to_xp(xp, np.random.rand(n_samples), device)
        if xp == "jax":
            jax.block_until_ready(jitted(data))

    def test():
        return scipy_function(data)

    def jax_test():
        jax.block_until_ready(jitted(data))

    return setup, test, jax_test
```

When several cases differ only in the function they call, write one builder, fix the
function with `partial` and pass the case name.

```python
register("as_rotvec", partial(method_case, call=R.as_rotvec))
```

Pass `harder_than` with the name of a cheaper case to skip this case from the first
size at which the cheaper one failed.

## How cases are timed

**Setup runs before every repeat.** It runs outside the timed calls, so each repeat
gets fresh inputs. Assign the inputs with `nonlocal` as in the example. Otherwise
`test` keeps using the inputs of the first repeat.

**GPU calls are waited for.** torch and cupy return before the GPU finishes, so `run`
waits for the GPU after each call to `test`. jax has no such global wait, so
`jax_test` must call `jax.block_until_ready` on the result. `run` does not wait after
`setup`, so GPU work started there can add to the first timed call.

**jax is timed after compilation.** Create the jit-compiled function once per case and
call it once in `setup`, as in the example. That call compiles it, so compilation is
not timed.

**Stored timings.** Each result file holds the seconds per call of every repeat. The
figures show the mean with the standard deviation as error bars, and the report gives
the minimum and the mean.

**How many runs you need.** Repeats in one process vary less than separate runs, by
up to 9 percent on jax gpu. A significance test on the repeats of a single run can
therefore call a chance difference real. To detect a difference reliably, about 80
percent of the time, run each variant `15.7 * CV^2 / effect^2` times, where CV is the
run to run variation. At 9 percent that is 3 runs for a 20 percent difference, 12 for
10 percent and 50 for 5 percent. Collect them with `--append`.

```bash
for i in $(seq 20); do
    pixi run -e main bench run --module spatial/distance --fn braycurtis --xp jax \
        --device gpu --low 3 --high 3 --append
done
```

**Time limits.** `run` skips a size when its `repeat * number` calls would take more
than five minutes, or when `setup` takes more than an hour. It then also skips all
larger sizes for that framework and device, as it does after running out of memory or
an unsupported operation.
