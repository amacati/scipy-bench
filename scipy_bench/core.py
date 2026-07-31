"""Timing engine, backend helpers and result storage shared by every suite.

Read scipy_bench/README.md before changing anything in here. The timing contract is
subtle and the numbers on disk are only comparable while it holds.
"""

import json
import multiprocessing
import resource
import timeit
import traceback
from pathlib import Path

import cupy
import jax
import numpy as np
import torch
from scipy._lib._array_api import array_namespace

FRAMEWORKS = ["numpy", "torch", "jax", "cupy"]
DEVICES = ["cpu", "gpu"]
SKIP_XP_DEVICES = [("numpy", "gpu"), ("cupy", "cpu")]
TIMEOUT = 60 * 5
COMPILE_TIMEOUT = 60 * 60
ROOT = Path(__file__).parent


class CallTimeout(Exception):
    """A single call took longer than the sweep's per size budget allows."""


class CompileTimeout(Exception):
    """Setup took longer than COMPILE_TIMEOUT, which on jax means XLA autotuning.

    Note:
        The budget is checked after setup returns. XLA compiles inside a C++ call that
        defers Python signals until it finishes, so an in-flight compile cannot be
        preempted from this process. Skipping the remaining sizes is what the budget
        buys, not aborting the compile that overran it.
    """


def enable_float64():
    """Let jax keep float64 arrays, which it otherwise downcasts to float32."""
    jax.config.update("jax_enable_x64", True)


def check_float64(*arrays):
    """Assert that arrays are float64, once float64 has been asked for.

    Cases call this on the inputs they build and on the outputs of the function they
    time, since jax narrows both to float32 without saying so.
    """
    if not jax.config.jax_enable_x64:
        return
    for array in arrays:
        xp = array_namespace(array)
        assert array.dtype == xp.float64, f"want float64, got {array.dtype}"


def to_xp(xp, array, device):
    """Move a numpy array to `xp` on `device`.

    Args:
        xp: Framework name, one of FRAMEWORKS.
        array: Source numpy array.
        device: "cpu" or "gpu".

    Returns:
        The array in the framework's own type, resident on `device`.
    """
    if xp == "numpy":
        return array
    if xp == "torch":
        return torch.from_numpy(array).to("cuda" if device == "gpu" else "cpu")
    if xp == "jax":
        return jax.numpy.asarray(array, device=jax.devices(device)[0])
    if xp == "cupy":
        assert device == "gpu", "cupy runs on gpu only"
        return cupy.asarray(array)
    raise ValueError(f"Unknown framework {xp}")


def to_numpy(array, xp):
    """Bring an array from `xp` back to numpy."""
    if xp == "numpy":
        return array
    if xp == "torch":
        return array.detach().cpu().numpy()
    if xp == "cupy":
        return cupy.asnumpy(array)
    return np.asarray(array)


def device_of(array):
    """Report which device an array lives on, as "cpu" or "gpu"."""
    return "gpu" if "cuda" in str(getattr(array, "device", "cpu")).lower() else "cpu"


def timed_call(xp, device, test, jax_test):
    """Pick the timed callable, synchronizing GPU work.

    torch and cupy launch kernels asynchronously, so an unsynchronized timer measures
    only the launch, not the computation. We force a device synchronization after the
    call on GPU. jax uses its own block_until_ready path in `jax_test`.
    """
    if xp == "jax":
        return jax_test
    if xp == "torch" and device == "gpu":
        return lambda: (test(), torch.cuda.synchronize())
    if xp == "cupy":
        return lambda: (test(), cupy.cuda.Device().synchronize())
    return test


def time_case(setup, test, repeat, number):
    """Time `test`, returning one seconds-per-call sample per repeat.

    `setup` runs once up front and then once per repeat, always outside the timed
    region. Cases rebind their input state from `setup` via `nonlocal`, so every
    repeat measures freshly built data.

    Args:
        setup: Callable building the inputs and any jitted wrapper.
        test: The timed callable, already wrapped for device synchronization.
        repeat: Number of samples to collect.
        number: Calls per sample. Timings are divided by it.

    Returns:
        Array of `repeat` per-call times in seconds.

    Raises:
        CompileTimeout: Setup, which holds the jit warmup, blew the compile budget.
        CallTimeout: One call would push the sweep of this size over TIMEOUT.
    """
    start = timeit.default_timer()
    setup()
    elapsed = timeit.default_timer() - start
    if elapsed > COMPILE_TIMEOUT:
        raise CompileTimeout(f"{elapsed:.0f}s")
    start = timeit.default_timer()
    test()
    elapsed = timeit.default_timer() - start
    if elapsed > TIMEOUT / (repeat * number):
        raise CallTimeout(f"{elapsed:.2f}s")
    timer = timeit.Timer(stmt=test, setup=setup)
    return np.array(timer.repeat(repeat=repeat, number=number)) / number


def sample_sizes(low, high, base=10):
    """Sample sizes for a sweep, one per power of `base` from base**low to base**high."""
    sizes = np.logspace(low, high, high - low + 1, base=base).astype(int)
    return np.sort(np.array(sorted(set(sizes.tolist()))))


def result_path(mirror, variant, xp, device, fn):
    """Path of one result file, mirroring the scipy module tree."""
    return ROOT / "results" / mirror / variant / xp / device / f"{fn}.json"


def load_result(mirror, variant, xp, device, fn):
    """Load one result file as {sample size: [seconds per call]}, None if absent."""
    path = result_path(mirror, variant, xp, device, fn)
    return json.loads(path.read_text()) if path.exists() else None


def save_result(mirror, variant, xp, device, fn, n_samples, timings, append=False):
    """Merge one sample size into a result file, keeping the other sizes.

    With `append`, the timings extend what is already stored for that size instead of
    replacing it, so repeated invocations accumulate samples across processes. Process
    to process variation dominates on some backends, and only samples drawn from
    separate processes capture it.
    """
    path = result_path(mirror, variant, xp, device, fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text()) if path.exists() else {}
    key = str(int(n_samples))
    previous = results[key] if append and key in results else []
    results[key] = previous + timings
    path.write_text(json.dumps(results, indent=2))


def peak_rss():
    """Peak resident set size of this process in bytes, counted from its start."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def peak_device_bytes(xp, device):
    """Peak device memory in bytes, or None where the framework does not report one.

    Read from each framework's own allocator, since all three keep a pool that hides
    the real usage from nvidia-smi. jax goes further and preallocates most of the card,
    so only its in-use counter says anything.
    """
    if device == "cpu":
        return None
    if xp == "jax":
        return jax.devices("gpu")[0].memory_stats()["peak_bytes_in_use"]
    if xp == "torch":
        return torch.cuda.max_memory_allocated()
    if xp == "cupy":
        return cupy.get_default_memory_pool().total_bytes()
    return None


def save_memory(mirror, variant, xp, device, fn, n_samples, rss, device_bytes):
    """Merge one sample size into the memory file, keeping the other sizes."""
    path = result_path(mirror, variant, xp, device, fn).with_suffix(".mem.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = json.loads(path.read_text()) if path.exists() else {}
    usage[str(int(n_samples))] = {"rss": rss, "device": device_bytes}
    path.write_text(json.dumps(usage, indent=2))


def save_skip(mirror, variant, xp, device, fn, n_samples, reason):
    """Merge one sample size into the skip file, keeping the reasons of the others."""
    path = result_path(mirror, variant, xp, device, fn).with_suffix(".skip.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    skips = json.loads(path.read_text()) if path.exists() else {}
    skips[str(int(n_samples))] = reason
    path.write_text(json.dumps(skips, indent=2))


def _skip_reason(exc):
    """Why a case cannot run on this backend, or None if the error is a real bug."""
    if isinstance(exc, CompileTimeout):
        return f"compile timeout ({exc})"
    if isinstance(exc, CallTimeout):
        return f"call timeout ({exc})"
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return "out of memory"
    if isinstance(exc, AttributeError | ValueError | TypeError):
        # A function that is not array API converted yet raises here, for instance by
        # calling np.asarray on a traced jax array.
        return f"unsupported ({str(exc).splitlines()[0]})"
    return None


def _child(mirror, fn, xp, device, n_samples, repeat, number, float64, conn):
    """Time one case and report back over `conn`, as the entry point of a child process.

    Sends ("result", {timings, rss, device}), ("skip", reason) for an error the sweep
    can carry on past, or ("fatal", traceback) for one it cannot. The process is fresh,
    so its peak memory is the peak of this measurement, minus what importing costs.
    """
    from scipy_bench import registry

    if float64:
        enable_float64()
    try:
        builder = registry.discover()[mirror][fn]
        # Frameworks init lazily, jax only reaching for its compiler on the first jit.
        # Everything they set up for themselves belongs in the baseline, not the case.
        warm = to_xp(xp, np.zeros(1), device)
        if xp == "jax":
            jax.block_until_ready(jax.jit(lambda a: -a)(warm))
        baseline = peak_rss()
        setup, test, jax_test = builder(xp, device, n_samples)
        timed = timed_call(xp, device, test, jax_test)
        timings = time_case(setup, timed, repeat, number)
        conn.send(("result", {
            "timings": timings.tolist(),
            "rss": peak_rss() - baseline,
            "device": peak_device_bytes(xp, device),
        }))
    except Exception as exc:
        reason = _skip_reason(exc)
        conn.send(("skip", reason) if reason else ("fatal", traceback.format_exc()))
    conn.close()


def _measure(mirror, fn, xp, device, n_samples, repeat, number, float64):
    """Time one case in a process we can kill, so a stuck compile cannot hang the sweep.

    XLA compiles inside a C++ call that defers Python signals until it returns, so no
    in-process timeout can interrupt one. Isolation also keeps a segfault or an
    out-of-memory kill from taking the whole sweep down with it.

    Returns:
        The (result, reason) pair, of which exactly one is None. The result holds the
        timings and the peak host and device memory of the measurement.
    """
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    args = (mirror, fn, xp, device, n_samples, repeat, number, float64, sender)
    process = ctx.Process(target=_child, args=args)
    process.start()
    sender.close()  # the child holds the only writing end, so recv sees its death
    budget = COMPILE_TIMEOUT + TIMEOUT
    if not receiver.poll(budget):
        process.kill()
        process.join()
        return None, f"hard timeout ({budget}s)"
    try:
        kind, payload = receiver.recv()
    except EOFError:
        process.join()
        return None, f"crashed (exit {process.exitcode})"
    process.join()
    if kind == "fatal":
        raise RuntimeError(f"{fn} on {xp} {device} n={n_samples}\n{payload}")
    return (payload, None) if kind == "result" else (None, payload)


def _blocked(fn, n_samples, failures, dominated):
    """Why this size need not be attempted, walking the chain of easier cases."""
    case = fn
    while case is not None:
        failed = failures.get(case)
        if failed is not None and n_samples >= failed[0]:
            return f"skipped after {case} n={failed[0]}: {failed[1]}"
        case = dominated.get(case)
    return None


def _depth(fn, dominated):
    """How many cases `fn` dominates, so that easier ones are swept first."""
    depth, case = 0, dominated.get(fn)
    while case is not None:
        depth, case = depth + 1, dominated.get(case)
    return depth


def sweep(
    mirror,
    fns,
    frameworks,
    devices,
    low,
    high,
    repeat,
    number,
    variant,
    append=False,
    base=10,
    float64=False,
    dominated=None,
):
    """Run cases across frameworks, devices and sample sizes, saving as we go.

    Every size that produces no timings records why in a <fn>.skip.json beside them. A
    failure stands for every size above it, and for the cases the suite declares harder
    still, none of which are attempted.

    Args:
        mirror: Scipy module path of the suite, e.g. "spatial/distance".
        fns: Case names to run. Each measurement rediscovers the builder in its own
            process, so the sweep only needs the names.
        frameworks: Framework names to run.
        devices: Device names to run.
        low: Exponent of the smallest sample size.
        high: Exponent of the largest sample size.
        repeat: Samples per sample size.
        number: Calls per sample.
        variant: Result tree to write into, "current" or "baseline".
        append: Add to the stored samples instead of replacing them, so repeated
            invocations accumulate across processes.
        base: Base the size exponents are taken to, 10 for decades, 2 for octaves.
        float64: Ask the child processes to keep jax in float64.
        dominated: {case: the case it is harder than}, from the registry. The chains it
            forms decide which cases a failure carries over to.
    """
    dominated = dominated or {}
    fns = sorted(fns, key=lambda fn: _depth(fn, dominated))
    for xp in frameworks:
        for device in devices:
            if (xp, device) in SKIP_XP_DEVICES:
                continue
            failures = {}
            for fn in fns:
                for n_samples in sample_sizes(low, high, base):
                    print(f"{mirror}/{fn}: {xp} {device} n={n_samples}")
                    blocked = _blocked(fn, n_samples, failures, dominated)
                    if blocked is not None:
                        save_skip(mirror, variant, xp, device, fn, n_samples, blocked)
                        continue
                    result, reason = _measure(
                        mirror, fn, xp, device, int(n_samples), repeat, number, float64
                    )
                    if reason is not None:
                        print(f"  SKIP {fn} on {xp} {device} - {reason}")
                        save_skip(mirror, variant, xp, device, fn, n_samples, reason)
                        failures[fn] = (n_samples, reason)
                        continue
                    save_result(
                        mirror,
                        variant,
                        xp,
                        device,
                        fn,
                        n_samples,
                        result["timings"],
                        append,
                    )
                    save_memory(
                        mirror,
                        variant,
                        xp,
                        device,
                        fn,
                        n_samples,
                        result["rss"],
                        result["device"],
                    )
