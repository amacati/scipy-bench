"""Array API benchmark framework for scipy.

Suites live in scipy_bench/suites and mirror the scipy module tree. Each registers cases
that the runner times across numpy, torch, jax and cupy on cpu and gpu.
"""

from scipy_bench.core import (
    DEVICES,
    FRAMEWORKS,
    SKIP_XP_DEVICES,
    device_of,
    load_result,
    result_path,
    sample_sizes,
    sweep,
    time_case,
    timed_call,
    to_numpy,
    to_xp,
)
from scipy_bench.registry import discover, register

__all__ = [
    "DEVICES",
    "FRAMEWORKS",
    "SKIP_XP_DEVICES",
    "device_of",
    "discover",
    "load_result",
    "register",
    "result_path",
    "sample_sizes",
    "sweep",
    "time_case",
    "timed_call",
    "to_numpy",
    "to_xp",
]
