"""Benchmarks for scipy.spatial.distance."""

from functools import partial

import jax
import numpy as np
from scipy.spatial import distance

from scipy_bench import check_float64, device_of, register, to_xp

# Feature dimension of the observation matrices of pdist and cdist. The main cases use
# N_FEATURES, the diagnostic dimensions are registered as separate cases so each one is
# swept over the full range of observation counts.
N_FEATURES = 3
DIAGNOSTIC_DIMS = [4, 16, 64, 256, 1024]

# Pair metrics take two 1-D vectors and return a scalar. Scaling the vector length
# measures the elementwise work an array API backend parallelizes. Each entry is
# (callable, dtype, jittable).
PAIR_SPECS = {
    "euclidean": (distance.euclidean, "float64", True),
    "sqeuclidean": (distance.sqeuclidean, "float64", True),
    "cityblock": (distance.cityblock, "float64", True),
    "chebyshev": (distance.chebyshev, "float64", True),
    "minkowski": (partial(distance.minkowski, p=3), "float64", True),
    "cosine": (distance.cosine, "float64", True),
    "correlation": (distance.correlation, "float64", True),
    "braycurtis": (distance.braycurtis, "float64", True),
    "canberra": (distance.canberra, "float64", True),
    "hamming": (distance.hamming, "float64", True),
    "jensenshannon": (distance.jensenshannon, "float64", True),
    "jaccard": (distance.jaccard, "bool", True),
    "yule": (distance.yule, "bool", True),
    "dice": (distance.dice, "bool", True),
    "rogerstanimoto": (distance.rogerstanimoto, "bool", True),
    "russellrao": (distance.russellrao, "bool", True),
    "sokalsneath": (
        distance.sokalsneath,
        "bool",
        False,
    ),  # data-dependent guard, no jit
}

# Validators take either a square distance matrix ("dm") or a condensed vector ("y").
VALIDATOR_SPECS = {
    "is_valid_dm": (distance.is_valid_dm, "dm"),
    "num_obs_dm": (distance.num_obs_dm, "dm"),
    "is_valid_y": (distance.is_valid_y, "y"),
    "num_obs_y": (distance.num_obs_y, "y"),
}


def create_vector(xp, device, n, dtype="float64"):
    """Random vector of length n.

    Args:
        xp: Framework name.
        device: "cpu" or "gpu".
        n: Vector length.
        dtype: "bool" for the boolean metrics, any float dtype otherwise.

    Returns:
        The vector on `device`. Boolean vectors always contain a True and, from n = 2
        on, also a False, so sokalsneath, yule, dice and russellrao stay defined.
    """
    if dtype == "bool":
        arr = np.random.rand(n) > 0.5
        arr[0] = True
        if n > 1:
            arr[1] = False
    else:
        arr = np.random.rand(n).astype(dtype)
    return to_xp(xp, arr, device)


def create_matrix(xp, device, m, n_features):
    """Observation matrix of m observations with n_features features each."""
    X = to_xp(xp, np.random.rand(m, n_features), device)
    check_float64(X)
    return X


def create_distance_matrix(xp, device, k):
    """Symmetric k by k matrix with an exact zero diagonal.

    Both properties are required for is_valid_dm to run all of its checks instead of
    short-circuiting on the first failing one.
    """
    a = np.random.rand(k, k)
    d = a + a.T
    np.fill_diagonal(d, 0.0)
    return to_xp(xp, d, device)


def create_condensed(xp, device, n):
    """Condensed distance vector whose length is the valid binomial size nearest n."""
    m = int((1 + (1 + 8 * n) ** 0.5) // 2)
    return to_xp(xp, np.random.rand(max(m * (m - 1) // 2, 1)), device)


def _pair_case(call, dtype, jittable):
    """Case builder for a two-vector scalar metric, scaling the vector length."""

    def build(xp, device, n_samples):
        u, v, jfn = None, None, None

        def setup():
            nonlocal u, v, jfn
            u = create_vector(xp, device, n_samples, dtype)
            v = create_vector(xp, device, n_samples, dtype)
            assert device_of(u) == device, f"setup on {device_of(u)}, want {device}"
            if xp == "jax" and jittable:
                jfn = jax.jit(call)
                jax.block_until_ready(jfn(u, v))

        def test():
            return call(u, v)

        def jax_test():
            jax.block_until_ready((jfn if jittable else call)(u, v))

        return setup, test, jax_test

    return build


def _validator_case(call, kind):
    """Case builder for a validator, fed a distance matrix or a condensed vector."""
    create = create_distance_matrix if kind == "dm" else create_condensed

    def build(xp, device, n_samples):
        data = None

        def setup():
            nonlocal data
            data = create(xp, device, n_samples)
            assert device_of(data) == device, (
                f"setup on {device_of(data)}, not {device}"
            )

        def test():
            return call(data)

        def jax_test():
            jax.block_until_ready(call(data))

        return setup, test, jax_test

    return build


for _name, (_call, _dtype, _jittable) in PAIR_SPECS.items():
    register(_name, _pair_case(_call, _dtype, _jittable))

for _name, (_call, _kind) in VALIDATOR_SPECS.items():
    register(_name, _validator_case(_call, _kind))


@register
def mahalanobis(xp, device, n_samples):
    u, v, VI = None, None, None

    def setup():
        nonlocal u, v, VI
        u = create_vector(xp, device, n_samples)
        v = create_vector(xp, device, n_samples)
        VI = to_xp(xp, np.random.rand(n_samples, n_samples), device)
        assert device_of(u) == device, f"setup on {device_of(u)}, want {device}"

    def test():
        return distance.mahalanobis(u, v, VI)

    def jax_test():
        jax.block_until_ready(distance.mahalanobis(u, v, VI))

    return setup, test, jax_test


@register
def seuclidean(xp, device, n_samples):
    u, v, V = None, None, None

    def setup():
        nonlocal u, v, V
        u = create_vector(xp, device, n_samples)
        v = create_vector(xp, device, n_samples)
        V = create_vector(xp, device, n_samples)  # positive variances in [0, 1)
        assert device_of(u) == device, f"setup on {device_of(u)}, want {device}"

    def test():
        return distance.seuclidean(u, v, V)

    def jax_test():
        jax.block_until_ready(distance.seuclidean(u, v, V))

    return setup, test, jax_test


def pdist(n_features, xp, device, n_samples):
    X, jfn = None, None

    def setup():
        nonlocal X, jfn
        X = create_matrix(xp, device, n_samples, n_features)
        assert device_of(X) == device, f"setup on {device_of(X)}, want {device}"
        if xp == "jax":
            jfn = jax.jit(distance.pdist)
            check_float64(jax.block_until_ready(jfn(X)))

    def test():
        return distance.pdist(X)

    def jax_test():
        jax.block_until_ready(jfn(X))

    return setup, test, jax_test


def cdist(n_features, xp, device, n_samples):
    XA, XB, jfn = None, None, None

    def setup():
        nonlocal XA, XB, jfn
        XA = create_matrix(xp, device, n_samples, n_features)
        XB = create_matrix(xp, device, n_samples, n_features)
        assert device_of(XA) == device, f"setup on {device_of(XA)}, want {device}"
        if xp == "jax":
            jfn = jax.jit(distance.cdist)
            check_float64(jax.block_until_ready(jfn(XA, XB)))

    def test():
        return distance.cdist(XA, XB)

    def jax_test():
        jax.block_until_ready(jfn(XA, XB))

    return setup, test, jax_test


register("pdist", partial(pdist, N_FEATURES))
register("cdist", partial(cdist, N_FEATURES))

for _n_features in DIAGNOSTIC_DIMS:
    register(f"pdist_d{_n_features}", partial(pdist, _n_features))
    register(f"cdist_d{_n_features}", partial(cdist, _n_features))


@register
def squareform(xp, device, n_samples):
    y = None

    def setup():
        nonlocal y
        y = create_condensed(xp, device, n_samples)
        assert device_of(y) == device, f"setup on {device_of(y)}, want {device}"

    def test():
        return distance.squareform(y)

    def jax_test():
        jax.block_until_ready(distance.squareform(y))

    return setup, test, jax_test
