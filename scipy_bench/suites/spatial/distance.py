"""Benchmarks for scipy.spatial.distance."""

from functools import partial

import jax
import numpy as np
from scipy.spatial import distance

from scipy_bench import check_float64, device_of, register, to_xp

# Features per row in pdist, cdist and the batched metrics. Each diagnostic dimension
# is its own pdist and cdist case.
N_FEATURES = 3
DIAGNOSTIC_DIMS = [4, 16, 64, 256, 1024]

# Metrics with optional weights, as (callable, "float" or "bool" inputs).
WEIGHTED_METRICS = {
    "euclidean": (distance.euclidean, "float"),
    "sqeuclidean": (distance.sqeuclidean, "float"),
    "cityblock": (distance.cityblock, "float"),
    "chebyshev": (distance.chebyshev, "float"),
    "minkowski": (partial(distance.minkowski, p=3), "float"),
    "cosine": (distance.cosine, "float"),
    "correlation": (distance.correlation, "float"),
    "braycurtis": (distance.braycurtis, "float"),
    "canberra": (distance.canberra, "float"),
    "hamming": (distance.hamming, "float"),
    "jaccard": (distance.jaccard, "bool"),
    "yule": (distance.yule, "bool"),
    "dice": (distance.dice, "bool"),
    "rogerstanimoto": (distance.rogerstanimoto, "bool"),
    "russellrao": (distance.russellrao, "bool"),
    "sokalsneath": (distance.sokalsneath, "bool"),
}

# Functions of a square distance matrix ("dm") or a condensed vector ("y")
MATRIX_SPECS = {
    "is_valid_dm": (distance.is_valid_dm, "dm"),
    "num_obs_dm": (distance.num_obs_dm, "dm"),
    "is_valid_y": (distance.is_valid_y, "y"),
    "num_obs_y": (distance.num_obs_y, "y"),
    "squareform": (distance.squareform, "y"),
}


def create_array(xp, device, shape, kind="float"):
    """Create a random array.

    Args:
        xp: Framework name.
        device: "cpu" or "gpu".
        shape: Array shape.
        kind: "float" or "bool".

    Returns:
        The array on `device`. Boolean rows hold a True and, from length 2 on, a
        False, so every boolean metric stays defined.
    """
    if kind == "bool":
        arr = np.random.rand(*shape) > 0.5
        arr[..., 0] = True
        if shape[-1] > 1:
            arr[..., 1] = False
    else:
        arr = np.random.rand(*shape)
    return to_xp(xp, arr, device)


def create_weights(xp, device, n_features):
    """Create positive weights or variances, one per feature."""
    return create_array(xp, device, (n_features,))


def create_inverse_covariance(xp, device, n_features):
    """Create a random symmetric positive definite inverse covariance."""
    a = np.random.rand(n_features, n_features)
    # Diagonally dominant, so positive definite without an O(n^3) Gram matrix
    return to_xp(xp, (a + a.T) / 2 + n_features * np.eye(n_features), device)


def create_matrix(xp, device, m, n_features):
    """Observation matrix of m observations with n_features features each."""
    return to_xp(xp, np.random.rand(m, n_features), device)


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


def metric(call, kind, extra, batched, xp, device, n_samples):
    """Build a case timing a metric, jitted on jax.

    Args:
        call: The metric.
        kind: Element kind of u and v, "float" or "bool".
        extra: Keyword and builder of a third argument, or None.
        batched: Whether u and v hold n_samples rows of N_FEATURES.
    """
    shape = (n_samples, N_FEATURES) if batched else (n_samples,)
    u, v, kwargs = None, None, None
    jfn = jax.jit(call) if xp == "jax" else None

    def setup():
        nonlocal u, v, kwargs
        u = create_array(xp, device, shape, kind)
        v = create_array(xp, device, shape, kind)
        kwargs = {} if extra is None else {extra[0]: extra[1](xp, device, shape[-1])}
        assert device_of(u) == device, f"setup on {device_of(u)}, want {device}"
        if xp == "jax":
            check_float64(jax.block_until_ready(jfn(u, v, **kwargs)))

    def test():
        return call(u, v, **kwargs)

    def jax_test():
        jax.block_until_ready(jfn(u, v, **kwargs))

    return setup, test, jax_test


# Weights only add work, so a failing plain metric rules out its weighted case
for _name, (_call, _kind) in WEIGHTED_METRICS.items():
    register(_name, partial(metric, _call, _kind, None, False))
    register(
        f"{_name}_w",
        partial(metric, _call, _kind, ("w", create_weights), False),
        harder_than=_name,
    )
    register(f"{_name}_batch", partial(metric, _call, _kind, None, True))

# Metrics without weights. jensenshannon defaults to axis 0, the others reduce axis -1
for _name, (_call, _extra) in {
    "jensenshannon": (partial(distance.jensenshannon, axis=-1), None),
    "seuclidean": (distance.seuclidean, ("V", create_weights)),
    "mahalanobis": (distance.mahalanobis, ("VI", create_inverse_covariance)),
}.items():
    register(_name, partial(metric, _call, "float", _extra, False))
    register(f"{_name}_batch", partial(metric, _call, "float", _extra, True))


def matrix(call, kind, xp, device, n_samples):
    """Build a case timing a function of a distance matrix, jitted on jax."""
    create = create_distance_matrix if kind == "dm" else create_condensed
    data = None
    jfn = jax.jit(call) if xp == "jax" else None

    def setup():
        nonlocal data
        data = create(xp, device, n_samples)
        assert device_of(data) == device, f"setup on {device_of(data)}, want {device}"
        if xp == "jax":
            jax.block_until_ready(jfn(data))

    def test():
        return call(data)

    def jax_test():
        jax.block_until_ready(jfn(data))

    return setup, test, jax_test


for _name, (_call, _kind) in MATRIX_SPECS.items():
    register(_name, partial(matrix, _call, _kind))


def pdist(n_features, xp, device, n_samples):
    X = None
    # Built once per case, not per repeat: a fresh jax.jit brings an empty compilation
    # cache with it, and XLA autotunes every shape it has not seen
    jfn = jax.jit(distance.pdist) if xp == "jax" else None

    def setup():
        nonlocal X
        X = create_matrix(xp, device, n_samples, n_features)
        assert device_of(X) == device, f"setup on {device_of(X)}, want {device}"
        if xp == "jax":
            check_float64(jax.block_until_ready(jfn(X)))

    def test():
        return distance.pdist(X)

    def jax_test():
        jax.block_until_ready(jfn(X))

    return setup, test, jax_test


def cdist(n_features, xp, device, n_samples):
    XA, XB = None, None
    jfn = jax.jit(distance.cdist) if xp == "jax" else None

    def setup():
        nonlocal XA, XB
        XA = create_matrix(xp, device, n_samples, n_features)
        XB = create_matrix(xp, device, n_samples, n_features)
        assert device_of(XA) == device, f"setup on {device_of(XA)}, want {device}"
        if xp == "jax":
            check_float64(jax.block_until_ready(jfn(XA, XB)))

    def test():
        return distance.cdist(XA, XB)

    def jax_test():
        jax.block_until_ready(jfn(XA, XB))

    return setup, test, jax_test


register("pdist", partial(pdist, N_FEATURES))
register("cdist", partial(cdist, N_FEATURES))

# A wider observation costs more at every count, so a dimension that fails rules out
# the ones above it. pdist and cdist are not comparable this way: cdist does twice the
# pair work, but pdist gathers its pairs into an array cdist never builds.
for _easier, _n_features in zip([None, *DIAGNOSTIC_DIMS], DIAGNOSTIC_DIMS):
    for _fn, _case in [(pdist, "pdist"), (cdist, "cdist")]:
        register(
            f"{_case}_d{_n_features}",
            partial(_fn, _n_features),
            harder_than=f"{_case}_d{_easier}" if _easier else None,
        )
