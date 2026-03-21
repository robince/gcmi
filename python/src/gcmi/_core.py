"""Gaussian-copula mutual information estimation.

The estimators in this module follow the historical MATLAB implementation in
this repository, but are packaged for modern Python use.
"""

from __future__ import annotations

import math
import warnings
from numbers import Integral
from statistics import NormalDist

import numpy as np

__version__ = "0.4.0"

__all__ = [
    "ctransform",
    "copnorm",
    "ent_g",
    "mi_gg",
    "gcmi_cc",
    "mi_model_gd",
    "gcmi_model_cd",
    "mi_mixture_gd",
    "gcmi_mixture_cd",
    "cmi_ggg",
    "gccmi_ccc",
    "gccmi_ccd",
    "__version__",
]

_NORMAL = NormalDist()
_NORM_PPF = np.vectorize(_NORMAL.inv_cdf, otypes=[float])


def _as_continuous_2d(x: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim > 2:
        raise ValueError(f"{name} must be at most 2d")
    return np.atleast_2d(arr)


def _as_discrete_1d(x: np.ndarray | list[int] | tuple[int, ...], name: str) -> np.ndarray:
    arr = np.squeeze(np.asarray(x))
    if arr.ndim > 1:
        raise ValueError(f"only univariate discrete variables supported for {name}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} should be an integer array")
    return np.atleast_1d(arr.astype(np.int64, copy=False))


def _require_integral(value: Integral, name: str) -> int:
    if not isinstance(value, Integral):
        raise ValueError(f"{name} should be an integer")
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} should be positive")
    return ivalue


def _require_sample_capacity(n_samples: int, n_vars: int, context: str) -> None:
    if n_samples <= n_vars:
        raise ValueError(
            f"{context} requires more samples than variables "
            f"({n_samples} samples, {n_vars} variables)"
        )


def _logdet_from_cholesky(chol: np.ndarray) -> float:
    return float(np.log(np.diagonal(chol)).sum())


def _digamma_scalar(x: float) -> float:
    if x <= 0.0:
        if float(x).is_integer():
            raise ValueError("digamma is undefined at non-positive integers")
        return _digamma_scalar(1.0 - x) - math.pi / math.tan(math.pi * x)

    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0

    inv = 1.0 / x
    inv2 = inv * inv
    result += math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0
        - inv2 * (
            1.0 / 120.0
            - inv2 * (
                1.0 / 252.0
                - inv2 * (
                    1.0 / 240.0
                    - inv2 * (5.0 / 660.0)
                )
            )
        )
    )
    return result


def _digamma(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return float(_digamma_scalar(float(arr)))
    vec = np.vectorize(_digamma_scalar, otypes=[float])
    return vec(arr)


def _bias_correction(n_samples: int, n_vars: int) -> float:
    dims = np.arange(1, n_vars + 1, dtype=float)
    psi_terms = _digamma((n_samples - dims) / 2.0) / 2.0
    dterm = (math.log(2.0) - math.log(n_samples - 1.0)) / 2.0
    return float(n_vars * dterm + psi_terms.sum())


def _bias_sequence(n_samples: int, n_vars: int) -> tuple[float, np.ndarray]:
    dims = np.arange(1, n_vars + 1, dtype=float)
    psi_terms = _digamma((n_samples - dims) / 2.0) / 2.0
    dterm = (math.log(2.0) - math.log(n_samples - 1.0)) / 2.0
    return float(dterm), np.cumsum(np.asarray(psi_terms, dtype=float))


def _class_counts(y: np.ndarray, n_classes: int, name: str) -> np.ndarray:
    if y.size == 0:
        raise ValueError(f"{name} is empty")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"{name} is out of bounds")
    counts = np.bincount(y, minlength=n_classes)
    if counts.size != n_classes:
        raise ValueError(f"{name} is out of bounds")
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise ValueError(f"{name} contains empty classes: {empty.tolist()}")
    return counts.astype(float)


def _warn_repeated_values(x: np.ndarray, name: str) -> None:
    n_samples = x.shape[1]
    for idx in range(x.shape[0]):
        if np.unique(x[idx, :]).size / float(n_samples) < 0.9:
            warnings.warn(f"Input {name} has more than 10% repeated values", stacklevel=2)
            break


def _gaussian_covariance(x: np.ndarray, demeaned: bool) -> np.ndarray:
    if not demeaned:
        x = x - x.mean(axis=1, keepdims=True)
    return (x @ x.T) / float(x.shape[1] - 1)


def _gaussian_entropy_from_samples(x: np.ndarray, biascorrect: bool, demeaned: bool) -> float:
    n_vars, n_samples = x.shape
    _require_sample_capacity(n_samples, n_vars, "Gaussian entropy")
    cov = _gaussian_covariance(x, demeaned)
    chol = np.linalg.cholesky(cov)
    entropy_nats = _logdet_from_cholesky(chol) + 0.5 * n_vars * (math.log(2.0 * math.pi) + 1.0)
    if biascorrect:
        entropy_nats -= _bias_correction(n_samples, n_vars)
    return entropy_nats


def ctransform(x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Empirical CDF transform along the last axis."""

    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.array(0.5, dtype=float)
    order = np.argsort(arr, axis=-1, kind="mergesort")
    ranks = np.argsort(order, axis=-1, kind="mergesort")
    return (ranks.astype(float) + 1.0) / (arr.shape[-1] + 1.0)


def copnorm(x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Copula-normalize values along the last axis."""

    return _NORM_PPF(ctransform(x))


def ent_g(x: np.ndarray | list[float] | tuple[float, ...], biascorrect: bool = True) -> float:
    """Entropy of a Gaussian variable in bits."""

    arr = _as_continuous_2d(x, "x")
    entropy_nats = _gaussian_entropy_from_samples(arr, biascorrect=biascorrect, demeaned=False)
    return entropy_nats / math.log(2.0)


def mi_gg(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[float] | tuple[float, ...],
    biascorrect: bool = True,
    demeaned: bool = False,
) -> float:
    """Mutual information between two Gaussian variables in bits."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_continuous_2d(y, "y")
    if y_arr.shape[1] != x_arr.shape[1]:
        raise ValueError("number of trials do not match")

    n_samples = x_arr.shape[1]
    n_x = x_arr.shape[0]
    n_y = y_arr.shape[0]
    n_xy = n_x + n_y
    _require_sample_capacity(n_samples, n_xy, "Gaussian mutual information")

    xy = np.vstack((x_arr, y_arr))
    cov_xy = _gaussian_covariance(xy, demeaned=demeaned)
    cov_x = cov_xy[:n_x, :n_x]
    cov_y = cov_xy[n_x:, n_x:]

    chol_xy = np.linalg.cholesky(cov_xy)
    chol_x = np.linalg.cholesky(cov_x)
    chol_y = np.linalg.cholesky(cov_y)

    hx = _logdet_from_cholesky(chol_x)
    hy = _logdet_from_cholesky(chol_y)
    hxy = _logdet_from_cholesky(chol_xy)

    if biascorrect:
        dterm, cumpsi = _bias_sequence(n_samples, n_xy)
        hx -= n_x * dterm + cumpsi[n_x - 1]
        hy -= n_y * dterm + cumpsi[n_y - 1]
        hxy -= n_xy * dterm + cumpsi[n_xy - 1]

    return (hx + hy - hxy) / math.log(2.0)


def gcmi_cc(x: np.ndarray | list[float] | tuple[float, ...], y: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Gaussian-copula MI between two continuous variables."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_continuous_2d(y, "y")
    if y_arr.shape[1] != x_arr.shape[1]:
        raise ValueError("number of trials do not match")

    _warn_repeated_values(x_arr, "x")
    _warn_repeated_values(y_arr, "y")
    return mi_gg(copnorm(x_arr), copnorm(y_arr), biascorrect=True, demeaned=True)


def mi_model_gd(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[int] | tuple[int, ...],
    Ym: Integral,
    biascorrect: bool = True,
    demeaned: bool = False,
) -> float:
    """Mutual information between a Gaussian and a discrete variable in bits."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_discrete_1d(y, "y")
    Ym_int = _require_integral(Ym, "Ym")
    if y_arr.size != x_arr.shape[1]:
        raise ValueError("number of trials do not match")

    counts = _class_counts(y_arr, Ym_int, "y")
    n_vars = x_arr.shape[0]
    _require_sample_capacity(x_arr.shape[1], n_vars, "Gaussian/discrete mutual information")
    if np.any(counts <= n_vars):
        missing = np.flatnonzero(counts <= n_vars)
        raise ValueError(
            f"each class needs more than {n_vars} samples for covariance estimation; "
            f"problem classes: {missing.tolist()}"
        )

    if not demeaned:
        x_arr = x_arr - x_arr.mean(axis=1, keepdims=True)

    hcond = np.zeros(Ym_int, dtype=float)
    for yi in range(Ym_int):
        xm = x_arr[:, y_arr == yi]
        xm = xm - xm.mean(axis=1, keepdims=True)
        cov = (xm @ xm.T) / float(xm.shape[1] - 1)
        chol = np.linalg.cholesky(cov)
        hcond[yi] = _logdet_from_cholesky(chol)

    cov_x = (x_arr @ x_arr.T) / float(x_arr.shape[1] - 1)
    chol_x = np.linalg.cholesky(cov_x)
    hunc = _logdet_from_cholesky(chol_x)

    if biascorrect:
        hunc -= _bias_correction(x_arr.shape[1], n_vars)
        for yi in range(Ym_int):
            hcond[yi] -= _bias_correction(int(counts[yi]), n_vars)

    weights = counts / float(x_arr.shape[1])
    return (hunc - float(np.sum(weights * hcond))) / math.log(2.0)


def gcmi_model_cd(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[int] | tuple[int, ...],
    Ym: Integral,
) -> float:
    """Gaussian-copula MI between a continuous and a discrete variable."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_discrete_1d(y, "y")
    Ym_int = _require_integral(Ym, "Ym")
    if y_arr.size != x_arr.shape[1]:
        raise ValueError("number of trials do not match")
    if y_arr.min() != 0 or y_arr.max() != Ym_int - 1:
        raise ValueError("values of discrete variable y are out of bounds")

    _warn_repeated_values(x_arr, "x")
    return mi_model_gd(copnorm(x_arr), y_arr, Ym_int, biascorrect=True, demeaned=True)


def mi_mixture_gd(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[int] | tuple[int, ...],
    Ym: Integral,
) -> float:
    """Mutual information between a Gaussian and a discrete variable via a Gaussian mixture."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_discrete_1d(y, "y")
    Ym_int = _require_integral(Ym, "Ym")
    if y_arr.size != x_arr.shape[1]:
        raise ValueError("number of trials do not match")

    counts = _class_counts(y_arr, Ym_int, "y")
    n_vars = x_arr.shape[0]
    _require_sample_capacity(x_arr.shape[1], n_vars, "Gaussian/discrete mutual information")
    if np.any(counts <= n_vars):
        missing = np.flatnonzero(counts <= n_vars)
        raise ValueError(
            f"each class needs more than {n_vars} samples for covariance estimation; "
            f"problem classes: {missing.tolist()}"
        )

    class_means = np.zeros((Ym_int, n_vars), dtype=float)
    class_chol = np.zeros((Ym_int, n_vars, n_vars), dtype=float)
    hcond = np.zeros(Ym_int, dtype=float)

    for yi in range(Ym_int):
        xm = x_arr[:, y_arr == yi]
        mean = xm.mean(axis=1)
        class_means[yi, :] = mean
        xm = xm - mean[:, np.newaxis]
        cov = (xm @ xm.T) / float(xm.shape[1] - 1)
        chol = np.linalg.cholesky(cov)
        class_chol[yi, :, :] = chol
        hcond[yi] = _logdet_from_cholesky(chol) + 0.5 * n_vars * (math.log(2.0 * math.pi) + 1.0)

    weights = counts / float(x_arr.shape[1])
    D = n_vars
    scale = math.sqrt(n_vars)
    hmix = 0.0

    for yi in range(Ym_int):
        sigma = scale * class_chol[yi].T
        sigma_points = np.hstack(
            [class_means[yi][:, np.newaxis] + sigma, class_means[yi][:, np.newaxis] - sigma]
        )
        log_lik = np.zeros((Ym_int, 2 * n_vars), dtype=float)
        for mi in range(Ym_int):
            dx = sigma_points - class_means[mi][:, np.newaxis]
            log_lik[mi, :] = _norm_innerv(dx, class_chol[mi]) - hcond[mi] + 0.5 * n_vars
        hmix += weights[yi] * _logsumexp(log_lik, axis=0, weights=weights).sum()

    hmix = -hmix / (2.0 * D)
    return (hmix - float(np.sum(weights * hcond))) / math.log(2.0)


def _norm_innerv(x: np.ndarray, chol: np.ndarray) -> np.ndarray:
    solved = np.linalg.solve(chol, x)
    return -0.5 * np.sum(solved * solved, axis=0)


def _logsumexp(
    a: np.ndarray,
    axis: int = 0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    max_a = np.max(arr, axis=axis, keepdims=True)
    shifted = np.exp(arr - max_a)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if w.shape[0] != arr.shape[axis]:
            raise ValueError("weights do not match the reduction axis")
        shape = [1] * shifted.ndim
        shape[axis] = w.shape[0]
        shifted = shifted * w.reshape(shape)
    summed = np.sum(shifted, axis=axis, keepdims=True)
    return np.squeeze(max_a + np.log(summed), axis=axis)


def gcmi_mixture_cd(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[int] | tuple[int, ...],
    Ym: Integral,
) -> float:
    """Gaussian-copula MI between a continuous and a discrete variable via a Gaussian mixture."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_discrete_1d(y, "y")
    Ym_int = _require_integral(Ym, "Ym")
    if y_arr.size != x_arr.shape[1]:
        raise ValueError("number of trials do not match")
    if y_arr.min() != 0 or y_arr.max() != Ym_int - 1:
        raise ValueError("values of discrete variable y are out of bounds")

    _warn_repeated_values(x_arr, "x")

    transformed = []
    relabeled = []
    for yi in range(Ym_int):
        idx = y_arr == yi
        xm = x_arr[:, idx]
        if xm.shape[1] == 0:
            raise ValueError(f"y contains empty class {yi}")
        cxm = copnorm(xm)
        median = np.median(xm, axis=1, keepdims=True)
        mad = np.median(np.abs(xm - median), axis=1, keepdims=True)
        if np.any(mad == 0):
            raise ValueError(f"class {yi} has zero robust scale")
        transformed.append(cxm * (1.482602218505602 * mad) + median)
        relabeled.append(np.full(xm.shape[1], yi, dtype=np.int64))

    pooled_x = np.concatenate(transformed, axis=1)
    pooled_y = np.concatenate(relabeled)
    return mi_mixture_gd(pooled_x, pooled_y, Ym_int)


def cmi_ggg(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[float] | tuple[float, ...],
    z: np.ndarray | list[float] | tuple[float, ...],
    biascorrect: bool = True,
    demeaned: bool = False,
) -> float:
    """Conditional mutual information between Gaussian variables in bits."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_continuous_2d(y, "y")
    z_arr = _as_continuous_2d(z, "z")
    if y_arr.shape[1] != x_arr.shape[1] or z_arr.shape[1] != x_arr.shape[1]:
        raise ValueError("number of trials do not match")

    n_samples = x_arr.shape[1]
    n_x = x_arr.shape[0]
    n_y = y_arr.shape[0]
    n_z = z_arr.shape[0]
    n_xy = n_x + n_y
    n_xz = n_x + n_z
    n_yz = n_y + n_z
    n_xyz = n_x + n_y + n_z
    _require_sample_capacity(n_samples, n_xyz, "Gaussian conditional mutual information")

    xyz = np.vstack((x_arr, y_arr, z_arr))
    cov_xyz = _gaussian_covariance(xyz, demeaned=demeaned)
    cov_z = cov_xyz[n_xy:, n_xy:]
    cov_yz = cov_xyz[n_x:, n_x:]
    cov_xz = np.zeros((n_xz, n_xz), dtype=float)
    cov_xz[:n_x, :n_x] = cov_xyz[:n_x, :n_x]
    cov_xz[:n_x, n_x:] = cov_xyz[:n_x, n_xy:]
    cov_xz[n_x:, :n_x] = cov_xyz[n_xy:, :n_x]
    cov_xz[n_x:, n_x:] = cov_xyz[n_xy:, n_xy:]

    chol_z = np.linalg.cholesky(cov_z)
    chol_xz = np.linalg.cholesky(cov_xz)
    chol_yz = np.linalg.cholesky(cov_yz)
    chol_xyz = np.linalg.cholesky(cov_xyz)

    hz = _logdet_from_cholesky(chol_z)
    hxz = _logdet_from_cholesky(chol_xz)
    hyz = _logdet_from_cholesky(chol_yz)
    hxyz = _logdet_from_cholesky(chol_xyz)

    if biascorrect:
        dterm, cumpsi = _bias_sequence(n_samples, n_xyz)
        hz -= n_z * dterm + cumpsi[n_z - 1]
        hxz -= n_xz * dterm + cumpsi[n_xz - 1]
        hyz -= n_yz * dterm + cumpsi[n_yz - 1]
        hxyz -= n_xyz * dterm + cumpsi[n_xyz - 1]

    return (hxz + hyz - hxyz - hz) / math.log(2.0)


def gccmi_ccc(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[float] | tuple[float, ...],
    z: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Gaussian-copula conditional mutual information between three continuous variables."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_continuous_2d(y, "y")
    z_arr = _as_continuous_2d(z, "z")
    if y_arr.shape[1] != x_arr.shape[1] or z_arr.shape[1] != x_arr.shape[1]:
        raise ValueError("number of trials do not match")

    _warn_repeated_values(x_arr, "x")
    _warn_repeated_values(y_arr, "y")
    _warn_repeated_values(z_arr, "z")
    return cmi_ggg(copnorm(x_arr), copnorm(y_arr), copnorm(z_arr), biascorrect=True, demeaned=True)


def gccmi_ccd(
    x: np.ndarray | list[float] | tuple[float, ...],
    y: np.ndarray | list[float] | tuple[float, ...],
    z: np.ndarray | list[int] | tuple[int, ...],
    Zm: Integral,
) -> tuple[float, float]:
    """Gaussian-copula CMI between two continuous variables conditioned on a discrete variable."""

    x_arr = _as_continuous_2d(x, "x")
    y_arr = _as_continuous_2d(y, "y")
    z_arr = _as_discrete_1d(z, "z")
    Zm_int = _require_integral(Zm, "Zm")
    if y_arr.shape[1] != x_arr.shape[1] or z_arr.size != x_arr.shape[1]:
        raise ValueError("number of trials do not match")
    if z_arr.min() != 0 or z_arr.max() != Zm_int - 1:
        raise ValueError("values of discrete variable z are out of bounds")

    _warn_repeated_values(x_arr, "x")
    _warn_repeated_values(y_arr, "y")

    class_mi = np.zeros(Zm_int, dtype=float)
    counts = np.zeros(Zm_int, dtype=float)
    copula_x = []
    copula_y = []
    for zi in range(Zm_int):
        idx = z_arr == zi
        if not np.any(idx):
            raise ValueError(f"z contains empty class {zi}")
        counts[zi] = float(idx.sum())
        thsx = copnorm(x_arr[:, idx])
        thsy = copnorm(y_arr[:, idx])
        copula_x.append(thsx)
        copula_y.append(thsy)
        class_mi[zi] = mi_gg(thsx, thsy, biascorrect=True, demeaned=True)

    weights = counts / float(x_arr.shape[1])
    cmi = float(np.sum(weights * class_mi))
    pooled = mi_gg(np.hstack(copula_x), np.hstack(copula_y), biascorrect=True, demeaned=False)
    return cmi, pooled
