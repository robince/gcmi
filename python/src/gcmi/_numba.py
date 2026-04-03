"""Numba-accelerated batch kernels for GCMI."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

_LN2 = math.log(2.0)


@njit(cache=True, nogil=True)
def _digamma_scalar(x: float) -> float:
    if x <= 0.0:
        if x == math.floor(x):
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


@njit(cache=True, nogil=True)
def _bias_correction(n_samples: int, n_vars: int) -> float:
    dterm = (math.log(2.0) - math.log(n_samples - 1.0)) / 2.0
    total = n_vars * dterm
    for dim in range(1, n_vars + 1):
        total += _digamma_scalar((n_samples - dim) / 2.0) / 2.0
    return total


@njit(cache=True, nogil=True)
def _covariance_from_sums(sum_vec: np.ndarray, sum_outer: np.ndarray, count: int, demeaned: bool) -> np.ndarray:
    n_dim = sum_vec.shape[0]
    cov = np.empty((n_dim, n_dim), dtype=sum_outer.dtype)
    scale = 1.0 / (count - 1.0)
    alpha = -1.0 / count
    for i in range(n_dim):
        for j in range(n_dim):
            value = sum_outer[i, j]
            if not demeaned:
                value += alpha * sum_vec[i] * sum_vec[j]
            cov[i, j] = value * scale
    return cov


@njit(cache=True, nogil=True)
def _cross_covariance_from_sums(
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_xy: np.ndarray,
    count: int,
    demeaned: bool,
) -> np.ndarray:
    out = np.empty(sum_xy.shape, dtype=sum_xy.dtype)
    scale = 1.0 / (count - 1.0)
    alpha = -1.0 / count
    for i in range(sum_xy.shape[0]):
        for j in range(sum_xy.shape[1]):
            value = sum_xy[i, j]
            if not demeaned:
                value += alpha * sum_x[i] * sum_y[j]
            out[i, j] = value * scale
    return out


@njit(cache=True, nogil=True)
def _logdet_from_covariance(cov: np.ndarray) -> float:
    chol = np.linalg.cholesky(cov)
    total = 0.0
    for i in range(chol.shape[0]):
        total += math.log(chol[i, i])
    return total


@njit(cache=True, nogil=True)
def _shared_continuous_stats(y: np.ndarray, biascorrect: bool, demeaned: bool) -> tuple[np.ndarray, float]:
    y_dim, n_samples = y.shape
    sy = np.zeros(y_dim, dtype=y.dtype)
    syy = np.zeros((y_dim, y_dim), dtype=y.dtype)
    for trial in range(n_samples):
        for i in range(y_dim):
            value_i = y[i, trial]
            sy[i] += value_i
            for j in range(y_dim):
                syy[i, j] += value_i * y[j, trial]
    cov_y = _covariance_from_sums(sy, syy, n_samples, demeaned)
    hy = _logdet_from_covariance(cov_y)
    if biascorrect:
        hy -= _bias_correction(n_samples, y_dim)
    return cov_y, hy


@njit(cache=True, nogil=True)
def _ndtri(p: float) -> float:
    # Peter J. Acklam's inverse-normal approximation.
    a0 = -3.969683028665376e01
    a1 = 2.209460984245205e02
    a2 = -2.759285104469687e02
    a3 = 1.383577518672690e02
    a4 = -3.066479806614716e01
    a5 = 2.506628277459239e00

    b0 = -5.447609879822406e01
    b1 = 1.615858368580409e02
    b2 = -1.556989798598866e02
    b3 = 6.680131188771972e01
    b4 = -1.328068155288572e01

    c0 = -7.784894002430293e-03
    c1 = -3.223964580411365e-01
    c2 = -2.400758277161838e00
    c3 = -2.549732539343734e00
    c4 = 4.374664141464968e00
    c5 = 2.938163982698783e00

    d0 = 7.784695709041462e-03
    d1 = 3.224671290700398e-01
    d2 = 2.445134137142996e00
    d3 = 3.754408661907416e00

    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
        ) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
    if phigh < p:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
        ) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (
        (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
    ) / (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)


@njit(cache=True, nogil=True, parallel=True)
def copnorm_slice_numba(x: np.ndarray) -> np.ndarray:
    n_page, n_samples = x.shape
    out = np.empty_like(x)
    denom = n_samples + 1.0
    for page in prange(n_page):
        order = np.argsort(x[page])
        ranks = np.empty(n_samples, dtype=np.int64)
        for idx in range(n_samples):
            ranks[order[idx]] = idx + 1
        for idx in range(n_samples):
            out[page, idx] = _ndtri(ranks[idx] / denom)
    return out


@njit(cache=True, nogil=True)
def _mi_cc_pair(x: np.ndarray, y: np.ndarray, biascorrect: bool, demeaned: bool) -> float:
    x_dim, n_samples = x.shape
    y_dim = y.shape[0]
    xy_dim = x_dim + y_dim

    sx = np.zeros(x_dim, dtype=x.dtype)
    sy = np.zeros(y_dim, dtype=y.dtype)
    sxx = np.zeros((x_dim, x_dim), dtype=x.dtype)
    syy = np.zeros((y_dim, y_dim), dtype=y.dtype)
    sxy = np.zeros((x_dim, y_dim), dtype=x.dtype)

    for trial in range(n_samples):
        for i in range(x_dim):
            xv = x[i, trial]
            sx[i] += xv
            for j in range(x_dim):
                sxx[i, j] += xv * x[j, trial]
            for j in range(y_dim):
                sxy[i, j] += xv * y[j, trial]
        for i in range(y_dim):
            yv = y[i, trial]
            sy[i] += yv
            for j in range(y_dim):
                syy[i, j] += yv * y[j, trial]

    cov_x = _covariance_from_sums(sx, sxx, n_samples, demeaned)
    cov_y = _covariance_from_sums(sy, syy, n_samples, demeaned)
    cov_xy_cross = _cross_covariance_from_sums(sx, sy, sxy, n_samples, demeaned)

    cov_xy = np.empty((xy_dim, xy_dim), dtype=x.dtype)
    for i in range(x_dim):
        for j in range(x_dim):
            cov_xy[i, j] = cov_x[i, j]
        for j in range(y_dim):
            cov_xy[i, x_dim + j] = cov_xy_cross[i, j]
    for i in range(y_dim):
        for j in range(x_dim):
            cov_xy[x_dim + i, j] = cov_xy_cross[j, i]
        for j in range(y_dim):
            cov_xy[x_dim + i, x_dim + j] = cov_y[i, j]

    hx = _logdet_from_covariance(cov_x)
    hy = _logdet_from_covariance(cov_y)
    hxy = _logdet_from_covariance(cov_xy)
    if biascorrect:
        hx -= _bias_correction(n_samples, x_dim)
        hy -= _bias_correction(n_samples, y_dim)
        hxy -= _bias_correction(n_samples, xy_dim)
    return (hx + hy - hxy) / _LN2


@njit(cache=True, nogil=True, parallel=True)
def info_c1d_slice_numba(x: np.ndarray, y: np.ndarray, counts: np.ndarray, biascorrect: bool) -> np.ndarray:
    n_page, n_samples = x.shape
    n_classes = counts.shape[0]
    out = np.empty(n_page, dtype=x.dtype)
    weights = counts.astype(x.dtype) / n_samples

    unc_bias = _bias_correction(n_samples, 1) if biascorrect else 0.0
    cond_bias = np.zeros(n_classes, dtype=x.dtype)
    if biascorrect:
        for cls in range(n_classes):
            cond_bias[cls] = _bias_correction(int(counts[cls]), 1)

    for page in prange(n_page):
        sx = 0.0
        sxx = 0.0
        sxg = np.zeros(n_classes, dtype=x.dtype)
        sxxg = np.zeros(n_classes, dtype=x.dtype)

        for trial in range(n_samples):
            cls = y[trial]
            value = x[page, trial]
            sx += value
            sxx += value * value
            sxg[cls] += value
            sxxg[cls] += value * value

        hunc = 0.5 * math.log((sxx - (sx * sx) / n_samples) / (n_samples - 1.0))
        if biascorrect:
            hunc -= unc_bias

        hcond = 0.0
        for cls in range(n_classes):
            n_cls = counts[cls]
            logdet = 0.5 * math.log((sxxg[cls] - (sxg[cls] * sxg[cls]) / n_cls) / (n_cls - 1.0))
            if biascorrect:
                logdet -= cond_bias[cls]
            hcond += weights[cls] * logdet

        out[page] = (hunc - hcond) / _LN2
    return out


@njit(cache=True, nogil=True, parallel=True)
def info_cd_slice_numba(x: np.ndarray, y: np.ndarray, counts: np.ndarray, biascorrect: bool) -> np.ndarray:
    n_page, x_dim, n_samples = x.shape
    n_classes = counts.shape[0]
    out = np.empty(n_page, dtype=x.dtype)
    weights = counts.astype(x.dtype) / n_samples

    unc_bias = _bias_correction(n_samples, x_dim) if biascorrect else 0.0
    cond_bias = np.zeros(n_classes, dtype=x.dtype)
    if biascorrect:
        for cls in range(n_classes):
            cond_bias[cls] = _bias_correction(int(counts[cls]), x_dim)

    for page in prange(n_page):
        sx = np.zeros(x_dim, dtype=x.dtype)
        sxx = np.zeros((x_dim, x_dim), dtype=x.dtype)
        sxg = np.zeros((n_classes, x_dim), dtype=x.dtype)
        sxxg = np.zeros((n_classes, x_dim, x_dim), dtype=x.dtype)

        for trial in range(n_samples):
            cls = y[trial]
            for i in range(x_dim):
                value_i = x[page, i, trial]
                sx[i] += value_i
                sxg[cls, i] += value_i
                for j in range(x_dim):
                    prod = value_i * x[page, j, trial]
                    sxx[i, j] += prod
                    sxxg[cls, i, j] += prod

        cov_x = _covariance_from_sums(sx, sxx, n_samples, False)
        hunc = _logdet_from_covariance(cov_x)
        if biascorrect:
            hunc -= unc_bias

        hcond = 0.0
        for cls in range(n_classes):
            cov_cls = _covariance_from_sums(sxg[cls], sxxg[cls], int(counts[cls]), False)
            logdet = _logdet_from_covariance(cov_cls)
            if biascorrect:
                logdet -= cond_bias[cls]
            hcond += weights[cls] * logdet

        out[page] = (hunc - hcond) / _LN2
    return out


@njit(cache=True, nogil=True, parallel=True)
def info_dc_slice_numba(x: np.ndarray, y: np.ndarray, n_classes: int, biascorrect: bool) -> np.ndarray:
    n_page, n_samples = x.shape
    y_dim = y.shape[0]
    out = np.empty(n_page, dtype=y.dtype)

    cov_y, hunc = _shared_continuous_stats(y, biascorrect, False)
    _ = cov_y

    for page in prange(n_page):
        counts = np.zeros(n_classes, dtype=np.int64)
        syg = np.zeros((n_classes, y_dim), dtype=y.dtype)
        syyg = np.zeros((n_classes, y_dim, y_dim), dtype=y.dtype)

        for trial in range(n_samples):
            cls = x[page, trial]
            counts[cls] += 1
            for i in range(y_dim):
                value_i = y[i, trial]
                syg[cls, i] += value_i
                for j in range(y_dim):
                    syyg[cls, i, j] += value_i * y[j, trial]

        hcond = 0.0
        for cls in range(n_classes):
            count = counts[cls]
            cov_cls = _covariance_from_sums(syg[cls], syyg[cls], count, False)
            logdet = _logdet_from_covariance(cov_cls)
            if biascorrect:
                logdet -= _bias_correction(count, y_dim)
            hcond += (count / n_samples) * logdet

        out[page] = (hunc - hcond) / _LN2
    return out


@njit(cache=True, nogil=True, parallel=True)
def info_cc_slice_numba(
    x: np.ndarray,
    y: np.ndarray,
    cov_y: np.ndarray,
    hy: float,
    biascorrect: bool,
    demeaned: bool,
) -> np.ndarray:
    n_page, x_dim, n_samples = x.shape
    y_dim = y.shape[0]
    out = np.empty(n_page, dtype=x.dtype)

    hy_term = hy
    bias_x = _bias_correction(n_samples, x_dim) if biascorrect else 0.0
    bias_xy = _bias_correction(n_samples, x_dim + y_dim) if biascorrect else 0.0

    sy = np.zeros(y_dim, dtype=y.dtype)
    syy = np.zeros((y_dim, y_dim), dtype=y.dtype)
    if not demeaned:
        for trial in range(n_samples):
            for i in range(y_dim):
                value_i = y[i, trial]
                sy[i] += value_i
                for j in range(y_dim):
                    syy[i, j] += value_i * y[j, trial]

    for page in prange(n_page):
        sx = np.zeros(x_dim, dtype=x.dtype)
        sxx = np.zeros((x_dim, x_dim), dtype=x.dtype)
        sxy = np.zeros((x_dim, y_dim), dtype=x.dtype)

        for trial in range(n_samples):
            for i in range(x_dim):
                value_i = x[page, i, trial]
                sx[i] += value_i
                for j in range(x_dim):
                    sxx[i, j] += value_i * x[page, j, trial]
                for j in range(y_dim):
                    sxy[i, j] += value_i * y[j, trial]

        cov_x = _covariance_from_sums(sx, sxx, n_samples, demeaned)
        cov_xy_cross = _cross_covariance_from_sums(sx, sy, sxy, n_samples, demeaned)
        cov_xy = np.empty((x_dim + y_dim, x_dim + y_dim), dtype=x.dtype)
        for i in range(x_dim):
            for j in range(x_dim):
                cov_xy[i, j] = cov_x[i, j]
            for j in range(y_dim):
                cov_xy[i, x_dim + j] = cov_xy_cross[i, j]
        for i in range(y_dim):
            for j in range(x_dim):
                cov_xy[x_dim + i, j] = cov_xy_cross[j, i]
            for j in range(y_dim):
                cov_xy[x_dim + i, x_dim + j] = cov_y[i, j]

        hx = _logdet_from_covariance(cov_x)
        hxy = _logdet_from_covariance(cov_xy)
        if biascorrect:
            hx -= bias_x
            hxy -= bias_xy
        out[page] = (hx + hy_term - hxy) / _LN2
    return out


@njit(cache=True, nogil=True, parallel=True)
def info_cc_multi_numba(x: np.ndarray, y: np.ndarray, biascorrect: bool, demeaned: bool) -> np.ndarray:
    n_page = x.shape[0]
    out = np.empty(n_page, dtype=x.dtype)
    for page in prange(n_page):
        out[page] = _mi_cc_pair(x[page], y[page], biascorrect, demeaned)
    return out
