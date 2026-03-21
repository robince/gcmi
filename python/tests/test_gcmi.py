from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

import gcmi


def digamma_ref(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)

    def _scalar(v: float) -> float:
        if v <= 0.0:
            if float(v).is_integer():
                raise ValueError("digamma undefined at non-positive integers")
            return _scalar(1.0 - v) - math.pi / math.tan(math.pi * v)
        result = 0.0
        while v < 8.0:
            result -= 1.0 / v
            v += 1.0
        inv = 1.0 / v
        inv2 = inv * inv
        result += math.log(v) - 0.5 * inv - inv2 * (
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

    if arr.ndim == 0:
        return float(_scalar(float(arr)))
    return np.vectorize(_scalar, otypes=[float])(arr)


def gaussian_bias_correction(n_samples: int, n_vars: int) -> float:
    dims = np.arange(1, n_vars + 1, dtype=float)
    psi_terms = digamma_ref((n_samples - dims) / 2.0) / 2.0
    dterm = (math.log(2.0) - math.log(n_samples - 1.0)) / 2.0
    return float(n_vars * dterm + psi_terms.sum())


def entropy_bits(x: np.ndarray, biascorrect: bool = True, demeaned: bool = False) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if not demeaned:
        x = x - x.mean(axis=1, keepdims=True)
    cov = (x @ x.T) / float(x.shape[1] - 1)
    chol = np.linalg.cholesky(cov)
    h = np.log(np.diagonal(chol)).sum() + 0.5 * x.shape[0] * (math.log(2.0 * math.pi) + 1.0)
    if biascorrect:
        h -= gaussian_bias_correction(x.shape[1], x.shape[0])
    return h / math.log(2.0)


def mi_bits(
    x: np.ndarray,
    y: np.ndarray,
    biascorrect: bool = True,
    demeaned: bool = False,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if y.ndim == 1:
        y = y[np.newaxis, :]
    xy = np.vstack((x, y))
    if not demeaned:
        xy = xy - xy.mean(axis=1, keepdims=True)
    cov = (xy @ xy.T) / float(xy.shape[1] - 1)
    nx = x.shape[0]
    ny = y.shape[0]
    chol_x = np.linalg.cholesky(cov[:nx, :nx])
    chol_y = np.linalg.cholesky(cov[nx:, nx:])
    chol_xy = np.linalg.cholesky(cov)
    hx = np.log(np.diagonal(chol_x)).sum()
    hy = np.log(np.diagonal(chol_y)).sum()
    hxy = np.log(np.diagonal(chol_xy)).sum()
    if biascorrect:
        corr = gaussian_bias_correction(xy.shape[1], nx + ny)
        dims = np.arange(1, nx + ny + 1, dtype=float)
        psi_terms = digamma_ref((xy.shape[1] - dims) / 2.0) / 2.0
        dterm = (math.log(2.0) - math.log(xy.shape[1] - 1.0)) / 2.0
        hx -= nx * dterm + psi_terms[:nx].sum()
        hy -= ny * dterm + psi_terms[:ny].sum()
        hxy -= corr
    return (hx + hy - hxy) / math.log(2.0)


def cmi_bits(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    biascorrect: bool = True,
    demeaned: bool = False,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if y.ndim == 1:
        y = y[np.newaxis, :]
    if z.ndim == 1:
        z = z[np.newaxis, :]
    xyz = np.vstack((x, y, z))
    if not demeaned:
        xyz = xyz - xyz.mean(axis=1, keepdims=True)
    cov = (xyz @ xyz.T) / float(xyz.shape[1] - 1)
    nx = x.shape[0]
    ny = y.shape[0]
    nz = z.shape[0]
    cov_z = cov[nx + ny :, nx + ny :]
    cov_xz = np.zeros((nx + nz, nx + nz), dtype=float)
    cov_xz[:nx, :nx] = cov[:nx, :nx]
    cov_xz[:nx, nx:] = cov[:nx, nx + ny :]
    cov_xz[nx:, :nx] = cov[nx + ny :, :nx]
    cov_xz[nx:, nx:] = cov[nx + ny :, nx + ny :]
    cov_yz = cov[nx:, nx:]
    chol_z = np.linalg.cholesky(cov_z)
    chol_xz = np.linalg.cholesky(cov_xz)
    chol_yz = np.linalg.cholesky(cov_yz)
    chol_xyz = np.linalg.cholesky(cov)
    hz = np.log(np.diagonal(chol_z)).sum()
    hxz = np.log(np.diagonal(chol_xz)).sum()
    hyz = np.log(np.diagonal(chol_yz)).sum()
    hxyz = np.log(np.diagonal(chol_xyz)).sum()
    if biascorrect:
        dims = np.arange(1, nx + ny + nz + 1, dtype=float)
        psi_terms = digamma_ref((xyz.shape[1] - dims) / 2.0) / 2.0
        dterm = (math.log(2.0) - math.log(xyz.shape[1] - 1.0)) / 2.0
        hz -= nz * dterm + psi_terms[:nz].sum()
        hxz -= (nx + nz) * dterm + psi_terms[: nx + nz].sum()
        hyz -= (ny + nz) * dterm + psi_terms[: ny + nz].sum()
        hxyz -= (nx + ny + nz) * dterm + psi_terms[: nx + ny + nz].sum()
    return (hxz + hyz - hxyz - hz) / math.log(2.0)


def weighted_logsumexp(a: np.ndarray, weights: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    weights = np.asarray(weights, dtype=float)
    max_a = np.max(a, axis=0, keepdims=True)
    shifted = np.exp(a - max_a) * weights.reshape((-1, 1))
    return np.squeeze(max_a + np.log(np.sum(shifted, axis=0, keepdims=True)), axis=0)


def mixture_bits(x: np.ndarray, y: np.ndarray, Ym: int) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    n_vars = x.shape[0]
    counts = np.bincount(y, minlength=Ym).astype(float)
    weights = counts / float(x.shape[1])
    means = np.zeros((Ym, n_vars), dtype=float)
    chols = np.zeros((Ym, n_vars, n_vars), dtype=float)
    hcond = np.zeros(Ym, dtype=float)
    for yi in range(Ym):
        xm = x[:, y == yi]
        means[yi] = xm.mean(axis=1)
        xm = xm - means[yi][:, np.newaxis]
        cov = (xm @ xm.T) / float(xm.shape[1] - 1)
        chol = np.linalg.cholesky(cov)
        chols[yi] = chol
        hcond[yi] = np.log(np.diagonal(chol)).sum() + 0.5 * n_vars * (math.log(2.0 * math.pi) + 1.0)
    hmix = 0.0
    scale = math.sqrt(n_vars)
    for yi in range(Ym):
        sigma = scale * chols[yi].T
        sigma_points = np.hstack(
            [means[yi][:, np.newaxis] + sigma, means[yi][:, np.newaxis] - sigma]
        )
        log_lik = np.zeros((Ym, 2 * n_vars), dtype=float)
        for mi in range(Ym):
            dx = sigma_points - means[mi][:, np.newaxis]
            solved = np.linalg.solve(chols[mi], dx)
            log_lik[mi] = -0.5 * np.sum(solved * solved, axis=0) - hcond[mi] + 0.5 * n_vars
        hmix += weights[yi] * weighted_logsumexp(log_lik, weights).sum()
    hmix = -hmix / (2.0 * n_vars)
    return (hmix - float(np.sum(weights * hcond))) / math.log(2.0)


def test_ctransform_and_copnorm_follow_last_axis_ranks() -> None:
    x = np.array([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]])
    expected = np.array([[0.75, 0.25, 0.5], [0.5, 0.25, 0.75]])
    np.testing.assert_allclose(gcmi.ctransform(x), expected)
    nd = NormalDist()
    np.testing.assert_allclose(
        gcmi.copnorm(x),
        np.vectorize(nd.inv_cdf, otypes=[float])(expected),
    )


def test_ent_g_matches_manual_covariance_formula() -> None:
    x = np.array([[1.0, 0.0, -1.0, 2.0], [0.5, -0.5, 1.5, 0.0]])
    expected = entropy_bits(x, biascorrect=True)
    assert gcmi.ent_g(x) == pytest.approx(expected)


def test_mi_gg_matches_manual_formula_and_rejects_small_samples() -> None:
    x = np.array([1.0, 2.0, -1.0, 0.5])
    y = np.array([0.5, -0.5, 1.5, 0.0])
    expected = mi_bits(x, y, biascorrect=True)
    assert gcmi.mi_gg(x, y) == pytest.approx(expected)
    with pytest.raises(ValueError, match="more samples than variables"):
        gcmi.mi_gg([1.0, 0.0], [0.0, 1.0], biascorrect=False)


def test_gcmi_cc_wraps_copnorm_and_warns_on_repeats() -> None:
    x = np.array([3.0, 1.0, 2.0, 4.0])
    y = np.array([4.0, 2.0, 1.0, 3.0])
    expected = gcmi.mi_gg(gcmi.copnorm(x), gcmi.copnorm(y), biascorrect=True, demeaned=True)
    assert gcmi.gcmi_cc(x, y) == pytest.approx(expected)

    repeated = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    with pytest.warns(UserWarning, match="more than 10% repeated values"):
        gcmi.gcmi_cc(repeated, np.array([0.0, 4.0, 1.0, 3.0, 2.0]))


def test_mi_model_gd_matches_manual_formula_and_rejects_empty_classes() -> None:
    x = np.array([[0.0, 1.0, 2.0, 0.5, 1.5, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    expected = gcmi.mi_model_gd(x, y, 2)
    manual = mi_model_gd_manual(x, y, 2)
    assert expected == pytest.approx(manual)

    with pytest.raises(ValueError, match="empty classes"):
        gcmi.mi_model_gd(x, np.array([0, 0, 0, 1, 1, 1], dtype=np.int64), 3)


def mi_model_gd_manual(x: np.ndarray, y: np.ndarray, Ym: int) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    counts = np.bincount(y, minlength=Ym).astype(float)
    hcond = np.zeros(Ym, dtype=float)
    for yi in range(Ym):
        xm = x[:, y == yi]
        xm = xm - xm.mean(axis=1, keepdims=True)
        cov = (xm @ xm.T) / float(xm.shape[1] - 1)
        chol = np.linalg.cholesky(cov)
        hcond[yi] = np.log(np.diagonal(chol)).sum()
        hcond[yi] -= gaussian_bias_correction(xm.shape[1], x.shape[0])
    x = x - x.mean(axis=1, keepdims=True)
    cov = (x @ x.T) / float(x.shape[1] - 1)
    chol = np.linalg.cholesky(cov)
    hunc = np.log(np.diagonal(chol)).sum() - gaussian_bias_correction(x.shape[1], x.shape[0])
    return (hunc - float(np.sum((counts / float(x.shape[1])) * hcond))) / math.log(2.0)


def test_gcmi_model_cd_matches_wrapped_model_estimator() -> None:
    x = np.array([[0.0, 1.0, 2.0, 0.5, 1.5, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    expected = gcmi.mi_model_gd(gcmi.copnorm(x), y, 2, biascorrect=True, demeaned=True)
    assert gcmi.gcmi_model_cd(x, y, 2) == pytest.approx(expected)


def test_mi_mixture_gd_matches_manual_small_case() -> None:
    x = np.array([[0.0, 1.0, 2.0, 2.5, 3.0, 4.0]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    expected = mixture_bits(x, y, 2)
    assert gcmi.mi_mixture_gd(x, y, 2) == pytest.approx(expected)


def test_gcmi_mixture_cd_matches_wrapped_mixture_estimator() -> None:
    x = np.array([[0.0, 1.0, 2.0, 2.5, 3.0, 4.0]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    transformed = []
    relabeled = []
    for yi in range(2):
        idx = y == yi
        xm = x[:, idx]
        cxm = gcmi.copnorm(xm)
        median = np.median(xm, axis=1, keepdims=True)
        mad = np.median(np.abs(xm - median), axis=1, keepdims=True)
        transformed.append(cxm * (1.482602218505602 * mad) + median)
        relabeled.append(np.full(xm.shape[1], yi, dtype=np.int64))
    pooled_x = np.concatenate(transformed, axis=1)
    pooled_y = np.concatenate(relabeled)
    expected = gcmi.mi_mixture_gd(pooled_x, pooled_y, 2)
    assert gcmi.gcmi_mixture_cd(x, y, 2) == pytest.approx(expected)


def test_cmi_ggg_matches_manual_formula() -> None:
    x = np.array([1.0, 2.0, -1.0, 0.5])
    y = np.array([0.5, -0.5, 1.5, 0.0])
    z = np.array([2.0, 0.0, 1.0, 3.0])
    expected = cmi_bits(x, y, z, biascorrect=True)
    assert gcmi.cmi_ggg(x, y, z) == pytest.approx(expected)


def test_gccmi_ccc_and_ccd_match_expected_compositions() -> None:
    x = np.array([3.0, 1.0, 2.0, 4.0])
    y = np.array([4.0, 2.0, 1.0, 3.0])
    z = np.array([1.0, 4.0, 2.0, 3.0])
    assert gcmi.gccmi_ccc(x, y, z) == pytest.approx(
        gcmi.cmi_ggg(gcmi.copnorm(x), gcmi.copnorm(y), gcmi.copnorm(z), biascorrect=True, demeaned=True)
    )

    x2 = np.array([[0.0, 1.0, 2.0, 0.3, 1.7, 2.8]])
    y2 = np.array([[2.0, 0.0, 1.0, 1.1, 2.7, 0.4]])
    z2 = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    expected_cmi = []
    x_parts = []
    y_parts = []
    weights = []
    for zi in range(2):
        idx = z2 == zi
        weights.append(idx.sum())
        cx = gcmi.copnorm(x2[:, idx])
        cy = gcmi.copnorm(y2[:, idx])
        x_parts.append(cx)
        y_parts.append(cy)
        expected_cmi.append(gcmi.mi_gg(cx, cy, biascorrect=True, demeaned=True))
    pooled_expected = gcmi.mi_gg(np.hstack(x_parts), np.hstack(y_parts), biascorrect=True, demeaned=False)
    cmi, pooled = gcmi.gccmi_ccd(x2, y2, z2, 2)
    assert cmi == pytest.approx(np.sum(np.asarray(weights) / float(z2.size) * np.asarray(expected_cmi)))
    assert pooled == pytest.approx(pooled_expected)
