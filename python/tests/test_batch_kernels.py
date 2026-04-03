from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import gcmi

HAS_NUMBA = importlib.util.find_spec("numba") is not None
requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")


@pytest.fixture(autouse=True)
def restore_backend() -> None:
    original = gcmi.get_backend()
    try:
        yield
    finally:
        gcmi.set_backend(original)


def test_copnorm_slice_reference_and_numba_follow_legacy_rank_policy() -> None:
    x = np.array(
        [
            [3.0, 1.0, 2.0, 4.0],
            [1.5, -1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    expected = []
    for page in x:
        order = np.argsort(page)
        ranks = np.empty(page.shape[0], dtype=np.int64)
        ranks[order] = np.arange(1, page.shape[0] + 1, dtype=np.int64)
        expected.append(gcmi.copnorm(ranks.astype(np.float64)))
    expected = np.asarray(expected, dtype=np.float32)

    reference = gcmi.copnorm_slice(x, backend="reference")
    if not HAS_NUMBA:
        np.testing.assert_allclose(reference, expected, rtol=1e-6, atol=1e-6)
        return
    numba = gcmi.copnorm_slice(x, backend="numba")
    np.testing.assert_allclose(reference, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(numba, reference, rtol=1e-6, atol=1e-6)
    assert numba.dtype == np.float32


@requires_numba
def test_info_c1d_slice_matches_pagewise_scalar_for_reference_and_numba() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 48))
    y = np.repeat(np.arange(4, dtype=np.int64), 12)
    expected = np.array([gcmi.mi_model_gd(page, y, 4) for page in x])

    np.testing.assert_allclose(gcmi.info_c1d_slice(x, y, 4, backend="reference"), expected)
    np.testing.assert_allclose(gcmi.info_c1d_slice(x, y, 4, backend="numba"), expected)


@requires_numba
def test_info_cd_slice_matches_pagewise_scalar_for_reference_and_numba() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 3, 60))
    y = np.repeat(np.arange(5, dtype=np.int64), 12)
    expected = np.array([gcmi.mi_model_gd(page, y, 5) for page in x])

    np.testing.assert_allclose(gcmi.info_cd_slice(x, y, 5, backend="reference"), expected)
    np.testing.assert_allclose(gcmi.info_cd_slice(x, y, 5, backend="numba"), expected)


@requires_numba
def test_info_dc_slice_matches_symmetric_scalar_formula_for_reference_and_numba() -> None:
    rng = np.random.default_rng(2)
    x = np.vstack(
        [
            np.repeat(np.arange(4, dtype=np.int64), 15),
            np.tile(np.repeat(np.arange(4, dtype=np.int64), 15), 1),
            np.repeat(np.array([0, 1, 2, 3], dtype=np.int64), 15),
        ]
    )
    y = rng.standard_normal((2, 60))
    expected = np.array([gcmi.mi_model_gd(y, page, 4) for page in x])

    np.testing.assert_allclose(gcmi.info_dc_slice(x, y, 4, backend="reference"), expected)
    np.testing.assert_allclose(gcmi.info_dc_slice(x, y, 4, backend="numba"), expected)


@requires_numba
def test_info_cc_slice_matches_pagewise_scalar_for_reference_and_numba() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 2, 64))
    y = rng.standard_normal((3, 64))
    expected = np.array([gcmi.mi_gg(page, y) for page in x])

    np.testing.assert_allclose(gcmi.info_cc_slice(x, y, backend="reference"), expected)
    np.testing.assert_allclose(gcmi.info_cc_slice(x, y, backend="numba"), expected)


@requires_numba
def test_info_cc_multi_and_indexed_match_pagewise_scalar_for_reference_and_numba() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 2, 80))
    y = rng.standard_normal((5, 3, 80))
    expected_multi = np.array([gcmi.mi_gg(x_page, y_page) for x_page, y_page in zip(x, y, strict=True)])
    idx = np.array([3, 1, 3, 0], dtype=np.int64)
    shared_y = rng.standard_normal((3, 80))
    expected_indexed = np.array([gcmi.mi_gg(x[i], shared_y) for i in idx])

    np.testing.assert_allclose(gcmi.info_cc_multi(x, y, backend="reference"), expected_multi)
    np.testing.assert_allclose(gcmi.info_cc_multi(x, y, backend="numba"), expected_multi)
    np.testing.assert_allclose(
        gcmi.info_cc_slice_indexed(x, idx, shared_y, backend="reference"),
        expected_indexed,
    )
    np.testing.assert_allclose(
        gcmi.info_cc_slice_indexed(x, idx, shared_y, backend="numba"),
        expected_indexed,
    )


@requires_numba
def test_batch_kernels_support_float32_with_relaxed_tolerance() -> None:
    rng = np.random.default_rng(5)
    x_cd = rng.standard_normal((4, 2, 48), dtype=np.float32)
    y_disc = np.repeat(np.arange(4, dtype=np.int64), 12)
    x_cc = rng.standard_normal((4, 2, 48), dtype=np.float32)
    y_cc = rng.standard_normal((3, 48), dtype=np.float32)

    cd_ref = gcmi.info_cd_slice(x_cd, y_disc, 4, backend="reference")
    cd_numba = gcmi.info_cd_slice(x_cd, y_disc, 4, backend="numba")
    cc_ref = gcmi.info_cc_slice(x_cc, y_cc, backend="reference")
    cc_numba = gcmi.info_cc_slice(x_cc, y_cc, backend="numba")

    assert cd_numba.dtype == np.float32
    assert cc_numba.dtype == np.float32
    np.testing.assert_allclose(cd_numba, cd_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cc_numba, cc_ref, rtol=1e-5, atol=1e-6)


@requires_numba
def test_backend_controls_and_auto_dispatch() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((3, 2, 40))
    y = rng.standard_normal((2, 40))

    gcmi.set_backend("reference")
    ref = gcmi.info_cc_slice(x, y, backend="auto")
    np.testing.assert_allclose(ref, gcmi.info_cc_slice(x, y, backend="reference"))

    gcmi.set_backend("numba")
    opt = gcmi.info_cc_slice(x, y, backend="auto")
    np.testing.assert_allclose(opt, gcmi.info_cc_slice(x, y, backend="numba"))

    info = gcmi.get_backend_info()
    assert info["default_backend"] == "numba"
    assert info["numba_available"] is True


def test_source_tree_shim_exports_batch_api() -> None:
    shim_path = Path(__file__).resolve().parents[1] / "gcmi.py"
    spec = importlib.util.spec_from_file_location("gcmi_source_tree_shim", shim_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "info_cc_slice")
    assert hasattr(module, "get_backend_info")
