"""Backend selection for optional optimized kernels."""

from __future__ import annotations

import os
from typing import Any

_ALLOWED_BACKENDS = {"auto", "reference", "numba"}
_DEFAULT_BACKEND = os.environ.get("GCMI_BACKEND", "auto").strip().lower() or "auto"
if _DEFAULT_BACKEND not in _ALLOWED_BACKENDS:
    raise ValueError(
        "GCMI_BACKEND must be one of: auto, reference, numba; "
        f"got {_DEFAULT_BACKEND!r}"
    )


def _numba_module():
    try:
        import numba  # type: ignore
    except ImportError:
        return None
    return numba


def numba_available() -> bool:
    return _numba_module() is not None


def set_backend(mode: str) -> None:
    """Set the process-wide default backend."""

    global _DEFAULT_BACKEND
    normalized = mode.strip().lower()
    if normalized not in _ALLOWED_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_ALLOWED_BACKENDS)}")
    _DEFAULT_BACKEND = normalized


def get_backend() -> str:
    """Return the process-wide default backend."""

    return _DEFAULT_BACKEND


def resolve_backend(requested: str, operation: str, *, numba_supported: bool) -> str:
    """Resolve a requested backend to a concrete implementation."""

    normalized = requested.strip().lower()
    if normalized not in _ALLOWED_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_ALLOWED_BACKENDS)}")

    if normalized == "auto":
        normalized = _DEFAULT_BACKEND

    if normalized == "auto":
        return "numba" if numba_supported and numba_available() else "reference"

    if normalized == "reference":
        return "reference"

    if not numba_available():
        raise RuntimeError(
            f"Numba backend requested for {operation}, but numba is not installed"
        )
    if not numba_supported:
        raise ValueError(
            f"Numba backend is not available for {operation} with the provided inputs"
        )
    return "numba"


def get_backend_info() -> dict[str, Any]:
    """Return backend and threading metadata for diagnostics."""

    info: dict[str, Any] = {
        "default_backend": _DEFAULT_BACKEND,
        "env_backend": os.environ.get("GCMI_BACKEND"),
        "numba_available": False,
        "numba_threading_layer": None,
        "numba_num_threads": None,
        "numba_version": None,
    }

    numba = _numba_module()
    if numba is None:
        return info

    info["numba_available"] = True
    info["numba_version"] = getattr(numba, "__version__", None)
    info["numba_num_threads"] = numba.get_num_threads()
    try:
        info["numba_threading_layer"] = numba.threading_layer()
    except ValueError:
        info["numba_threading_layer"] = None
    return info
