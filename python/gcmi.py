"""Backward-compatible source-tree shim for the packaged Python API."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_CORE_PATH = Path(__file__).resolve().parent / "src" / "gcmi" / "_core.py"
_SPEC = spec_from_file_location("_gcmi_core", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import-time guard
    raise ImportError(f"Unable to load gcmi core implementation from {_CORE_PATH}")

_CORE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CORE)

__all__ = list(getattr(_CORE, "__all__", ()))
__version__ = getattr(_CORE, "__version__", "0.0.0")
__doc__ = _CORE.__doc__

for _name in __all__:
    globals()[_name] = getattr(_CORE, _name)

del _name, _CORE, _CORE_PATH, _SPEC, module_from_spec, spec_from_file_location, Path
