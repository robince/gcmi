"""Backward-compatible source-tree shim for the packaged Python API."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_PKG_PATH = _SRC / "gcmi" / "__init__.py"
_SPEC = spec_from_file_location(
    "_gcmi_pkg",
    _PKG_PATH,
    submodule_search_locations=[str(_SRC / "gcmi")],
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import-time guard
    raise ImportError(f"Unable to load gcmi package implementation from {_PKG_PATH}")

_CORE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CORE
_SPEC.loader.exec_module(_CORE)

__all__ = list(getattr(_CORE, "__all__", ()))
__version__ = getattr(_CORE, "__version__", "0.0.0")
__doc__ = _CORE.__doc__

for _name in __all__:
    globals()[_name] = getattr(_CORE, _name)

del _name, _CORE, _PKG_PATH, _SPEC, _SRC, module_from_spec, spec_from_file_location, Path, sys
