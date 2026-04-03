"""Backward-compatible source-tree shim for the packaged Python API."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_PKG_PATH = Path(__file__).resolve().parent / "src" / "gcmi"
_INIT_PATH = _PKG_PATH / "__init__.py"
_SPEC = spec_from_file_location(
    "_gcmi_source",
    _INIT_PATH,
    submodule_search_locations=[str(_PKG_PATH)],
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import-time guard
    raise ImportError(f"Unable to load gcmi package implementation from {_INIT_PATH}")

_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

__all__ = list(getattr(_MODULE, "__all__", ()))
__version__ = getattr(_MODULE, "__version__", "0.0.0")
__doc__ = _MODULE.__doc__

for _name in __all__:
    globals()[_name] = getattr(_MODULE, _name)

del _name, _INIT_PATH, _MODULE, _PKG_PATH, _SPEC, module_from_spec, spec_from_file_location, Path, sys
