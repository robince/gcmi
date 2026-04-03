"""Runtime version lookup for gcmi.

The authoritative version lives in ``python/pyproject.toml``. Installed
packages read it from package metadata; source-tree imports fall back to the
local pyproject file.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path


def _version_from_pyproject() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(pyproject)
    in_project = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version = "):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"Unable to determine version from {pyproject}")


def get_version() -> str:
    try:
        return _version_from_pyproject()
    except (FileNotFoundError, RuntimeError):
        pass
    try:
        return dist_version("gcmi")
    except PackageNotFoundError:
        return _version_from_pyproject()


__version__ = get_version()
