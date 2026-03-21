"""Minimal PEP 517/660 backend for the gcmi package.

The repository is small and pure Python, so a lightweight in-tree backend keeps
installation self-contained and avoids depending on setuptools being present in
the execution environment.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

NAME = "gcmi"
VERSION = "0.4.0"
SUMMARY = "Gaussian-copula mutual information estimators"
REQUIRES_PYTHON = ">=3.10"
REQUIRES_DIST = ["numpy>=1.23"]

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PACKAGE = SRC / "gcmi"


@dataclass(frozen=True)
class Artifact:
    arcname: str
    source: Path | None = None
    data: bytes | None = None

    def read(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.source is None:
            raise ValueError("artifact has neither data nor source")
        return self.source.read_bytes()


def _dist_info_dir() -> str:
    return f"{NAME}-{VERSION}.dist-info"


def _wheel_name() -> str:
    return f"{NAME}-{VERSION}-py3-none-any.whl"


def _metadata() -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {NAME}",
        f"Version: {VERSION}",
        f"Summary: {SUMMARY}",
        f"Requires-Python: {REQUIRES_PYTHON}",
    ]
    for requirement in REQUIRES_DIST:
        lines.append(f"Requires-Dist: {requirement}")
    lines.append("")
    return "\n".join(lines)


def _wheel_metadata() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: gcmi.build_backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _package_artifacts(editable: bool) -> list[Artifact]:
    artifacts = [
        Artifact(f"{NAME}-{VERSION}.dist-info/METADATA", data=_metadata().encode()),
        Artifact(f"{NAME}-{VERSION}.dist-info/WHEEL", data=_wheel_metadata().encode()),
        Artifact(
            f"{NAME}-{VERSION}.dist-info/top_level.txt",
            data=f"{NAME}\n".encode(),
        ),
    ]
    if editable:
        artifacts.append(
            Artifact(
                f"{NAME}.pth",
                data=(str(SRC) + os.linesep).encode(),
            )
        )
    else:
        artifacts.extend(
            [
                Artifact(f"{NAME}/__init__.py", source=PACKAGE / "__init__.py"),
                Artifact(f"{NAME}/_core.py", source=PACKAGE / "_core.py"),
            ]
        )
    return artifacts


def _record_for(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _write_wheel(wheel_directory: str, editable: bool = False) -> str:
    wheel_path = Path(wheel_directory) / _wheel_name()
    artifacts = _package_artifacts(editable)
    record_rows: list[tuple[str, str, str]] = []

    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as zf:
        for artifact in artifacts:
            data = artifact.read()
            zf.writestr(_zipinfo(artifact.arcname), data)
            record_rows.append((artifact.arcname, _record_for(data), str(len(data))))

        record_rows.append((f"{_dist_info_dir()}/RECORD", "", ""))
        record_data = _render_record(record_rows)
        zf.writestr(_zipinfo(f"{_dist_info_dir()}/RECORD"), record_data)

    return wheel_path.name


def _zipinfo(name: str) -> ZipInfo:
    info = ZipInfo(filename=name)
    info.compress_type = ZIP_DEFLATED
    return info


def _render_record(rows: Iterable[tuple[str, str, str]]) -> bytes:
    fp = io.StringIO()
    writer = csv.writer(fp)
    for row in rows:
        writer.writerow(row)
    return fp.getvalue().encode("utf-8")


def _write_metadata(metadata_directory: str) -> str:
    dist_info = Path(metadata_directory) / _dist_info_dir()
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_metadata(), encoding="utf-8")
    (dist_info / "top_level.txt").write_text(f"{NAME}\n", encoding="utf-8")
    return dist_info.name


def get_requires_for_build_wheel(config_settings=None):  # noqa: D401
    return []


def get_requires_for_build_editable(config_settings=None):  # noqa: D401
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):  # noqa: D401
    return _write_metadata(metadata_directory)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):  # noqa: D401
    return _write_metadata(metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):  # noqa: D401
    return _write_wheel(wheel_directory, editable=False)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):  # noqa: D401
    return _write_wheel(wheel_directory, editable=True)


def build_sdist(sdist_directory, config_settings=None):  # noqa: D401
    sdist_name = f"{NAME}-{VERSION}.tar.gz"
    sdist_path = Path(sdist_directory) / sdist_name
    with tarfile.open(sdist_path, "w:gz") as tf:
        for path in [ROOT / "pyproject.toml", ROOT.parent / "README.md", ROOT / "build_backend.py"]:
            tf.add(path, arcname=f"{NAME}-{VERSION}/{path.name}")
        for path in (ROOT / "src").rglob("*"):
            if path.is_file():
                tf.add(path, arcname=f"{NAME}-{VERSION}/{path.relative_to(ROOT)}")
    return sdist_name
