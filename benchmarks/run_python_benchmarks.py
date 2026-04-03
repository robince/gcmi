from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import gcmi

try:
    import numba
except ImportError:  # pragma: no cover - benchmark environment specific
    numba = None


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    kernel: str
    dtype: str
    ntrl: int
    npage: int
    xdim: int | None = None
    ydim: int | None = None
    ym: int | None = None
    xm: int | None = None
    seed: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Python benchmark suite for GCMI batch kernels.")
    parser.add_argument(
        "--fixture-id",
        action="append",
        dest="fixture_ids",
        help="Run only the given fixture id. May be supplied multiple times.",
    )
    parser.add_argument(
        "--thread-count",
        action="append",
        type=int,
        dest="thread_counts",
        help="Numba thread count to benchmark. May be supplied multiple times.",
    )
    parser.add_argument("--repeat", type=int, default=10, help="Steady-state repetitions per case.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "benchmarks" / "runs"),
        help="Directory under which run outputs are created.",
    )
    parser.add_argument("--notes", default=None, help="Optional free-form note for environment metadata.")
    return parser.parse_args()


def _load_fixtures(manifest_path: Path, requested_ids: set[str] | None) -> list[Fixture]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = [Fixture(**item) for item in payload["fixtures"]]
    if requested_ids is None:
        return fixtures
    selected = [fixture for fixture in fixtures if fixture.fixture_id in requested_ids]
    missing = requested_ids.difference({fixture.fixture_id for fixture in selected})
    if missing:
        raise ValueError(f"Unknown fixture ids: {sorted(missing)}")
    return selected


def _dtype(name: str) -> np.dtype[np.float64] | np.dtype[np.float32]:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported dtype: {name}")


def _balanced_labels(rng: np.random.Generator, n_samples: int, n_classes: int) -> np.ndarray:
    repeats = int(math.ceil(n_samples / n_classes))
    labels = np.tile(np.arange(n_classes, dtype=np.int64), repeats)[:n_samples]
    rng.shuffle(labels)
    return labels


def _fixture_data(fixture: Fixture) -> dict[str, Any]:
    rng = np.random.default_rng(fixture.seed)
    dtype = _dtype(fixture.dtype)

    if fixture.kernel == "copnorm_slice":
        return {"x": rng.random((fixture.npage, fixture.ntrl), dtype=dtype)}

    if fixture.kernel == "info_cc_slice":
        return {
            "x": rng.standard_normal((fixture.npage, fixture.xdim, fixture.ntrl), dtype=dtype),
            "y": rng.standard_normal((fixture.ydim, fixture.ntrl), dtype=dtype),
        }

    if fixture.kernel == "info_cc_multi":
        return {
            "x": rng.standard_normal((fixture.npage, fixture.xdim, fixture.ntrl), dtype=dtype),
            "y": rng.standard_normal((fixture.npage, fixture.ydim, fixture.ntrl), dtype=dtype),
        }

    if fixture.kernel == "info_cc_slice_indexed":
        return {
            "x": rng.standard_normal((fixture.npage, fixture.xdim, fixture.ntrl), dtype=dtype),
            "x_idx": rng.integers(0, fixture.npage, size=fixture.npage, dtype=np.int64),
            "y": rng.standard_normal((fixture.ydim, fixture.ntrl), dtype=dtype),
        }

    if fixture.kernel == "info_c1d_slice":
        return {
            "x": rng.standard_normal((fixture.npage, fixture.ntrl), dtype=dtype),
            "y": _balanced_labels(rng, fixture.ntrl, int(fixture.ym)),
        }

    if fixture.kernel == "info_cd_slice":
        return {
            "x": rng.standard_normal((fixture.npage, fixture.xdim, fixture.ntrl), dtype=dtype),
            "y": _balanced_labels(rng, fixture.ntrl, int(fixture.ym)),
        }

    if fixture.kernel == "info_dc_slice_bc":
        labels = np.empty((fixture.npage, fixture.ntrl), dtype=np.int64)
        for page in range(fixture.npage):
            labels[page] = _balanced_labels(rng, fixture.ntrl, int(fixture.xm))
        return {
            "x": labels,
            "y": rng.standard_normal((fixture.ydim, fixture.ntrl), dtype=dtype),
        }

    raise ValueError(f"Unsupported kernel: {fixture.kernel}")


def _kernel_call(fixture: Fixture, data: dict[str, Any], backend: str):
    if fixture.kernel == "copnorm_slice":
        return gcmi.copnorm_slice(data["x"], backend=backend)
    if fixture.kernel == "info_cc_slice":
        return gcmi.info_cc_slice(data["x"], data["y"], backend=backend)
    if fixture.kernel == "info_cc_multi":
        return gcmi.info_cc_multi(data["x"], data["y"], backend=backend)
    if fixture.kernel == "info_cc_slice_indexed":
        return gcmi.info_cc_slice_indexed(data["x"], data["x_idx"], data["y"], backend=backend)
    if fixture.kernel == "info_c1d_slice":
        return gcmi.info_c1d_slice(data["x"], data["y"], int(fixture.ym), backend=backend)
    if fixture.kernel == "info_cd_slice":
        return gcmi.info_cd_slice(data["x"], data["y"], int(fixture.ym), backend=backend)
    if fixture.kernel == "info_dc_slice_bc":
        return gcmi.info_dc_slice(data["x"], data["y"], int(fixture.xm), biascorrect=True, backend=backend)
    raise ValueError(f"Unsupported kernel: {fixture.kernel}")


def _measure(fixture: Fixture, data: dict[str, Any], implementation: str, thread_count: int, repeat: int) -> dict[str, Any] | None:
    if implementation == "numba":
        assert numba is not None
        numba.set_num_threads(thread_count)

    start = time.perf_counter()
    _kernel_call(fixture, data, implementation)
    first_call_s = time.perf_counter() - start

    times_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        _kernel_call(fixture, data, implementation)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    median_ms = statistics.median(times_ms)
    compile_ms = None
    if implementation == "numba":
        compile_ms = max(first_call_s * 1000.0 - median_ms, 0.0)

    return {
        "kernel": fixture.kernel,
        "language": "python",
        "implementation": implementation,
        "dtype": fixture.dtype,
        "thread_count": thread_count,
        "ntrl": fixture.ntrl,
        "npage": fixture.npage,
        "xdim": fixture.xdim,
        "ydim": fixture.ydim,
        "ym": fixture.ym,
        "xm": fixture.xm,
        "compile_time_ms": compile_ms,
        "steady_state_time_ms": median_ms,
        "p10_time_ms": float(np.percentile(times_ms, 10)),
        "p90_time_ms": float(np.percentile(times_ms, 90)),
        "slices_per_second": fixture.npage / (median_ms / 1000.0),
        "speedup_vs_reference": None,
        "speedup_vs_1thread": None,
        "scaling_efficiency": None,
        "fixture_id": fixture.fixture_id,
        "run_id": "",
        "git_revision": "",
        "notes": None,
    }


def _git_revision() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:  # pragma: no cover - environment specific
        return None


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            try:
                return subprocess.check_output(
                    ["sysctl", "-n", "hw.model"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                pass
    return platform.processor() or platform.machine()


def _physical_cores() -> int:
    if sys.platform == "darwin":
        try:
            return int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except Exception:
            pass
    return os.cpu_count() or 1


def _blas_vendor() -> str | None:
    config = getattr(np.__config__, "CONFIG", None)
    if isinstance(config, dict):
        if "Accelerate" in json.dumps(config):
            return "Accelerate"
        if "OpenBLAS" in json.dumps(config):
            return "OpenBLAS"
    return None


def _environment(run_id: str, notes: str | None) -> dict[str, Any]:
    logical = os.cpu_count() or 1
    data = {
        "run_id": run_id,
        "language": "python",
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "physical_cores": _physical_cores(),
        "logical_cores": logical,
        "os_version": platform.version(),
        "matlab_release": None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numba_version": getattr(numba, "__version__", None) if numba is not None else None,
        "llvmlite_version": None,
        "compiler": None,
        "compiler_version": None,
        "openmp_runtime": None,
        "blas_vendor": _blas_vendor(),
        "git_revision": _git_revision(),
        "notes": notes,
    }
    if numba is not None:
        try:
            import llvmlite  # type: ignore

            data["llvmlite_version"] = getattr(llvmlite, "__version__", None)
        except ImportError:
            data["llvmlite_version"] = None
        try:
            data["openmp_runtime"] = numba.threading_layer()
        except ValueError:
            data["openmp_runtime"] = None
    return data


def _thread_counts(requested: list[int] | None) -> list[int]:
    logical = os.cpu_count() or 1
    defaults = [1, 2, 4, logical]
    values = requested or defaults
    filtered = sorted({count for count in values if 1 <= count <= logical})
    return filtered or [1]


def _attach_relative_metrics(records: list[dict[str, Any]]) -> None:
    by_fixture_impl: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_fixture_reference: dict[str, dict[str, Any]] = {}
    by_fixture_numba_1t: dict[str, dict[str, Any]] = {}

    for record in records:
        key = (record["fixture_id"], record["implementation"], record["thread_count"])
        by_fixture_impl[key] = record
        if record["implementation"] == "reference":
            by_fixture_reference[record["fixture_id"]] = record
        if record["implementation"] == "numba" and record["thread_count"] == 1:
            by_fixture_numba_1t[record["fixture_id"]] = record

    for record in records:
        ref = by_fixture_reference.get(record["fixture_id"])
        if ref is not None:
            record["speedup_vs_reference"] = ref["steady_state_time_ms"] / record["steady_state_time_ms"]
        one_thread = by_fixture_numba_1t.get(record["fixture_id"])
        if record["implementation"] == "numba" and one_thread is not None:
            record["speedup_vs_1thread"] = one_thread["steady_state_time_ms"] / record["steady_state_time_ms"]
            record["scaling_efficiency"] = record["speedup_vs_1thread"] / record["thread_count"]


def main() -> int:
    args = _parse_args()
    fixtures = _load_fixtures(
        ROOT / "benchmarks" / "fixtures_manifest.json",
        set(args.fixture_ids) if args.fixture_ids else None,
    )
    thread_counts = _thread_counts(args.thread_counts)
    git_revision = _git_revision() or "nogit"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-python-{platform.machine()}-{git_revision}"

    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        data = _fixture_data(fixture)
        reference_record = _measure(fixture, data, "reference", 1, args.repeat)
        if reference_record is not None:
            reference_record["run_id"] = run_id
            reference_record["git_revision"] = git_revision
            results.append(reference_record)

        if numba is None:
            continue

        for thread_count in thread_counts:
            record = _measure(fixture, data, "numba", thread_count, args.repeat)
            if record is None:
                continue
            record["run_id"] = run_id
            record["git_revision"] = git_revision
            results.append(record)

    _attach_relative_metrics(results)
    environment = _environment(run_id, args.notes)

    with (output_dir / "environment.json").open("w", encoding="utf-8") as fp:
        json.dump(environment, fp, indent=2, sort_keys=True)
        fp.write("\n")

    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as fp:
        for record in results:
            fp.write(json.dumps(record, sort_keys=True))
            fp.write("\n")

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
