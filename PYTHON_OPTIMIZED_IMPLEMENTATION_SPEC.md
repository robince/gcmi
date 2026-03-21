# Python Optimized Implementation Spec

## Goal

Build a Python-only optimized implementation path that is easy to distribute and benchmark, without inheriting the MATLAB binary and OpenMP packaging constraints.

The recommended starting point is Numba.

## Scope

- Keep the existing pure Python implementation as the reference oracle.
- Add an optimized execution path for the batch kernels used in performance-sensitive workloads.
- Standardize correctness tests and multicore benchmarks so results are directly comparable to the MATLAB path.

## Non-Goals

- Sharing a native backend with MATLAB.
- Building Python around a custom C++ extension in phase 1.
- Rewriting the public API unless required for performance dispatch.

## Strategy

### Primary Path

Use Numba first.

Reasons:

- easiest distribution story
- no custom binary wheel build required for your package
- native threaded execution without the GIL
- good fit for explicit loops and one-pass accumulation kernels

### Escalation Path

Escalate only if benchmarks justify it:

1. `Numba`
2. `Pythran`
3. `Cython`

This keeps the implementation pragmatic and lets the benchmark results drive complexity.

## Kernel Priority

Implement in this order:

1. `info_c1d_slice`
2. `info_cd_slice`
3. `info_dc_slice_bc`
4. `info_cc_slice`
5. `info_cc_multi`
6. `info_cc_slice_indexed`
7. `copnorm_slice`

Reasoning:

- the one-pass accumulation kernels are a strong fit for Numba
- `copnorm` ranking may be the least attractive Numba target and should be benchmarked separately

## Public API Strategy

Keep the public API stable.

Suggested internal structure:

- `python/gcmi_ref.py`
- `python/gcmi_numba.py`
- `python/gcmi_dispatch.py`
- `python/tests/`
- `python/benchmarks/`

Dispatch rules:

- default to optimized kernels when available
- fall back to reference implementation when unsupported
- allow an environment variable or config flag to force reference mode

## Numba Design Rules

### Compilation Policy

- use `@njit(cache=True, nogil=True)` wherever possible
- use `parallel=True` only at the outer slice/page level
- avoid object mode entirely
- keep dtypes explicit and stable
- prefer contiguous arrays

### Algorithm Policy

- preserve the current optimized algorithm shape
- preserve one-pass accumulation where it matters
- preserve current semantics before attempting algorithmic changes

### Threading Policy

- prefer Numba native multithreading over Python multiprocessing
- use `prange` on independent slices/pages
- begin with the default built-in threading layer such as `workqueue`
- do not require Python-side OpenMP in phase 1

### Workspace Policy

- preallocate arrays where practical
- avoid repeated heap allocation in hot loops
- keep intermediate matrices small and local to the slice loop

## Numba Suitability Assessment

### Good Fit

Numba should be a good fit for:

- one-pass covariance accumulation
- bincount-style class accumulation
- per-slice outer loops
- explicit reductions
- small dense per-slice matrix assembly

### Possible but Needs Measurement

Numba may be acceptable but must be benchmarked for:

- sorting and ranking in `copnorm`
- repeated small Cholesky operations vs manual closed-form code
- very small matrix kernels where Python+NumPy may already be competitive

### Conclusion

Numba is good enough for cumulative or one-pass covariance style optimizations.

The main risk area is not accumulation. The main risk area is whether `copnorm` ranking is competitive enough in Numba relative to NumPy or a more specialized implementation.

## Package and Distribution Strategy

### Distribution Goal

Users should be able to install the optimized Python path without compiling your own extension module.

### Packaging Model

- your package remains Python source
- users install `numba` and `llvmlite` wheels from PyPI or conda
- kernels are JIT-compiled on first use

### Platform Targets

- Windows x86-64
- Linux x86-64
- Linux ARM64
- macOS Apple Silicon
- macOS Intel only if supported by the chosen Python and Numba version policy

### Practical Consequence

This avoids owning:

- custom OpenMP runtime bundling
- custom wheel repair
- custom compiler-toolchain support for your package

## Correctness Test Plan

### Baselines

Use:

- the existing Python reference implementation
- selected cross-checks against current MATLAB/MEX outputs

### Core Test Cases

Test all optimized kernels for:

- numerical parity on deterministic synthetic data
- randomized property tests
- edge shapes and singleton dimensions
- different class counts
- varying page counts

### Tolerances

- `float64`: `rtol=1e-12`, `atol=1e-12`
- `float32`: `rtol=1e-5`, `atol=1e-6`

### Additional Tests

- optimized path disabled vs enabled
- first-call compile path
- cached execution path
- thread-count control behavior

## Standard Benchmark Suite

The benchmark suite should mirror the MATLAB benchmark suite so results are publication-ready.

### Reported Metrics

Always report:

- wall time
- slices per second
- speedup vs pure Python reference
- speedup vs NumPy reference path
- speedup vs 1-thread optimized path
- scaling efficiency: `speedup / threads`

### Additional Python Metrics

- first-call JIT compile time
- warmed-call steady-state time
- optional multiprocessing overhead if tested

### Benchmark Cases

#### Copula Normalization

- `copnorm_slice`
- `Ntrl = 1000`
- `Npage = 1e3, 1e4`

#### Continuous-Continuous MI

- `info_cc_slice`
- `Ntrl = 200, 1000, 5000`
- `xdim = 1, 2, 4`
- `ydim = 1, 2, 4`
- `Npage = 1e3, 1e4`

#### Continuous-Discrete 1D

- `info_c1d_slice`
- `Ntrl = 200, 1000, 5000`
- `Ym = 2, 4, 8`
- `Npage = 1e3, 1e4`

#### Continuous-Discrete Multivariate

- `info_cd_slice`
- `Ntrl = 200, 1000, 5000`
- `xdim = 1, 2, 4, 8`
- `Ym = 2, 4, 8`
- `Npage = 1e3, 1e4`

#### Discrete-Continuous

- `info_dc_slice_bc`
- `Ntrl = 200, 1000, 5000`
- `ydim = 1, 2, 4`
- `Xm = 2, 4, 8`
- `Npage = 1e3, 1e4`

### Thread Counts

- `1`
- `2`
- `4`
- `8`
- `max physical cores`

### Benchmark Method

- warm up once for compilation
- separate compile time from steady-state timing
- run at least 10 repetitions for steady-state
- report median plus p10 and p90
- capture CPU model, OS, Python, NumPy, Numba, and llvmlite versions

### Output Artifacts

- machine-readable CSV or JSON
- plots for scaling and throughput
- environment metadata suitable for later publication

## Benchmark Comparison Matrix

Every benchmark case should compare:

- reference Python implementation
- optimized Python implementation, 1 thread
- optimized Python implementation, multiple threads
- MATLAB MEX 1 thread
- MATLAB MEX multiple threads

This is the key bridge to the eventual academic blog post.

## Escalation Criteria

Move beyond Numba only if one of the following happens:

1. `copnorm_slice` is materially too slow in Numba.
2. `info_cc_slice` or `info_cd_slice` fail to scale adequately with threads.
3. Numba support on one required platform becomes too weak for the package policy.
4. First-call JIT latency is unacceptable for real workflows.

If escalation is needed:

- try `Pythran` for array-oriented kernels
- try `Cython` for the smallest, most control-sensitive kernels

## Deliverables

1. Numba prototype module for the prioritized kernels.
2. Internal dispatch layer.
3. Correctness tests against the reference implementation.
4. Standardized multicore benchmark harness.
5. Benchmark outputs comparable to MATLAB results.

## Decision Notes

- Numba is the right first implementation target.
- It is strong enough for cumulative covariance and one-pass accumulation kernels.
- The main question is performance quality on ranking-heavy `copnorm`, not whether Numba can express the algorithm.
