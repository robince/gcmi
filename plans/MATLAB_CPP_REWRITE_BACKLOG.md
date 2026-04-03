# MATLAB C++ Rewrite Backlog

## Purpose

This backlog turns the MATLAB C++ rewrite spec into concrete execution work.

The intended end state is:

- the current MATLAB accelerated Fortran MEX layer is replaced by a C++ MEX implementation
- the optimized algorithm shape is preserved where it still benchmarks best
- `mwblas` and `mwlapack` are used where they match the current fast kernels
- OpenMP remains a first-class feature for slice-level scaling
- the new implementation is validated against the current Fortran MEX outputs and the shared benchmark contract

## Scope

Phase 1 is a MATLAB-only native rewrite.

It does not attempt to share a backend with Python and does not rewrite the scalar reference MATLAB functions.

## Decision Summary

Architecture choice:

- C++17/20
- MEX built via MATLAB `mex`
- direct BLAS/LAPACK calls for the `cc` family
- explicit one-pass accumulation loops for `cd` and `dc`
- OpenMP over pages/slices

Wrapper choice:

- start with a thin classic MEX array wrapper from C++
- reconsider the modern C++ Data API only if it improves maintainability without hurting low-level control

## Workstreams

## Workstream 1: Project Skeleton

### Task 1.1: Create C++ Source Layout

Create a new source tree for the rewrite, for example:

- `extern/gcmi_cpp_mex/include/`
- `extern/gcmi_cpp_mex/src/`
- `extern/gcmi_cpp_mex/mex/`
- `extern/gcmi_cpp_mex/tests/`
- `extern/gcmi_cpp_mex/bench/`

Goal:

- separate kernel implementation from MATLAB gateway code

### Task 1.2: Define Shared Internal Interfaces

Define internal kernel interfaces for:

- `copnorm_slice`
- `info_cc_slice`
- `info_cc_multi`
- `info_cc_slice_indexed`
- `info_c1d_slice`
- `info_cd_slice`
- `info_dc_slice`
- `info_dc_slice_bc`

Goal:

- the MATLAB gateway layer stays thin
- kernels are testable separately where practical

## Workstream 2: Build and Linking Prototype

### Task 2.1: Minimal MEX Build

Create a minimal proof-of-build C++ MEX target.

Requirements:

- compiles through `mex`
- returns a simple output
- uses the selected gateway style

### Task 2.2: BLAS/LAPACK Link Prototype

Build a prototype MEX that:

- links against `mwblas`
- links against `mwlapack`
- performs one small covariance or Cholesky operation

Goal:

- validate the direct BLAS/LAPACK toolchain early

### Task 2.3: OpenMP Link Prototype

Build a prototype MEX that:

- runs a simple OpenMP slice loop
- accepts a thread count argument
- returns deterministic output

Goal:

- validate OpenMP runtime behavior inside MATLAB before porting kernels

### Task 2.4: Combined Runtime Prototype

Build a prototype that combines:

- MATLAB MEX gateway
- BLAS/LAPACK call
- OpenMP outer loop

Goal:

- validate the exact runtime combination the final kernels will use

## Workstream 3: Prototype Gate

### Task 3.1: Port `info_cc_slice` First

This is the first real kernel to port.

Required behavior:

- `Y` shared across slices
- `dsyrk` for covariance
- `dgemm` for cross-covariance
- `dpotrf` for Cholesky
- OpenMP over slices
- BLAS effectively single-threaded inside the OpenMP loop

Success criteria:

- numerical parity with current Fortran MEX
- successful multicore scaling
- no runtime instability in MATLAB

### Task 3.2: Port `info_cd_slice` First-Class Prototype

After `info_cc_slice`, port `info_cd_slice`.

Required behavior:

- preserve one-pass accumulation of:
  - `Sx`
  - `Sxx`
  - `Sxg`
  - `Sxxg`
  - `Ntrl_g`
- preserve current covariance conversion formulas
- OpenMP over slices

Success criteria:

- numerical parity with current Fortran MEX
- speed that remains competitive with the existing kernel

### Decision Gate

Do not port the whole kernel family before these two prototypes are benchmarked.

Continue only if:

- build complexity is materially lower than the Fortran toolchain
- `info_cc_slice` remains competitive
- `info_cd_slice` remains competitive
- OpenMP behavior is stable across targeted MATLAB/compiler combinations

## Workstream 4: Port Remaining Kernels

### Task 4.1: `info_cc_multi`

Port with:

- paged `X`
- paged `Y`
- same BLAS/LAPACK structure as `info_cc_slice`

### Task 4.2: `info_cc_slice_indexed`

Port with:

- indexed page lookup
- preserved index semantics

### Task 4.3: `info_c1d_slice`

Port with:

- scalar one-pass class accumulation
- `0.5 * log(var)` path

### Task 4.4: `info_dc_slice`

Port with:

- unconditional continuous entropy of `Y`
- explicit classwise accumulation
- conditional covariance and entropy path

### Task 4.5: `info_dc_slice_bc`

Port with:

- same as `info_dc_slice`
- in-kernel analytic bias correction

### Task 4.6: `copnorm_slice`

Port with:

- explicit sort/index/rank logic
- inverse-normal transform
- preserved legacy tie behavior unless intentionally changed

This kernel should not be first in the port order. It should follow after the MEX/OpenMP toolchain is proven.

## Workstream 5: Physical Layout Validation

### Task 5.1: Preserve Existing Performance-Driven Layouts

Do not normalize physical layouts just for consistency.

Preserve candidate layouts from the current implementation:

- `info_cc_slice`: equivalent to `[Ntrl, xdim, Npage]`
- `info_cd_slice`: equivalent to `[xdim, Ntrl, Npage]`

### Task 5.2: Re-benchmark Before Simplifying

Only simplify layout conventions if a benchmark shows no material regression.

## Workstream 6: MATLAB Gateway Layer

### Task 6.1: Input Validation

Implement thin gateway checks for:

- dimensionality
- class-label conventions
- shape mismatches
- supported dtypes

### Task 6.2: Output Allocation

Keep output allocation minimal and explicit.

### Task 6.3: Compatibility Layer for Labels

Recommended internal standard:

- zero-based labels

Compatibility option:

- if needed, accept legacy MATLAB-facing one-based labels at the boundary and normalize immediately

## Workstream 7: Numerical Validation

### Task 7.1: Freeze Fortran MEX Baselines

Generate deterministic baseline outputs from the current Fortran MEX implementation for:

- small fixtures
- medium fixtures
- edge cases
- multiclass cases
- indexed-page cases

### Task 7.2: Compare Against Shared Golden Fixtures

Use the shared benchmark/fixture contract under:

- `benchmarks/README.md`
- `benchmarks/fixtures_manifest.json`

### Task 7.3: Tolerance Rules

- `float64`: `rtol=1e-12`, `atol=1e-12`
- `float32`: `rtol=1e-5`, `atol=1e-6`

## Workstream 8: Benchmark Harness

### Task 8.1: MATLAB C++ Benchmark Driver

Create a benchmark harness for the C++ rewrite that emits:

- `results.jsonl`
- `environment.json`

matching the shared schemas in:

- `benchmarks/results_schema.json`
- `benchmarks/environment_schema.json`

### Task 8.2: Required Cases

Benchmark:

- `copnorm_slice`
- `info_cc_slice`
- `info_cc_multi`
- `info_cc_slice_indexed`
- `info_c1d_slice`
- `info_cd_slice`
- `info_dc_slice_bc`

Dimensions:

- `Ntrl = 200, 1000, 5000`
- `Npage = 1e3, 1e4`
- `xdim = 1, 2, 4, 8`
- `ydim = 1, 2, 4`
- `Ym = 2, 4, 8`
- `Xm = 2, 4, 8`

Thread counts:

- `1`
- `2`
- `4`
- `8`
- `max physical cores`

### Task 8.3: Comparison Set

Compare against:

- MATLAB reference implementation
- current Fortran MEX
- C++ rewrite, 1 thread
- C++ rewrite, multithreaded

Report:

- wall time
- slices/sec
- speedup vs MATLAB reference
- speedup vs Fortran MEX
- speedup vs 1-thread C++ MEX
- scaling efficiency

## Workstream 9: Build and Release Automation

### Task 9.1: Top-Level Build Integration

Integrate the C++ rewrite into a MATLAB `buildtool` flow.

Tasks:

- `compile`
- `test`
- `bench`
- `package`

### Task 9.2: Compiler Matrix Validation

Validate the C++ rewrite against the target compiler matrix:

- Windows
- Linux x86-64
- macOS Intel
- macOS Apple Silicon

### Task 9.3: Release Packaging

Package binaries per:

- MATLAB release
- OS
- architecture

Include:

- MEX binaries
- build manifest
- compiler info
- benchmark metadata
- install notes

## Workstream 10: Runtime Validation

### Task 10.1: BLAS/LAPACK Runtime

Validate:

- successful linking to `mwblas` and `mwlapack`
- correct behavior across target MATLAB releases

### Task 10.2: OpenMP Runtime

Validate:

- stable MEX loading
- stable execution under repeated calls
- no obvious runtime conflicts
- expected thread scaling

### Task 10.3: Nested Threading Control

Ensure:

- BLAS does not oversubscribe threads inside OpenMP loops
- benchmark metadata records the effective runtime configuration

## Suggested Execution Order

1. Build minimal C++ MEX proof-of-build.
2. Validate BLAS/LAPACK linking.
3. Validate OpenMP linking and runtime behavior.
4. Port `info_cc_slice`.
5. Port `info_cd_slice`.
6. Benchmark and decide whether to continue the full migration.
7. Port the remaining kernels.
8. Add full benchmark harness and release packaging.

## Acceptance Criteria

### Build

- reproducible C++ MEX builds on target platforms
- materially simpler developer setup than the Fortran path

### Correctness

- parity with current Fortran MEX outputs within agreed tolerances

### Performance

- preserved or competitive speed on core workloads
- preserved multicore slice-parallel scaling

### Maintainability

- simpler compiler/toolchain story than the current Fortran path
- clearer top-level build and release process
