# MATLAB C++ Rewrite Spec

## Goal

Design a replacement for the current templated Fortran MATLAB acceleration layer using modern C++, while preserving the current optimized algorithm shape and OpenMP slice-level parallelism.

This path is intended to reduce compiler and packaging pain relative to Fortran, especially on new MATLAB releases and on Windows.

## Scope

- Replace the current `extern/gcmi_mex` accelerated MATLAB layer with a C++ MEX implementation.
- Preserve kernel semantics and performance-critical structure where practical.
- Preserve OpenMP as a first-class feature.
- Use MATLAB-provided BLAS/LAPACK libraries where those match the current kernels well.

## Non-Goals

- Sharing a backend with Python in phase 1.
- Rewriting the scalar MATLAB reference functions.
- Replacing the pure MATLAB API.

## Recommendation

Recommended implementation style:

- C++17 or C++20
- C/C++ MEX build via `mex`
- direct BLAS/LAPACK calls to MATLAB-provided `mwblas` and `mwlapack` for the continuous-continuous kernels
- manual one-pass accumulation loops for continuous-discrete and discrete-continuous kernels
- OpenMP over pages/slices

This is effectively a C++ port of the current optimized kernel design, not a high-level linear-algebra rewrite.

## Why This Instead of Armadillo

Armadillo is usable, but not preferred here because:

- it adds another abstraction and dependency layer
- it encourages a more generic dense linear algebra style than these kernels really need
- it does not directly solve MATLAB packaging/runtime concerns
- your existing kernels are already organized around BLAS/LAPACK plus explicit one-pass accumulation

If the objective is a low-risk port with very similar optimizations, direct C++ plus BLAS/LAPACK is the cleanest fit.

## Why This Instead of Eigen

Eigen remains a good option in general, but for this MATLAB-only rewrite the direct BLAS/LAPACK path is the best first recommendation because:

- the current fast `cc` kernels already map directly to `dsyrk`, `dgemm`, `dpotrf`
- MATLAB already ships `mwblas` and `mwlapack`
- preserving the current kernel shape is easier with direct calls
- performance comparisons against the current Fortran are more straightforward

Eigen can still be considered for helper code, workspace management, or a later simplification pass.

## MATLAB Linking Strategy

### BLAS and LAPACK

Yes, the intended design is to link against MATLAB-provided BLAS/LAPACK:

- `mwblas`
- `mwlapack`

MathWorks documents that MEX files can call BLAS/LAPACK and link against these libraries.

### OpenMP

OpenMP remains required, but this part is more nuanced than BLAS/LAPACK.

MathWorks documents that customer-written OpenMP MEX files are not specifically tested or supported, but also provides guidance on making them work in practice.

Design implication:

- OpenMP should be treated as a supported project feature
- runtime selection and compiler compatibility must be validated per platform and MATLAB release
- release builds should be benchmarked and smoke-tested inside MATLAB

## Wrapper API Choice

Two viable wrapper choices exist:

1. modern C++ MEX/Data API
2. classic C MEX array API used from C++

Recommendation:

- prototype with the classic MEX array model first if it keeps the port closer to the current Fortran memory model and lowers wrapper overhead
- only move to the modern C++ Data API if it clearly improves code quality without hurting low-level control

Reason:

- this rewrite is performance-first and gateway wrappers should stay thin

## Kernels To Port

Phase 1 kernel set:

- `copnorm_slice`
- `info_cc_slice`
- `info_cc_multi`
- `info_cc_slice_indexed`
- `info_c1d_slice`
- `info_cd_slice`
- `info_dc_slice`
- `info_dc_slice_bc`

These correspond to the current accelerated MATLAB kernels in `extern/gcmi_mex`.

## Kernel Design

### 1. `copnorm_slice`

Current behavior to preserve:

- sort values per slice
- compute inverse-rank transform on the open interval
- apply inverse normal transform
- legacy tie handling: preserve current speed-oriented ordering semantics unless intentionally changed

Parallelization:

- OpenMP over pages/slices

Implementation notes:

- likely no BLAS/LAPACK needed
- use explicit per-slice work buffers for indices and outputs

### 2. `info_cc_slice`

Current behavior to preserve:

- `Y` is shared across slices
- unconditional covariance of `Y` computed once
- each `X` page gets its own covariance and cross-covariance with `Y`
- MI from Cholesky log-determinants

BLAS/LAPACK mapping:

- covariance of `Y`: `dsyrk`
- covariance of `X`: `dsyrk`
- cross-covariance `X^T Y`: `dgemm`
- Cholesky: `dpotrf`

Parallelization:

- OpenMP over slices
- keep BLAS single-threaded inside the OpenMP loop where needed to avoid nested oversubscription

### 3. `info_cc_multi`

Current behavior to preserve:

- both `X` and `Y` are paged
- each slice computes its own `X`, `Y`, and joint covariance

BLAS/LAPACK mapping:

- same as `info_cc_slice`, but no shared `Y` precompute

### 4. `info_cc_slice_indexed`

Current behavior to preserve:

- same logic as `info_cc_slice`
- pages of `X` selected by an index vector

Implementation note:

- preserve current index semantics
- do not force physical layout unification unless benchmarks show it is free

### 5. `info_c1d_slice`

Current behavior to preserve:

- one-pass accumulation of classwise scalar sums and sums of squares
- 1D special case with `0.5 * log(var)`

Parallelization:

- OpenMP over slices

Implementation note:

- no BLAS/LAPACK needed in the hot path

### 6. `info_cd_slice`

Current behavior to preserve:

- one-pass accumulation:
  - `Sx`
  - `Sxx`
  - `Sxg`
  - `Sxxg`
  - `Ntrl_g`
- conversion to covariance matrices
- weighted conditional entropy subtraction

Implementation note:

- keep this as explicit loops, not as a forced generic linear algebra call
- this is one of the most important kernels to preserve faithfully

### 7. `info_dc_slice`

Current behavior to preserve:

- unconditional entropy of continuous `Y` computed once
- per-slice classwise accumulation on `Y`
- conditional covariance matrices per class

BLAS/LAPACK mapping:

- unconditional covariance can use `dsyrk` + `dpotrf`
- conditional path should remain explicit accumulation unless benchmarking proves otherwise

### 8. `info_dc_slice_bc`

Current behavior to preserve:

- same as `info_dc_slice`
- analytic bias correction in-kernel
- digamma-based correction terms

Implementation note:

- retain in-kernel bias correction because it differs by slice/class structure and is not as cleanly separable as the shared-bias `cc` case

## Physical Layout Policy

Do not unify kernel-specific physical layouts unless benchmarks show it is free.

Examples from the current implementation that may remain intentionally different:

- `info_cc_slice`: layout equivalent to `[Ntrl, xdim, Npage]`
- `info_cd_slice`: layout equivalent to `[xdim, Ntrl, Npage]`

The rewrite should preserve or rediscover the fastest layout per kernel.

## Build System

Recommended build flow:

- top-level MATLAB `buildtool`
- MEX compilation via `mex`
- compiler and flag selection centralized in one configuration helper
- release packaging per MATLAB release and platform

Optional:

- CMake for local developer builds if it reduces complexity

But the canonical release path should remain `mex`-based because that aligns directly with MATLAB packaging.

## Compiler Strategy

### Windows

Primary objective:

- supported free/open C++ compiler path with OpenMP

Fallback:

- Intel oneAPI or another supported toolchain if OpenMP/runtime compatibility requires it

### Linux x86-64

Primary objective:

- GNU toolchain with OpenMP

### macOS Intel

Primary objective:

- supported C++ compiler path for the targeted MATLAB release

### macOS Apple Silicon

Primary objective:

- native `maca64` build with supported C++ compiler path

### Linux ARM

Only in scope if native Linux ARM MATLAB support exists for the targeted release.

## OpenMP Runtime Policy

OpenMP is required, but release engineering must explicitly validate runtime behavior.

Rules:

- benchmark 1-thread and multithread variants on every release build
- record compiler and OpenMP runtime in benchmark metadata
- avoid nested BLAS/OpenMP oversubscription
- rebuild per MATLAB release

## Validation Plan

### Numerical Parity

Validate against the current Fortran MEX outputs for:

- deterministic fixtures
- edge-shape cases
- multivariate cases
- indexed-page cases
- bias-corrected cases

### Tolerances

- `float64`: `rtol=1e-12`, `atol=1e-12`
- `float32`: `rtol=1e-5`, `atol=1e-6`

### Semantic Decisions

Recommended canonical semantics:

- zero-based discrete labels internally and in fixtures
- boundary adapters for legacy MATLAB compatibility only if needed
- no-bias-correction fast kernels remain no-bias-correction
- preserve current tie handling until intentionally changed

## Benchmark Plan

Use the shared benchmark contract in:

- `benchmarks/README.md`
- `benchmarks/results_schema.json`
- `benchmarks/environment_schema.json`
- `benchmarks/fixtures_manifest.json`

Report:

- wall time
- slices/sec
- speedup vs pure MATLAB reference
- speedup vs current Fortran MEX
- speedup vs 1-thread C++ MEX
- scaling efficiency

Thread counts:

- `1`
- `2`
- `4`
- `8`
- `max physical cores`

Benchmark cases:

- `copnorm_slice`
- `info_cc_slice`
- `info_cc_multi`
- `info_cc_slice_indexed`
- `info_c1d_slice`
- `info_cd_slice`
- `info_dc_slice_bc`

## Deliverables

1. One prototype C++ MEX kernel for `info_cc_slice`
2. One prototype C++ MEX kernel for `info_cd_slice`
3. Decision point:
   - if performance and build quality are good, port the remaining kernels
4. Standardized tests and benchmarks
5. Packaged binaries per MATLAB release and platform

## Decision Gate

Before full migration, prototype and compare:

- current Fortran MEX
- C++ + direct BLAS/LAPACK + OpenMP prototype

If the prototype preserves most of the current speed and materially improves build/portability, continue with the full rewrite.

## Sources

- MathWorks MEX overview:
  - https://www.mathworks.com/help/matlab/matlab_external/choosing-mex-applications.html
- Calling BLAS/LAPACK from MEX:
  - https://www.mathworks.com/help/matlab/matlab_external/calling-lapack-and-blas-functions-from-mex-files.html
- `mex` command:
  - https://www.mathworks.com/help/matlab/ref/mex.html
- OpenMP in MEX guidance:
  - https://www.mathworks.com/matlabcentral/answers/237411-can-i-make-use-of-openmp-in-my-matlab-mex-files
