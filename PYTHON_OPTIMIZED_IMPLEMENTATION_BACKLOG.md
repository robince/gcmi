# Python Optimized Implementation Backlog

## Purpose

This backlog turns the Python optimization spec into concrete work items.

The strategy assumed here is:

- MATLAB keeps its own native accelerated path.
- Python gets an independently optimized implementation.
- The first Python optimization target is Numba.
- The public MATLAB and Python APIs do not need to share a binary backend.

This document also captures the numerical details of the current accelerated MATLAB implementation so the Python work can explicitly reproduce the same optimized algorithm shape where it matters.

It should also use `frites` as a comparison baseline where practical, especially for tensorized `cc` and `cd` workloads.

## Current Accelerated MATLAB Context To Reproduce

The current accelerated MATLAB layer in `extern/gcmi_mex` is a narrow batch-kernel MEX implementation for permutation-style workloads.

### Kernels

Priority kernels and current sources:

- `copnorm_slice_omp`
  - [`extern/gcmi_mex/copnorm_slice_omp.F90T`](/Users/robince/code/gcmi/extern/gcmi_mex/copnorm_slice_omp.F90T)
  - typed copula-normalization over pages
- `info_cc_slice_nobc_omp`
  - [`extern/gcmi_mex/info_cc_slice_nobc_omp.f`](/Users/robince/code/gcmi/extern/gcmi_mex/info_cc_slice_nobc_omp.f)
  - many continuous `X` pages against one shared continuous `Y`
- `info_cc_multi_nobc_omp`
  - [`extern/gcmi_mex/info_cc_multi_nobc_omp.f`](/Users/robince/code/gcmi/extern/gcmi_mex/info_cc_multi_nobc_omp.f)
  - many continuous `X` pages against matching continuous `Y` pages
- `info_cc_slice_indexed_nobc_omp`
  - [`extern/gcmi_mex/info_cc_slice_indexed_nobc_omp.f`](/Users/robince/code/gcmi/extern/gcmi_mex/info_cc_slice_indexed_nobc_omp.f)
  - many continuous `X` pages selected by an index vector against one shared continuous `Y`
- `info_c1d_slice_nobc_omp`
  - [`extern/gcmi_mex/info_c1d_slice_nobc_omp.f`](/Users/robince/code/gcmi/extern/gcmi_mex/info_c1d_slice_nobc_omp.f)
  - optimized special case for 1D continuous-discrete MI
- `info_cd_slice_nobc_omp`
  - [`extern/gcmi_mex/info_cd_slice_nobc_omp.f`](/Users/robince/code/gcmi/extern/gcmi_mex/info_cd_slice_nobc_omp.f)
  - multivariate continuous-discrete MI with one-pass conditional covariance accumulation
- `info_dc_slice_nobc_omp`
  - [`extern/gcmi_mex/info_dc_slice_nobc_omp.f`](/Users/robince/code/gcmi/extern/gcmi_mex/info_dc_slice_nobc_omp.f)
  - discrete-continuous MI without analytic bias correction
- `info_dc_slice_bc_omp`
  - [`extern/gcmi_mex/info_dc_slice_bc_omp.f`](/Users/robince/code/gcmi/extern/gcmi_mex/info_dc_slice_bc_omp.f)
  - discrete-continuous MI with analytic bias correction

### Algorithmic Shape

The current accelerated kernels rely on two patterns:

1. Parallelism over independent slices/pages.
2. Small dense linear algebra or explicit one-pass covariance accumulation within each slice.

This shape should be preserved in Python before considering alternative formulations.

### Continuous-Continuous Kernels

The current MATLAB MEX implementation computes Gaussian MI by:

- computing covariance of `Y` once when shared
- computing covariance of each `X` slice
- computing cross-covariance between `X` and `Y`
- assembling the joint covariance
- taking Cholesky factors
- summing `log(diag(chol(...)))`
- converting to MI in bits via `(HX + HY - HXY) / log(2)`

BLAS/LAPACK routines used:

- `dsyrk` for covariance accumulation
- `dgemm` for cross-covariance
- `dpotrf` for Cholesky

Other current behavior:

- the implementation explicitly sets MKL thread count to `1` inside each OpenMP slice loop to avoid nested threading
- the outer page loop is OpenMP-parallel

### Continuous-Discrete Kernels

The current MATLAB MEX implementation does not call BLAS in the hot class-conditional loop.

Instead it uses one-pass accumulation:

- `Sx`: unconditional sum
- `Sxx`: unconditional sum of outer products
- `Sxg`: per-class sums
- `Sxxg`: per-class sums of outer products
- `Ntrl_g`: per-class counts

Then it converts those sums to covariance matrices using:

- unconditional covariance:
  - `alpha = -1 / Ntrl`
  - `alpha1 = 1 / (Ntrl - 1)`
  - `Cov = alpha1 * (Sxx + alpha * Sx * Sx^T)`
- conditional covariance for each class with the same formula using class counts and class sums

Then it computes:

- unconditional entropy from unconditional covariance
- conditional entropies from class covariances
- weighted conditional sum
- MI in bits as `(Hunc - dot(w, Hcond)) / log(2)`

For `info_c1d_slice_nobc_omp`, the 1D special case uses scalar variances and `0.5 * log(var)` instead of full matrix Cholesky.

### Discrete-Continuous Kernels

The current MATLAB MEX implementation computes unconditional Gaussian entropy of `Y` once, then for each discrete `X` slice:

- accumulates classwise sums `Sy`
- accumulates classwise second moments `Syy`
- converts to classwise covariance matrices
- computes conditional Cholesky factors and entropies
- weights them by class probabilities

The bias-corrected variant adds analytic digamma-based bias correction terms per class and for the unconditional entropy.

BLAS/LAPACK used in the unconditional path:

- `dsyrk`
- `dpotrf`

### Copula Normalization

The current accelerated copula path:

- sorts values
- builds rank order by index
- maps ranks to open-interval empirical CDF positions using `i / (Ntrl + 1)`
- applies an inverse normal transform

Current tie handling:

- duplicates are ignored for speed
- whatever ordering comes out of sort is used

That semantic choice must be made explicit in the Python optimized path.

### Important Behavioral Details

The Python implementation should preserve or explicitly document:

- tie handling in `copnorm`
- sample-axis conventions for the optimized kernels
- class label conventions
- whether bias correction is applied inside the kernel or externally
- whether `Y`-side shared entropy/covariance is precomputed once or per slice

### Physical Layout Policy

Do not unify physical array layouts across kernels unless benchmarks show there is no measurable cost.

The Python optimized implementation may choose different contiguous internal layouts for different kernels if that reproduces the performance logic of the current MATLAB accelerated implementation more faithfully.

## Recommended Python Implementation Order

### Phase 1: Infrastructure

1. Create `python/gcmi_ref.py`
   - isolate the current pure reference implementations
2. Create `python/gcmi_numba.py`
   - house optimized kernels
3. Create `python/gcmi_dispatch.py`
   - runtime dispatch between reference and optimized paths
4. Create `python/tests/`
5. Create `python/benchmarks/`

Deliverable:

- clean separation between reference and optimized code paths

### Phase 2: Benchmark Harness First

1. Implement a benchmark driver before optimizing kernels.
2. Standardize inputs and outputs so MATLAB and Python benchmarks can be compared directly.
3. Capture machine and package metadata with each run.
4. Keep benchmark semantics canonical while leaving kernel-specific physical layout choices free.
5. Include `frites` as an optional comparison baseline where the corresponding estimator is available.

Deliverable:

- benchmark harness that can time reference code immediately

### Phase 3: Implement Numba Kernels In Priority Order

#### Task Group A: `info_c1d_slice`

1. Implement 1D continuous-discrete kernel in Numba.
2. Preserve one-pass class accumulation.
3. Use `prange` over slices.
4. Validate against Python reference and current MATLAB outputs.

Reason:

- smallest and most controlled kernel
- best first test of Numba parallelization and reduction behavior

#### Task Group B: `info_cd_slice`

1. Implement multivariate continuous-discrete kernel in Numba.
2. Preserve explicit `Sx`, `Sxx`, `Sxg`, `Sxxg`, `Ntrl_g` accumulation.
3. Benchmark alternative small-matrix covariance/cholesky strategies.
4. Validate numerical parity.

Reason:

- central kernel
- strong fit for Numba explicit loops

#### Task Group C: `info_dc_slice_bc`

1. Implement discrete-continuous kernel without bias correction.
2. Implement bias-corrected variant.
3. Port digamma-based bias correction logic carefully.
4. Validate weighted conditional entropy path.

Reason:

- algorithmically similar to current Fortran
- good parity target with current MATLAB implementation

#### Task Group D: `info_cc_slice`

1. Implement shared-`Y` continuous-continuous kernel.
2. Precompute `Y` covariance or entropy once outside the slice loop.
3. Benchmark:
   - explicit covariance assembly + `np.linalg.cholesky`
   - alternate small-matrix manual approaches if needed
4. Validate parity.

#### Task Group E: `info_cc_multi`

1. Implement matching-page `X` and `Y` kernel.
2. Reuse `info_cc_slice` helpers where possible.
3. Validate and benchmark.

#### Task Group F: `info_cc_slice_indexed`

1. Implement indexed-page selection variant.
2. Preserve index semantics.
3. Validate against current MATLAB behavior.

#### Task Group G: `copnorm_slice`

1. Benchmark NumPy-based rank path as a baseline.
2. Implement Numba rank path.
3. Compare:
   - NumPy sort + NumPy inverse-normal
   - Numba sort/rank + inverse-normal
4. Decide whether `copnorm` remains in optimized Numba or stays NumPy-backed.

This kernel is intentionally last because it is the highest risk for marginal performance gain relative to implementation effort.

## Concrete Development Tasks

### Task 1: Build Reference Test Fixtures

- create deterministic synthetic datasets for all kernel families
- include small hand-checkable cases
- include realistic batch cases
- include cases with repeated values for `copnorm`

### Task 2: Implement Common Test Helpers

- array-shape normalizers
- tolerance assertions
- timing harness
- metadata capture helpers

### Task 3: Implement Numba Kernel Utilities

- covariance helpers
- entropy-from-Cholesky helpers
- class-count helpers
- workspace allocation helpers

### Task 4: Add Thread Control

- expose thread count control in the benchmark harness
- record Numba threading layer
- record actual thread count used

### Task 5: Add Dispatch and Fallback

- dispatch to optimized kernels only when supported
- fall back cleanly to reference code
- allow an override to force reference path

### Task 6: Add Cross-Language Parity Fixtures

- produce benchmark/test fixtures that can also be consumed from MATLAB
- store them in machine-readable form
- use them for article-quality comparisons later

## Benchmark Specification

The benchmark suite should mirror the MATLAB benchmark suite exactly where possible.

### Metrics

Always report:

- wall time
- slices per second
- speedup vs pure Python reference
- speedup vs 1-thread optimized path
- scaling efficiency
- first-call compile time
- steady-state warmed-call time

### Cases

#### Copula Normalization

- `Ntrl = 1000`
- `Npage = 1e3, 1e4`

#### Continuous-Continuous

- `Ntrl = 200, 1000, 5000`
- `xdim = 1, 2, 4`
- `ydim = 1, 2, 4`
- `Npage = 1e3, 1e4`

#### Continuous-Discrete 1D

- `Ntrl = 200, 1000, 5000`
- `Ym = 2, 4, 8`
- `Npage = 1e3, 1e4`

#### Continuous-Discrete Multivariate

- `Ntrl = 200, 1000, 5000`
- `xdim = 1, 2, 4, 8`
- `Ym = 2, 4, 8`
- `Npage = 1e3, 1e4`

#### Discrete-Continuous

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

### Reporting Protocol

- warm up once
- report compile time separately
- run at least 10 repetitions
- report median, p10, p90
- capture CPU model, OS, Python, NumPy, Numba, llvmlite versions

## Acceptance Criteria

### Correctness

- optimized kernels match reference implementation within agreed tolerances
- selected cases match current MATLAB accelerated outputs

### Performance

- multicore speedup is demonstrated on slice-parallel workloads
- benchmark outputs are reproducible and publication-ready

### Distribution

- no custom compiled Python extension is required in phase 1
- installation works with standard Python package tooling

## Risks To Track

1. `copnorm` may not be a strong Numba target.
2. small-matrix linear algebra may need benchmarking against NumPy-backed linalg calls.
3. Numba platform support may constrain Python version policy.
4. first-call JIT latency may matter for short-running workflows.

## Explicit Decisions

- Start with Numba.
- Preserve the optimized algorithm shape from MATLAB before attempting redesign.
- Treat `copnorm` as a benchmark-driven decision, not an assumed Numba success.
- Keep MATLAB and Python optimization paths independent.
