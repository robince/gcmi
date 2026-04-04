# Benchmarks

This directory defines the shared benchmark and fixture contract for the MATLAB and Python optimization efforts.

The purpose is to make sure:

- both implementations target the same logical workloads
- benchmark results are directly comparable
- the eventual technical blog post can be built from one standardized result format

## Canonical Semantics

The benchmark suite uses canonical logical semantics that are independent of language-specific memory layout.

### Sample Axis

- MATLAB-facing arrays may continue to use samples on the first axis where that is the natural storage/layout choice.
- Python-facing arrays may continue to use samples on the last axis where that is the natural storage/layout choice.

That difference is intentional and should remain.

### What Should Not Differ

The following should be canonical across implementations:

- kernel meaning
- label encoding semantics
- benchmark case names
- fixture generation rules
- expected outputs and result schema

### Batch Terminology

The benchmark contract uses the following logical terms:

- `n_var`: number of independent MI computations evaluated in batch
- `n_mv`: multivariate feature dimension within one variable
- `n_samples`: number of trials / observations

This matches the batching vocabulary used by `frites`, where arrays are often treated conceptually as `(n_var, n_mv, n_samples)`.

For this project:

- `n_var` is usually equivalent to the number of slices/pages to evaluate
- `n_mv` is the within-slice multivariate dimensionality

### Physical Layout Policy

Physical array layout does not need to be unified across kernels or across languages.

Rule:

- keep canonical logical semantics at the benchmark and fixture level
- allow each implementation to choose the in-memory layout that benchmarks best
- do not normalize physical layout across kernels just for API neatness

This explicitly allows layout choices such as:

- `info_cc_slice` using a layout equivalent to `[Ntrl, xdim, Npage]`
- `info_cd_slice` using a layout equivalent to `[xdim, Ntrl, Npage]`

if those choices are materially beneficial for performance.

### Label Semantics

Canonical logical label convention:

- discrete labels are `0 .. M-1`

Recommendation:

- treat `0 .. M-1` as the internal and benchmark-level standard for both MATLAB and Python
- if the legacy MATLAB MEX layer still wants `1 .. M`, handle that as a thin adapter or compatibility shim at the boundary

This removes a known design wart without forcing the benchmark contract to inherit it.

### Current Convention Warts To Isolate

These are current implementation quirks, not benchmark-level semantics:

- `info_cd_slice_nobc_omp` expects `X` as `[xdim, Ntrl, Npage]`
- `info_cc_slice_nobc_omp` expects `X` as `[Ntrl, xdim, Npage]`
- `info_dc_slice_*` expects discrete inputs as 1-based `int16`
- `copnorm` currently ignores ties for speed

The benchmark and fixture contract should record these explicitly, but new implementations should prefer one canonical logical interface internally.

## Success Criteria

### MATLAB

- preserve current kernel behavior
- preserve OpenMP scaling over slices/pages
- preserve absolute speed to within an agreed tolerance where compiler changes allow

### Python

- materially outperform the pure reference path
- demonstrate multicore slice-parallel speedup
- use the same logical workloads and reporting format as MATLAB

## Required Benchmark Outputs

Each benchmark run must produce:

- `results.jsonl`
- `environment.json`
- optional `plots/`

The required schemas are:

- [`results_schema.json`](results_schema.json)
- [`environment_schema.json`](environment_schema.json)

## Fixture Policy

Benchmark fixtures are defined by:

- kernel
- dimensions
- dtype
- seed
- label convention
- tie policy where relevant

The shared fixture manifest lives in:

- [`fixtures_manifest.json`](fixtures_manifest.json)

## Required Metrics

Every benchmark record must report:

- `kernel`
- `implementation`
- `language`
- `dtype`
- `thread_count`
- `ntrl`
- `npage`
- `xdim`
- `ydim`
- `ym`
- `xm`
- `compile_time_ms` when applicable
- `steady_state_time_ms`
- `slices_per_second`
- `speedup_vs_reference`
- `speedup_vs_1thread`
- `scaling_efficiency`

## Comparison Baselines

Where available, benchmark reports should compare against:

- current reference implementation
- optimized implementation under test
- `frites` tensorized GCMI implementation for the corresponding kernel family

Purpose:

- `frites` provides a useful batched NumPy/SciPy baseline
- it should be treated as a comparison target, not as the required implementation strategy

## Benchmark Method

- warm up once
- separate compile time from steady-state time
- run at least 10 repetitions for steady-state timing
- report median plus p10 and p90
- record machine metadata in `environment.json`

## Expected Output Policy

There are two kinds of expected outputs.

### Correctness Outputs

These are golden outputs for deterministic fixtures used in parity tests.

Rules:

- generated from the current trusted implementation
- versioned
- stored in machine-readable format
- compared with explicit tolerances

### Performance Outputs

These are not fixed numbers, because they vary by machine and compiler.

Instead, performance expectations are threshold-based:

- required minimum speedup vs reference
- required multicore scaling behavior
- required absence of regressions vs previous baseline on the same machine class

## Initial Performance Thresholds

These are starting values and should be refined after baseline collection.

- MATLAB optimized vs MATLAB reference:
  - target `>= 20x` on large slice-batched workloads
- MATLAB multithreaded vs MATLAB 1-thread:
  - target scaling efficiency `>= 0.6` at 4 threads
- Python optimized vs Python reference:
  - target `>= 5x` on large slice-batched workloads
- Python multithreaded vs Python 1-thread:
  - target scaling efficiency `>= 0.5` at 4 threads

These are governance thresholds, not publication claims.

## Result Naming

Recommended output layout:

- `benchmarks/runs/<run_id>/results.jsonl`
- `benchmarks/runs/<run_id>/environment.json`
- `benchmarks/runs/<run_id>/plots/`

Where `run_id` contains:

- date
- implementation
- platform
- compiler
- short git revision

## Next Step

Implementation work in MATLAB and Python should use this directory as the reporting contract and should not invent separate benchmark schemas.

## MATLAB Runner

The MATLAB side benchmark runner is:

- [`run_matlab_benchmarks.m`](run_matlab_benchmarks.m)

Run it from MATLAB with the repository root as the current directory, or add the `benchmarks/` directory to the path first.

Example using the current Fortran/OpenMP MEX entrypoints:

```matlab
addpath('benchmarks');
run_matlab_benchmarks('FixtureIds', {'copnorm_medium_f64', 'cc_small_f64'}, ...
    'ThreadCounts', [1 2 4 8], ...
    'Repeat', 10);
```

This writes:

- `benchmarks/runs/<run_id>/environment.json`
- `benchmarks/runs/<run_id>/results.jsonl`

The runner automatically adapts the canonical fixture semantics to the current legacy MEX boundary quirks, including:

- converting benchmark-level `0 .. M-1` labels to the current MEX layer's `1 .. M`
- applying `biasterms_cc` or `biasterms_cd` for `_nobc_` kernels so results stay comparable to the bias-corrected MATLAB reference path
- permuting arrays into the kernel-specific physical layouts required by the existing MEX functions

To benchmark a future C++ MEX implementation with different entrypoint names, pass a function-name map:

```matlab
addpath('benchmarks');
fmap = struct( ...
    'copnorm_slice', 'copnorm_slice_cpp', ...
    'info_cc_slice', 'info_cc_slice_cpp', ...
    'info_cc_multi', 'info_cc_multi_cpp', ...
    'info_cc_slice_indexed', 'info_cc_slice_indexed_cpp', ...
    'info_c1d_slice', 'info_c1d_slice_cpp', ...
    'info_cd_slice', 'info_cd_slice_cpp', ...
    'info_dc_slice_bc', 'info_dc_slice_bc_cpp');
run_matlab_benchmarks('OptimizedLabel', 'cpp_mex', ...
    'OptimizedFunctions', fmap, ...
    'OptimizedPaths', fullfile(pwd, 'matlab', 'cpp_mex', 'bin', version('-release'), mexext), ...
    'OptimizedLabelEncoding', 'zero_based', ...
    'ThreadCounts', [1 2 4 8], ...
    'Repeat', 10);
```

This keeps the fixture set and output schema unchanged, so the current MEX and a replacement C++ MEX can be compared directly.
