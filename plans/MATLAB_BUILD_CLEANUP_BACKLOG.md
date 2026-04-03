# MATLAB Build Cleanup Backlog

## Purpose

This backlog turns the MATLAB build cleanup spec into concrete work items.

The intended end state is:

- `MatlabAPI_lite` is maintained in its own repo and consumed here as a submodule
- `PyF95++` is fully replaced by `fypp`
- the full MATLAB accelerated stack builds reproducibly from one top-level build
- modern supported compilers are exercised in CI, with free/open compiler paths preferred on Windows and Linux
- release artifacts are produced per MATLAB release and platform

## Current State Summary

The current MATLAB acceleration stack has two components:

1. `extern/gcmi_mex`
   - accelerated GCMI kernels
   - OpenMP-parallel page/slice kernels
   - a small amount of templated Fortran for typed `copnorm`
   - hand-written high-value kernels such as:
     - `info_cc_slice_nobc_omp`
     - `info_cc_multi_nobc_omp`
     - `info_cc_slice_indexed_nobc_omp`
     - `info_c1d_slice_nobc_omp`
     - `info_cd_slice_nobc_omp`
     - `info_dc_slice_nobc_omp`
     - `info_dc_slice_bc_omp`

2. `extern/MatlabAPI_lite`
   - Fortran pointer and allocation wrapper layer for MATLAB arrays
   - heavily templated via `PyF95++`
   - generated sources already checked in

The current pain points are:

- `PyF95++` and Python 2 era assumptions
- hardcoded local paths in MATLAB `make.m` scripts
- no unified top-level build
- no standard CI matrix
- no standard benchmark harness or release packaging

## Build Philosophy

The top-level MATLAB build should own the whole dependency chain.

That means:

- do not require users to install or build `MatlabAPI_lite` separately once submodules are checked out
- do not require users to run template generation manually
- do not require users to edit compiler paths in ad hoc scripts

Layout policy:

- do not normalize kernel-specific physical array layouts unless benchmarks show that doing so is free
- preserve current layout choices that appear to have been made for performance until they are explicitly re-benchmarked

## Target Platforms

Required platform matrix:

- Windows x86-64
- Linux x86-64
- macOS Intel
- macOS Apple Silicon
- Linux ARM only if MATLAB platform support exists for the chosen release

Compiler objective:

- prefer free/open compiler paths on Windows and Linux
- keep Intel oneAPI as a supported fallback
- test `gfortran` first on Apple Silicon and retain release-specific supported fallbacks if needed

## Compiler Context

As of March 21, 2026, the relevant MATLAB compiler milestones are:

- Windows MinGW-w64 support for MATLAB / MEX and documented MinGW-w64-linked Fortran object support in recent releases
- Intel `ifx` support added in recent MATLAB releases
- Apple Silicon support requires validating the practical `gfortran` path against MATLAB release behavior rather than relying only on old assumptions

This backlog assumes the build should actively validate those claims in CI rather than trust historical assumptions.

## Workstreams

## Workstream 1: Repository and Source Layout

### Task 1.1: Lock Repository Structure

Decision:

- keep `MatlabAPI_lite` in its own repository
- consume it here as a submodule at `extern/MatlabAPI_lite`

Deliverables:

- submodule policy documented in the build spec
- CI checkout initializes submodules
- `gcmi` pins a known-good `MatlabAPI_lite` revision

### Task 1.2: Normalize Source Layout

Create a standard structure under both `extern/gcmi_mex` and `extern/MatlabAPI_lite`:

- `templates/`
- `generated/`
- `src/`
- `tests/`
- `bench/`
- `tooling/`

Goal:

- make generated vs hand-written sources explicit

## Workstream 2: Replace `PyF95++` With `fypp`

### Task 2.1: Inventory All Template Inputs and Outputs

Inventory:

- `extern/gcmi_mex/copnorm.F90T`
- `extern/gcmi_mex/copnorm_slice_omp.F90T`
- `extern/gcmi_mex/fcinfo.F90T`
- `extern/gcmi_mex/instantiate.F90T`
- `extern/MatlabAPI_lite/MatlabAPImx.F90T`
- `extern/MatlabAPI_lite/tests/test_mx.F90T`
- `extern/MatlabAPI_lite/tests/instantiate.F90T`

Deliverable:

- mapping table from old template source to new `fypp` source and generated outputs

### Task 2.2: Migrate `gcmi_mex` Templates

Convert the `gcmi_mex` templates to `fypp`.

Required generated families:

- `fcinfo_*`
- `copnorm_*`
- `copnorm_slice_omp_*`
- instantiation helpers if still needed

Goal:

- identical generated procedure/module names to the current build where possible

### Task 2.3: Migrate `MatlabAPI_lite` Templates

Convert the `MatlabAPI_lite` templates to `fypp`.

Required generated families:

- typed and rank-specialized `fpGetPr`
- related pointer helpers
- test template families

Goal:

- preserve current behavior and test coverage
- remove dependence on `PyF95++`

### Task 2.4: Commit Generated Outputs

Even after moving to `fypp`, commit the generated `.F90` outputs used by normal builds.

Goal:

- keep ordinary builds and CI release jobs independent of template generation

### Task 2.5: Add Maintainer Regeneration Workflow

Add a maintainer-only task:

- `generate`

This task should:

- regenerate all `fypp` outputs
- fail if committed generated files are stale

## Workstream 3: Unify the MATLAB Build

### Task 3.1: Replace Ad Hoc `make.m` Flow

Create a top-level `buildfile.m` for the MATLAB acceleration stack.

Required tasks:

- `generate`
- `compile-matlabapi`
- `compile-gcmi`
- `test`
- `bench`
- `package`

### Task 3.2: Bundle `MatlabAPI_lite` Into the Top-Level Build

The build should:

1. compile `extern/MatlabAPI_lite/MatlabAPImx.F90`
2. compile `extern/MatlabAPI_lite/MatlabAPImex.f`
3. compile `gcmi_mex` sources
4. link the MEX binaries

Goal:

- no separate user action for `MatlabAPI_lite`
- one coordinated build even though `MatlabAPI_lite` is sourced from its own repo

### Task 3.3: Centralize Build Configuration

Move all platform/compiler settings into one helper:

- compiler selection
- flags
- OpenMP flags
- include paths
- object paths
- output paths

Goal:

- no machine-specific edits in build scripts

## Workstream 4: Compiler Matrix Modernization

### Task 4.1: Windows Free-Compiler Path

Primary target:

- MinGW-w64-based build path

Tasks:

- determine exact supported MATLAB release/compiler pair
- validate object compilation for `MatlabAPI_lite`
- validate object compilation for `gcmi_mex`
- validate MEX linking with MinGW-w64
- validate OpenMP behavior and runtime compatibility inside MATLAB

Fallback:

- Intel oneAPI / `ifx`

Success criteria:

- Windows builds do not require Intel as the only supported path

### Task 4.2: Linux GNU Path

Primary target:

- GNU Fortran / GNU toolchain

Tasks:

- validate builds on GitHub-hosted Linux
- validate OpenMP behavior
- validate MEX loading and benchmark scaling

Fallback:

- Intel oneAPI when required

### Task 4.3: macOS Intel Path

Tasks:

- validate supported compiler path for chosen MATLAB release
- validate MEX builds
- validate OpenMP support or document limitations

### Task 4.4: macOS Apple Silicon Path

Tasks:

- validate native `maca64` build path
- validate `gfortran` first as the preferred practical path
- validate Fortran compiler support for chosen MATLAB release
- use another supported compiler path only if `gfortran` proves incompatible for the targeted release
- validate benchmark scaling

### Task 4.5: Linux ARM Status Check

Tasks:

- confirm whether the target MATLAB release supports native Linux ARM
- if unsupported, explicitly mark Linux ARM as blocked in build docs and release matrix

## Workstream 5: OpenMP Validation

OpenMP is a core feature and must be treated as a required capability.

### Task 5.1: Preserve Current Parallel Structure

Preserve the current design:

- outer loop parallel over slices/pages
- inner BLAS or covariance operations remain single-threaded where appropriate
- avoid nested parallelism

### Task 5.2: Validate Runtime Compatibility

For each supported platform/compiler combination:

- build and load MEX files in MATLAB
- validate no runtime conflicts
- validate thread count control
- validate 1-thread and multithread parity

### Task 5.3: Expose Thread Count Cleanly

Standardize thread count handling in:

- tests
- benchmarks
- examples

## Workstream 6: Numerical Validation

### Task 6.1: Freeze Numerical Baselines

Create a baseline result set for:

- `copnorm_slice_omp`
- `info_cc_slice_nobc_omp`
- `info_cc_multi_nobc_omp`
- `info_cc_slice_indexed_nobc_omp`
- `info_c1d_slice_nobc_omp`
- `info_cd_slice_nobc_omp`
- `info_dc_slice_nobc_omp`
- `info_dc_slice_bc_omp`

### Task 6.2: Add Unit Tests

Required coverage:

- shape conventions
- label conventions
- error handling
- scalar/small examples
- multi-page examples
- data type coverage for wrapped pointer helpers in `MatlabAPI_lite`

### Task 6.3: Add Cross-Version Safety Tests

Where practical:

- test more than one MATLAB release
- verify that generated MEX files are rebuilt per release

## Workstream 7: Benchmark Harness

### Task 7.1: Create Standard Benchmark Driver

The benchmark harness must emit:

- machine-readable CSV or JSON
- plots or plot-ready tables
- environment metadata

### Task 7.2: Implement Benchmark Cases

Required cases:

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

### Task 7.3: Standardize Reporting

Report:

- wall time
- slices/sec
- speedup vs MATLAB reference implementation
- speedup vs 1-thread MEX
- scaling efficiency

Method:

- warm up once
- 10 or more repetitions
- median, p10, p90

## Workstream 8: CI

### Task 8.1: Build Workflows

Create workflows for:

- Windows hosted build
- Linux hosted build
- Apple Silicon self-hosted build
- optional Intel Mac self-hosted build

Checkout policy:

- all workflows must initialize submodules
- the pinned `MatlabAPI_lite` revision is part of the tested build definition

### Task 8.2: Test Workflows

Run:

- unit tests
- selected numerical parity tests
- selected smoke benchmarks

### Task 8.3: Release Packaging

Publish release bundles by:

- MATLAB release
- OS
- architecture

Include:

- compiled MEX files
- manifest
- compiler info
- benchmark metadata
- install instructions

## Workstream 9: Documentation

### Task 9.1: Developer Build Documentation

Document:

- normal build
- maintainer regeneration
- compiler matrix
- known platform exceptions

### Task 9.2: User Install Documentation

Document:

- how to install prebuilt MEX bundles
- supported MATLAB releases
- supported platforms
- thread/OpenMP usage

### Task 9.3: Performance Reporting Documentation

Document:

- benchmark methodology
- hardware metadata captured
- release-to-release comparability rules

## Suggested Execution Order

1. Establish `MatlabAPI_lite` as a submodule-backed dependency in `gcmi`.
2. Freeze and verify current generated sources.
3. Add top-level build pipeline that works without any template engine in ordinary builds.
4. Bundle `MatlabAPI_lite` into that top-level build.
5. Add tests and benchmark harness around the frozen build.
6. Migrate `MatlabAPI_lite` templates from `PyF95++` to `fypp` in its own repo.
7. Migrate `gcmi_mex` templates from `PyF95++` to `fypp`.
8. Turn on maintainer regeneration checks.
9. Validate compiler matrix, starting with Windows MinGW, Linux GNU, and Apple Silicon `gfortran`.
10. Add release packaging and CI artifact publishing.

## Acceptance Criteria

### Build

- one top-level MATLAB build produces all required MEX outputs
- no ordinary build depends on `PyF95++`
- no ordinary build depends on manual path edits

### Compiler Support

- Windows no longer depends exclusively on Intel compilers
- Linux GNU toolchain path is validated
- Apple Silicon path is documented and tested

### Correctness

- current numerical outputs are preserved within agreed tolerances

### Performance

- OpenMP scaling is preserved and reported in a standard benchmark suite

### Maintainability

- template generation is handled by `fypp`
- generated sources are committed
- CI catches stale generated files and broken platform builds
