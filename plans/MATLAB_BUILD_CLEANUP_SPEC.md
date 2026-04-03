# MATLAB Build Cleanup Spec

## Goal

Keep the current accelerated MATLAB kernels and OpenMP behavior, but modernize the build, test, benchmark, and release flow so that builds are reproducible and CI-driven.

This is a build cleanup and packaging project, not a numerical rewrite in phase 1.

## Scope

- Preserve the current accelerated kernels in `extern/gcmi_mex`.
- Preserve OpenMP as a first-class feature.
- Replace or remove the fragile template/build plumbing.
- Add standard validation and benchmark suites.
- Produce release artifacts per MATLAB release and platform.

## Target Platforms

The requested support matrix is:

- macOS Intel
- macOS Apple Silicon
- Windows x86-64
- Linux x86-64
- Linux ARM

Platform note:

- As of March 21, 2026, native Linux ARM is not a normal desktop MATLAB platform. Treat this as `best effort / blocked by MathWorks platform support` unless MathWorks publishes native Linux ARM MATLAB support for the target release.

## Non-Goals

- Rewriting the numerical kernels in C++ in phase 1.
- Changing the mathematical behavior of the existing MEX functions.
- Unifying MATLAB and Python onto one backend.

## Current Constraints

The current accelerated layer has three main maintainability problems:

1. `PyF95++` is obscure and wired into local-machine assumptions.
2. `make.m` contains hardcoded paths and manual platform handling.
3. Build/test/benchmark steps are not standardized for CI and release packaging.

The numerical kernels themselves are already useful and should be preserved while the infrastructure is cleaned up.

## MatlabAPI_lite Handling

`MatlabAPI_lite` should be maintained in its own repository and consumed from `gcmi` as a submodule at `extern/MatlabAPI_lite`.

### Recommendation

Treat `MatlabAPI_lite` as its own maintained project and consume it from `gcmi` as a submodule.

Use a two-tier workflow:

1. `MatlabAPI_lite` is cleaned up and migrated from `PyF95++` to `fypp` in its own repository.
2. `gcmi` consumes a pinned `MatlabAPI_lite` revision as a submodule under `extern/MatlabAPI_lite`.
3. Normal `gcmi` builds compile against the checked-out `MatlabAPI_lite` generated sources and do not require any template engine.
4. Maintainer workflows regenerate `MatlabAPI_lite` and `gcmi_mex` generated sources from `fypp`.

### Why

`MatlabAPI_lite` uses `PyF95++` much more heavily than `gcmi_mex` itself:

- the main wrapper module in `MatlabAPImx.F90T` generates type- and rank-specialized pointer helpers across many MATLAB numeric types
- the library test suite also relies on generated test sources

However, the key generated outputs are already checked in:

- `MatlabAPImx.F90`
- generated test `.F90` files

That means `gcmi` does not need template generation at build time if the generated sources are treated as source-of-build artifacts.

### Repo Structure Decision

Preferred approach:

- maintain `MatlabAPI_lite` in its own repository
- consume it in `gcmi` as a submodule at `extern/MatlabAPI_lite`
- build it automatically as part of the top-level MATLAB build once the submodule is present

Why this is now preferred:

- `MatlabAPI_lite` is itself under active cleanup and should retain its own history and issue tracking
- the `PyF95++ -> fypp` migration should be developed in one canonical place
- `gcmi` can pin a known-good commit while still allowing coordinated development across both repositories

Submodule policy:

- CI must initialize submodules explicitly
- release builds must use a pinned submodule revision
- end users should not need any manual extra step beyond normal clone instructions that include submodules

### Practical Build Rule

For `gcmi`:

- compile `extern/MatlabAPI_lite/MatlabAPImx.F90`
- compile `extern/MatlabAPI_lite/MatlabAPImex.f`
- link those objects into the `gcmi_mex` MEX builds
- do not require `PyF95++` or `fypp` for ordinary users or CI release builds

For `MatlabAPI_lite` maintenance:

- migrate `MatlabAPImx.F90T` and test templates to `fypp`
- keep template generation as a maintainer-only task
- regenerate committed outputs when the wrapper library changes

### Bundling Recommendation

Bundle the `MatlabAPI_lite` build inside the top-level MATLAB build pipeline once the submodule is checked out.

Do not require users to build `MatlabAPI_lite` separately.

The top-level MATLAB build should:

1. optionally run template generation in maintainer mode
2. compile `MatlabAPI_lite` objects
3. compile `gcmi_mex` objects
4. link MEX binaries
5. run tests
6. run benchmarks

This keeps the dependency graph explicit and makes CI much easier.

## Recommended Build Strategy

### Phase 1: Stabilize Current Sources

- Check in all generated Fortran sources required to build the current system.
- Make the normal build independent of the template engine.
- Verify that a clean checkout can build from committed sources alone.

This gives an immediate escape hatch from `PyF95++`.

### Phase 2: Replace Template Generation With `fypp`

- Replace `PyF95++` with `fypp` for all generated Fortran sources in both `gcmi_mex` and `MatlabAPI_lite`.
- Keep generated output paths deterministic.
- Make template generation an explicit build step, not an implicit local workflow.

Reasoning:

- `fypp` is a normal source generator and is much easier to automate in CI.
- It can generate the typed and repeated procedure families used here, including the `calcinfo` families and the heavily templated `MatlabAPI_lite` pointer helpers.
- The earlier suggestion to migrate `copnorm` first was only a risk-reduction sequencing choice, not a technical limit.

## Compiler Support Goals

The build cleanup should explicitly target modern supported compilers rather than preserving the historical Intel-only assumption.

### Windows

Primary target:

- MinGW-w64 toolchain support for top-level MEX builds and linked Fortran objects where supported by current MATLAB releases

Secondary target:

- Intel oneAPI / `ifx`

Notes:

- MathWorks documents MinGW-w64 support for MATLAB on Windows and, starting in recent releases, also documents that Fortran library or object files built with MinGW-w64 can be linked to MEX functions built with MinGW-w64.
- Intel oneAPI remains the fallback path when MinGW-based Fortran compatibility proves insufficient for a particular release or feature.

### Linux x86-64

Primary target:

- GNU toolchain / `gfortran`

Secondary target:

- Intel oneAPI if needed for compatibility or performance

### macOS Intel

Preferred target:

- GNU-based Fortran path where supported by the chosen MATLAB release and tested in CI

Fallback:

- release-specific supported compiler path if GNU support is insufficient

### macOS Apple Silicon

Known constraint:

- official MATLAB compiler support may not line up perfectly with the practical compiler path you use locally

Plan:

- support Apple Silicon as a first-class platform
- test `gfortran` first as the preferred practical path
- retain release-specific officially supported compiler paths as fallback if MATLAB compatibility requires them
- isolate any exception cases in CI and release documentation

### Linux ARM

Status:

- blocked unless MathWorks provides a native Linux ARM MATLAB target for the chosen release

## Compiler Compatibility Policy

- Free or open compiler paths should be the default objective on Windows and Linux.
- Apple Silicon may remain an exception due to MATLAB compiler support constraints.
- CI should continuously validate which matrix entries actually work for the supported MATLAB releases.
- The build system should make it easy to fall back to Intel oneAPI when needed without changing sources.

### Phase 3: Standardize the MATLAB Build Entry Point

- Replace ad hoc use of `make.m` with MATLAB `buildtool` tasks in `buildfile.m`.
- Define tasks:
  - `generate`
  - `compile`
  - `test`
  - `bench`
  - `package`
- Keep platform-specific compiler/link flags in one config helper.

### Phase 4: CI and Release Packaging

- Use GitHub Actions for hosted Windows and Linux builds.
- Use a self-hosted Apple Silicon runner for native `maca64` builds.
- Add Intel Mac support either through a self-hosted Intel runner or a dedicated legacy runner strategy if still needed.
- Publish per-platform release bundles.

## Template Migration Strategy

### Recommendation

Move every templated source to `fypp` if you decide to keep templating at all.

That should include:

- typed `copnorm` entry points
- typed support modules like `fcinfo`
- any repeated MEX-wrapper boilerplate that is currently easier to stamp out than hand-maintain

The hand-written `info_*` kernels do not all need to become templates if they are already clear and stable, but `fypp` is capable of handling them if you want to unify generation style.

### Migration Rule

Use templating only where it removes real duplication:

- data kind variants
- wrapper boilerplate
- small repeated families with stable structure

Do not template highly custom numerical kernels just to maximize generator usage if it makes the code harder to debug.

## Proposed Layout

- `extern/gcmi_mex/src/`
- `extern/gcmi_mex/templates/`
- `extern/gcmi_mex/generated/`
- `extern/gcmi_mex/tests/`
- `extern/gcmi_mex/bench/`
- `extern/gcmi_mex/buildfile.m`
- `extern/gcmi_mex/tooling/`

Suggested contents:

- `src/`: hand-written stable Fortran sources
- `templates/`: `fypp` sources
- `generated/`: build outputs from templates
- `tooling/`: generation helpers, config, packaging helpers

## OpenMP Policy

OpenMP is required.

The design assumption is:

- MATLAB performance over independent slices is a core feature.
- The MEX path should preserve current page-wise parallel scaling.
- Benchmarking must treat multi-thread speedup as a release quality metric, not an optional optimization.

Implementation guidance:

- Keep OpenMP at the outer slice/page loop level.
- Maintain a 1-thread mode for debugging and parity testing.
- Test thread scaling on all supported platforms.

## Build Artifacts

Package release artifacts by:

- MATLAB release
- platform
- architecture

Suggested naming:

- `gcmi_mex_<matlab-release>_win64.zip`
- `gcmi_mex_<matlab-release>_glnxa64.zip`
- `gcmi_mex_<matlab-release>_maci64.zip`
- `gcmi_mex_<matlab-release>_maca64.zip`

Each bundle should contain:

- compiled MEX files
- version manifest
- MATLAB release used to build
- compiler info
- thread/OpenMP notes
- short install instructions
- test and benchmark metadata

## Validation Test Plan

### Numerical Parity

Add test coverage for:

- `copnorm_slice_omp`
- `info_cc_slice_nobc_omp`
- `info_cc_multi_nobc_omp`
- `info_cc_slice_indexed_nobc_omp`
- `info_c1d_slice_nobc_omp`
- `info_cd_slice_nobc_omp`
- `info_dc_slice_bc_omp`

Reference baselines:

- current known-good MEX outputs
- pure MATLAB reference implementations where applicable

### Shape and Convention Coverage

Explicitly test:

- current axis conventions for each kernel
- 1-based `int16` class labels used by the MEX layer
- mismatched size errors
- singleton dimensions
- multi-page inputs

### Tolerances

- `float64`: `rtol=1e-12`, `atol=1e-12`
- `float32`: `rtol=1e-5`, `atol=1e-6`

## Standard Benchmark Suite

The benchmark suite should be standardized so that it can later feed a technical blog post.

### Reported Metrics

Always report:

- wall time
- slices per second
- speedup vs pure MATLAB reference
- speedup vs 1-thread MEX
- scaling efficiency: `speedup / threads`

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

- warm up once
- run at least 10 repetitions
- report median plus p10 and p90
- capture CPU model, OS, MATLAB release, compiler, and thread count

### Output Artifacts

- machine-readable CSV or JSON
- plots for throughput and scaling
- environment metadata for publication

## CI Specification

### GitHub Actions Jobs

- `matlab-build-windows`
- `matlab-build-linux`
- `matlab-test-windows`
- `matlab-test-linux`
- `matlab-bench-linux`

Self-hosted:

- `matlab-build-maca64`
- `matlab-test-maca64`
- `matlab-bench-maca64`

Optional:

- `matlab-build-maci64`
- `matlab-test-maci64`

### CI Outputs

- build logs
- unit test results
- benchmark result files
- packaged release bundles

## Deliverables

1. Reproducible build from clean checkout.
2. `buildfile.m` with standardized tasks.
3. `fypp`-based generation or committed generated sources.
4. GitHub Actions build/test workflows.
5. Cross-platform benchmark harness with standardized outputs.
6. Release bundles per MATLAB release and platform.

## Decision Notes

- If replacing `MatlabAPI_lite` is low-risk, that should be considered later.
- It is not required for this cleanup spec.
- The first objective is professionalizing the build and benchmark process around the existing kernels.
