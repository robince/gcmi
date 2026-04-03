# AGENTS.md

## Purpose

This repository contains a reference implementation of Gaussian-copula-based mutual information estimators in two languages:

- Python: an installable package in [`python/src/gcmi`](python/src/gcmi), plus a compatibility shim in [`python/gcmi.py`](python/gcmi.py)
- MATLAB: the primary reference implementation in [`matlab/`](matlab)


## Repository Layout

- [`README.md`](README.md): high-level project description and usage overview
- [`python/pyproject.toml`](python/pyproject.toml): Python packaging metadata for subdirectory installs
- [`python/src/gcmi`](python/src/gcmi): Python package source
- [`python/gcmi.py`](python/gcmi.py): backward-compatible source-tree shim
- [`python/tests`](python/tests): Python pytest suite
- [`matlab/*.m`](matlab): MATLAB estimators, vectorized variants, and helpers
- [`matlab/tests`](matlab/tests): MATLAB regression checks
- [`matlab_examples/*.m`](matlab_examples): tutorial/example scripts plus small example helpers
- [`setup_gcmi.m`](setup_gcmi.m): MATLAB path setup helper from the repository root

## Core Domain Conventions

- The package estimates mutual information (MI), conditional mutual information (CMI), and entropy using Gaussian assumptions after optional copula normalization.
- Function naming follows the pattern `quantity_variabletypes`, where:
  - `mi` = mutual information
  - `cmi` = conditional mutual information
  - `gcmi` / `gccmi` = Gaussian-copula estimator wrappers
  - `g` = Gaussian
  - `c` = continuous with arbitrary marginal distribution
  - `d` = discrete
- Discrete variables are expected to be encoded as contiguous classes `0..Ym-1` or `0..Zm-1`.
- Empty discrete classes are treated as invalid input and should fail fast.
- Continuous inputs with many repeated values are warned about because the rank-based copula transform becomes poorly defined.

## Critical Axis Convention

The languages deliberately differ:

| Language | Sample axis |
| --- | --- |
| MATLAB | first axis / rows |
| Python | last axis / columns |

Any changes, tests, or docs must preserve and repeatedly verify this distinction.

## Public API Surface

### Shared estimator families

Both implementations expose the same conceptual estimator groups:

- Copula preprocessing: `ctransform`, `copnorm`
- Gaussian quantities: `ent_g`, `mi_gg`, `cmi_ggg`
- Continuous/discrete Gaussian MI: `mi_model_gd`, `mi_mixture_gd`
- Gaussian-copula wrappers: `gcmi_cc`, `gcmi_model_cd`, `gcmi_mixture_cd`, `gccmi_ccc`, `gccmi_ccd`

### MATLAB-only extras

MATLAB also includes vectorized multi-signal routines:

- `mi_gg_vec`
- `mi_model_gd_vec`
- `mi_mixture_gd_vec`
- `cmi_ggg_vec`
- support helpers such as `vecchol` and `maxstar`

These vectorized routines are important when planning tests because they are additional behavior, not just performance wrappers.

## Current Technical State

- The Python package is installable from the `python/` subdirectory and intentionally keeps runtime dependencies minimal.
- There is automated coverage in [`python/tests`](python/tests) and [`matlab/tests`](matlab/tests), with CI expected to exercise them.
- The Python runtime depends on NumPy and SciPy, and uses `scipy.special` for core numerical functions such as `ndtri` and `psi`.
- MATLAB contains the broadest feature set, including vectorized helpers and tutorials.
- Example scripts are still tutorial workflows, not substitutes for regression tests.
- Version metadata is inconsistent with the README history and should be rationalized now that packaging is in place.

## Working Rules For Future Changes

- Preserve numerical behavior unless a change is explicitly intended and validated.
- Add tests before refactoring estimator internals or public signatures.
- Keep cross-language parity visible. If a function exists in both languages, document whether they are intended to be equivalent or intentionally diverge.
- Treat the MATLAB code as the fuller historical reference, but not automatically as bug-free.
- When fixing MATLAB vectorized code, test both scalar and multivariate dimensions because several helpers branch on dimensionality.
- When modernizing Python, split packaging/layout changes from estimator refactors when possible.
- Do not rely on example scripts as proof of correctness; use deterministic unit tests and cross-language comparison fixtures.

## Recommended Validation Strategy

When changing estimator code, cover at least:

- shape validation and sample-axis handling
- univariate and multivariate continuous MI
- continuous/discrete MI with all classes populated
- conditional MI with continuous and discrete conditioning
- repeated-value warnings
- numerical parity checks between Python and MATLAB on small synthetic inputs
- failure behavior for invalid inputs such as mismatched sample counts and missing classes

## Documentation Expectations

- Keep [`FUNCTIONALITY.md`](FUNCTIONALITY.md) aligned with the current implementation, not the aspirational future state.
- If an API inconsistency or bug is documented, call it out explicitly rather than normalizing it away in prose.
