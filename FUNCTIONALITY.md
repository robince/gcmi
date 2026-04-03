# FUNCTIONALITY.md

## Overview

This repository implements Gaussian-copula mutual information estimators in Python and MATLAB.

The core idea is:

1. Optionally rank-transform continuous variables to empirical CDF values.
2. Map those values to standard normal marginals with an inverse normal transform.
3. Compute entropy, MI, or CMI analytically under a Gaussian assumption.

Two important interface rules apply throughout:

| Language | Sample axis | Multivariate variable axis |
| --- | --- | --- |
| MATLAB | first axis | later axis / columns |
| Python | last axis | earlier axis / rows |

Discrete variables are encoded as integers `0..M-1`.

## Cross-Language Function Map

| Concept | Python | MATLAB | Notes |
| --- | --- | --- | --- |
| Empirical copula transform | `ctransform` | `ctransform` | Same role |
| Copula normalization | `copnorm` | `copnorm` | Same role |
| Gaussian entropy | `ent_g` | `ent_g` | Same role |
| Gaussian MI | `mi_gg` | `mi_gg` | Same role |
| Gaussian continuous-discrete MI (model comparison) | `mi_model_gd` | `mi_model_gd` | Same role |
| Gaussian continuous-discrete MI (mixture) | `mi_mixture_gd` | `mi_mixture_gd` | Same role |
| Gaussian CMI | `cmi_ggg` | `cmi_ggg` | Same role |
| Copula MI, continuous-continuous | `gcmi_cc` | `gcmi_cc` | Same role |
| Copula MI, continuous-discrete model | `gcmi_model_cd` | `gcmi_model_cd` | Same role |
| Copula MI, continuous-discrete mixture | `gcmi_mixture_cd` | `gcmi_mixture_cd` | Same role |
| Copula CMI, continuous-continuous-continuous | `gccmi_ccc` | `gccmi_ccc` | Same role |
| Copula CMI, continuous-continuous-discrete | `gccmi_ccd` | `gccmi_ccd` | Same role |

## Python Functionality

Primary package source: [`python/src/gcmi/_core.py`](python/src/gcmi/_core.py)

Compatibility shim: [`python/gcmi.py`](python/gcmi.py)

### Public functions

`ctransform(x)`

- Computes empirical CDF positions by ranking values along the last axis.
- Returns open-interval values `(rank / (n + 1))`.
- Used as the first step of the copula estimator.
- Accepts arrays that are coerced with `np.atleast_2d`.

`copnorm(x)`

- Applies the inverse standard normal CDF to `ctransform(x)`.
- Produces Gaussianized marginals while preserving rank order.
- Used by all copula-based wrappers.

`ent_g(x, biascorrect=True)`

- Returns the analytic entropy of a Gaussian variable in bits.
- Treats rows as variables and columns as samples.
- Demeans the data, forms the covariance matrix, and uses a Cholesky factor to compute log-determinant terms.
- Optional analytic bias correction is enabled by default.

`mi_gg(x, y, biascorrect=True, demeaned=False)`

- Returns Gaussian mutual information in bits between two variables.
- Supports multivariate `x` and `y`.
- Builds a joint covariance matrix and combines marginal and joint entropy terms.
- `demeaned=True` skips mean removal, which is useful after copula normalization.

`gcmi_cc(x, y)`

- Gaussian-copula MI between two continuous variables.
- Warns if either input has more than 10% repeated values along a variable axis.
- Applies `copnorm` to each input and then calls `mi_gg(..., biascorrect=True, demeaned=True)`.

`mi_model_gd(x, y, Ym, biascorrect=True, demeaned=False)`

- Gaussian MI between a continuous/Gaussian variable and a discrete variable.
- Uses an ANOVA-style model-comparison decomposition:
  - unconditional Gaussian entropy of `x`
  - minus weighted sum of class-conditional Gaussian entropies
- Requires `y` to be an integer dtype and currently assumes classes `0..Ym-1` exist.
- In the univariate case, the estimator is described as a lower bound.

`gcmi_model_cd(x, y, Ym)`

- Copula-based wrapper around `mi_model_gd`.
- Warns on repeated continuous values.
- Validates that the discrete labels span `0..Ym-1`.
- Applies `copnorm(x)` before calling `mi_model_gd(..., biascorrect=True, demeaned=True)`.

`mi_mixture_gd(x, y, Ym)`

- Gaussian MI between a continuous/Gaussian variable and a discrete variable using a Gaussian mixture.
- Fits a Gaussian to each class.
- Approximates mixture entropy with an unscented-transform-style procedure.
- Does not apply analytic bias correction to the final mixture estimate.

`gcmi_mixture_cd(x, y, Ym)`

- Copula-based wrapper around `mi_mixture_gd`.
- For each class:
  - copula-normalizes that class
  - rescales by a robust MAD-based scale estimate
  - recenters by the class median
- Concatenates transformed class data and evaluates `mi_mixture_gd`.

`cmi_ggg(x, y, z, biascorrect=True, demeaned=False)`

- Gaussian conditional mutual information in bits.
- Computes `I(x; y | z)` from Cholesky log-determinants of:
  - `z`
  - `[x, z]`
  - `[y, z]`
  - `[x, y, z]`
- Supports multivariate conditioning variables.

`gccmi_ccc(x, y, z)`

- Copula-based CMI for three continuous variables.
- Warns on repeated values in each input.
- Applies `copnorm` separately to `x`, `y`, and `z`.
- Calls `cmi_ggg(..., biascorrect=True, demeaned=True)`.

`gccmi_ccd(x, y, z, Zm)`

- Copula-based CMI for continuous `x`, continuous `y`, and discrete `z`.
- For each class of `z`:
  - copula-normalizes `x` and `y` within the class
  - computes class-conditional MI
- Returns a tuple:
  - `CMI`: class-weighted conditional MI
  - `I`: pooled MI after concatenating the within-class copula-normalized samples

### Internal Python helper

`_norm_innerv(x, chC)`

- Internal helper for `mi_mixture_gd`.
- Solves against a Cholesky factor and returns normalized quadratic-form log-likelihood terms.

### Current Python caveats

- The package is now installable from the `python/` subdirectory and can still be imported from the source tree via the compatibility shim.
- The runtime stack depends on NumPy and SciPy.
- The implementation explicitly rejects empty discrete classes and undersampled class covariances with user-facing errors.

## MATLAB Functionality

Source directory: [`matlab/`](matlab)

### Core public functions

`ctransform.m`

- Empirical copula transform along the first axis.
- Uses double sorting to convert values to ranks and scales by `(n + 1)`.

`copnorm.m`

- Copula normalization along the first axis.
- Equivalent to `norminv(ctransform(x))`, implemented with `erfcinv`.

`ent_g.m`

- Gaussian entropy in bits with optional bias correction.
- Treats rows as samples and columns as variables.

`mi_gg.m`

- Gaussian MI in bits for two variables.
- Supports multivariate `x` and `y`.
- Demeans unless `demeaned=true`.

`mi_model_gd.m`

- Gaussian continuous/discrete MI via model comparison.
- Computes unconditional entropy minus weighted class-conditional entropies.
- Supports optional bias correction and optional `demeaned=true`.

`mi_mixture_gd.m`

- Gaussian continuous/discrete MI via Gaussian mixture approximation.
- Computes class Gaussians and approximates the mixture entropy by unscented points.

`cmi_ggg.m`

- Gaussian conditional mutual information for continuous/Gaussian `x`, `y`, `z`.
- Uses covariance submatrices for `z`, `[x,z]`, `[y,z]`, and `[x,y,z]`.

`gcmi_cc.m`

- Copula-based MI for two continuous variables.
- Warns on repeated values, copula-normalizes both inputs, then calls `mi_gg`.

`gcmi_model_cd.m`

- Copula-based continuous/discrete MI using `mi_model_gd`.
- Validates that labels are integers in `0..Ym-1`.

`gcmi_mixture_cd.m`

- Copula-based continuous/discrete MI using `mi_mixture_gd`.
- Performs per-class copula normalization and robust median/MAD rescaling before mixture estimation.

`gccmi_ccc.m`

- Copula-based CMI for three continuous variables.

`gccmi_ccd.m`

- Copula-based CMI for two continuous variables conditioned on a discrete variable.
- Returns both:
  - `CMI`: weighted class-conditional MI
  - `I`: pooled MI from within-class copula-normalized data

`gcmi_version.m`

- Prints and returns a version string.
- Currently reports `0.3`.

`setup_gcmi.m`

- Repository-root helper for adding [`matlab/`](matlab) to the MATLAB path.
- Optionally adds [`matlab_examples/`](matlab_examples) and can persist the path.

### MATLAB vectorized functions

`mi_gg_vec.m`

- Vectorized MI between many Gaussian `x` variables and one common Gaussian `y`.
- Expected shape:
  - `x`: `[Ntrl, Nvec, Ndim]`
  - `y`: `[Ntrl, NyDim]`
- Each output element corresponds to one `x(:, i, :)`.

`mi_model_gd_vec.m`

- Vectorized model-comparison MI between many Gaussian `x` variables and one discrete `y`.
- Same `x` layout convention as `mi_gg_vec`.
- Uses one-hot encoding of classes plus class-wise centered covariance accumulation.

`mi_mixture_gd_vec.m`

- Vectorized mixture-based MI between many Gaussian `x` variables and one discrete `y`.
- Extends the per-class Gaussian-mixture logic to batches of signals.

`cmi_ggg_vec.m`

- Vectorized Gaussian conditional MI.
- Supports a batched `x`, a common `y`, and a batched or broadcastable `z`.

### MATLAB helper functions

`vecchol.m`

- Batched/vectorized Cholesky helper.
- Uses explicit formulas for 1D, 2D, 3D, and 4D cases, then falls back to a loop over `chol`.

`maxstar.m`

- Stable log-sum-exp helper.
- Supports optional signed weights.
- Used by mixture-based estimators.

### Nested/local MATLAB helpers

These are local functions inside vectorized files, not standalone entry points:

- [`matlab/mi_model_gd_vec.m`](matlab/mi_model_gd_vec.m)
  - `removeclassmeans`
  - `indexed2boolean`
- [`matlab/mi_mixture_gd_vec.m`](matlab/mi_mixture_gd_vec.m)
  - `removeclassmeans`
  - `indexed2boolean`
  - `norm_innerv`
- [`matlab/mi_mixture_gd.m`](matlab/mi_mixture_gd.m)
  - `norm_innerv`
- [`matlab/cmi_ggg_vec.m`](matlab/cmi_ggg_vec.m)
  - `vecdiag`

## MATLAB Example Scripts And Example Helpers

Source directory: [`matlab_examples/`](matlab_examples)

### Tutorial scripts

`bias_demo.m`

- Demonstrates why MI estimates can be negative after bias correction under the null.

`discrete_eeg.m`

- End-to-end tutorial for continuous/discrete MI on event-related EEG data.
- Covers permutation testing and sensor/time-point scanning.

`continuous_meg.m`

- End-to-end tutorial for continuous/continuous MI on MEG data.
- Covers lagged analysis, block permutation, multivariate responses, and vector direction/amplitude analyses.

`eeg_temporal_interaction.m`

- End-to-end tutorial for conditional MI and interaction-information analyses over time.
- Covers raw signals, temporal gradients, joint responses, and emergence of novel information.

### Example helper functions

`block_index.m`

- Returns the sample indices for one block in the block-permutation example.

`block_delay.m`

- Applies a lag between blocked MEG data and blocked stimulus data, then concatenates the delayed blocks.

## Current Functional Gaps And Irregularities

- Python and MATLAB naming/documentation are not fully synchronized.
- Some MATLAB helpers are public only by virtue of file layout, not because they are polished standalone APIs.
- The repository now contains regression suites for Python and MATLAB, but cross-language numerical parity checks are still a future improvement area.
