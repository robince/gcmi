# MATLAB C++ MEX Rewrite

This directory contains the new MATLAB-native C++ MEX implementation for the
optimized batch kernels.

## Scope

The initial `R2024b` implementation currently provides:

- build/runtime probes:
  - `gcmi_cpp_ping`
  - `gcmi_cpp_blas_probe`
  - `gcmi_cpp_omp_probe`
  - `gcmi_cpp_runtime_probe`
- first real kernels:
  - `copnorm_slice_cpp`
  - `info_cc_slice_cpp`
  - `info_cd_slice_cpp`

The current build and release flow targets macOS and Linux `x86_64`.

## Direct entrypoint API

The current estimator entrypoints are:

```matlab
CX = copnorm_slice_cpp(X, Nthread)
I = info_cc_slice_cpp(X, Xdim, Y, Ntrl, Nthread)
I = info_cd_slice_cpp(X, Xdim, Y, Ym, Ntrl, Nthread)
```

Expected layouts:

| Entrypoint | X layout | Other inputs | Output |
| :-- | :-- | :-- | :-- |
| `copnorm_slice_cpp` | `[Ntrl, Npage]` | `Nthread` | `Ntrl x Npage` |
| `info_cc_slice_cpp` | `[Ntrl, Xdim, Npage]` or `[Ntrl, Xdim]` for a single page | `Y`: `[Ntrl, Ydim]` | `1 x Npage` |
| `info_cd_slice_cpp` | `[Xdim, Ntrl, Npage]` or `[Xdim, Ntrl]` for a single page | `Y`: labels of length `Ntrl` | `1 x Npage` |

Typical reshaping from the main MATLAB reference layout:

```matlab
nativeXcc = permute(xcc, [1 3 2]);  % [Ntrl, Npage, Xdim] -> [Ntrl, Xdim, Npage]
nativeXcd = permute(xcd, [3 1 2]);  % [Ntrl, Npage, Xdim] -> [Xdim, Ntrl, Npage]
```

## Label convention

All discrete labels accepted by the new C++ MEX entrypoints are zero-based:

- valid labels are `0 .. M-1`
- empty classes are rejected
- legacy `1 .. M` direct-MEX semantics are not supported here

This is intentionally different from the historical direct-call convention used
by the legacy Fortran MEX layer under `extern/gcmi_mex`.

## Runtime layout

Compiled binaries are written to:

- `matlab/cpp_mex/bin/<MATLAB release>/<mexext>/`

`setup_gcmi` auto-discovers that directory for the current MATLAB release when
matching binaries are present.
