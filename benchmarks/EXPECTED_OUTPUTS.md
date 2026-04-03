# Expected Outputs

This document defines what should be fixed and versioned versus what should remain threshold-based.

## Fixed Golden Outputs

Use fixed expected outputs for deterministic correctness fixtures.

These outputs should be versioned per fixture and kernel and stored in machine-readable form.

Suggested future layout:

- `benchmarks/golden/v1/<fixture_id>.json`

Each golden file should contain:

- fixture metadata
- kernel name
- canonical label convention
- canonical logical shape description
- output values
- tolerance policy
- source implementation used to generate the golden outputs

## Performance Expectations

Do not treat absolute performance numbers as fixed golden outputs.

Instead, performance expectations should be expressed as:

- minimum speedup vs reference
- minimum scaling efficiency at chosen thread counts
- no-regression checks against previous baselines on the same machine class

## Versioning

If semantics intentionally change, increment the golden-output version.

Examples of semantics that would require a version review:

- changing tie handling in `copnorm`
- changing bias-correction behavior inside optimized kernels
- changing label normalization behavior
- changing the canonical logical interface
