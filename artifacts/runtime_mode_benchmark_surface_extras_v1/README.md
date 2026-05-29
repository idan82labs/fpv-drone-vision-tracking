# Runtime Mode Benchmark Surface Extras v1

Date: 2026-05-29

Purpose: verify that heavy surface extras are branch-gated when explicitly
enabled. This run used `--scenario_balance`, `--temporal_stack_peaks`, and
`--hybrid_coast_proposals` on short 45-frame slices.

## Main Result

Heavy surface extras are still too slow as a general live path in Python:

| Clip | auto_log | auto_apply | clean_sky | surface |
| --- | ---: | ---: | ---: | ---: |
| d129 | 142.58 ms | 50.23 ms | 50.69 ms | 133.60 ms |
| e271 | 119.83 ms | 119.97 ms | 29.88 ms | 121.47 ms |

Interpretation:

- On d129, `auto_apply` suppressed heavy surface branches for non-surface frames
  and cut runtime from about 143 ms to about 50 ms.
- On e271, the frame router classified the slice as surface, so `auto_apply`
  allowed the heavy path and stayed slow.
- This supports the plan: use surface extras only after router behavior is
  stable, and do not make them global.

## Artifacts

- `runtime_mode_benchmark.csv` - all mode summaries.
- Each run directory contains `report.json`, `summary.md`, and
  `timing_summary.csv`.
