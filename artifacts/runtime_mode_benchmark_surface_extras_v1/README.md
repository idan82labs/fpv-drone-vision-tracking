# Runtime Mode Benchmark Surface Extras v1

Date: 2026-05-29

Purpose: verify that heavy surface extras are branch-gated when explicitly
enabled. This run used `--scenario_balance`, `--temporal_stack_peaks`, and
`--hybrid_coast_proposals` on short 45-frame slices.

## Main Result

Heavy surface extras are still too slow as a general live path in Python:

| Clip | auto_log | auto_apply | clean_sky | surface |
| --- | ---: | ---: | ---: | ---: |
| d129 | - | 60.46 ms | 37.10 ms | 90.28 ms |
| e271 | - | 21.47 ms | 19.23 ms | 54.56 ms |

Interpretation:

- `auto_apply` keeps temporal-stack acquisition off by default unless explicitly
  requested, but allows lock-local coast proposals from mature tracks.
- Forced `surface` remains too slow for live Python deployment.
- This supports the plan: use surface extras only after router behavior is
  stable, and do not make them global.

## Artifacts

- `runtime_mode_benchmark.csv` - all mode summaries.
- Each run directory contains `report.json`, `summary.md`, and
  `timing_summary.csv`.
