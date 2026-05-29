# Runtime Mode Benchmark Surface Extras v1

Date: 2026-05-29

Purpose: verify that heavy surface extras are branch-gated when explicitly
enabled. This run used `--scenario_balance`, `--temporal_stack_peaks`, and
`--hybrid_coast_proposals` on short 45-frame slices.

## Main Result

Heavy surface extras are still too slow as a general live path in Python:

| Clip | auto_log | auto_apply | clean_sky | surface |
| --- | ---: | ---: | ---: | ---: |
| d129 | - | 57.84 ms | 35.61 ms | 86.87 ms |
| e271 | - | 19.89 ms | 18.71 ms | 53.27 ms |

p90 ms/frame:

| Clip | auto_apply | clean_sky | surface |
| --- | ---: | ---: | ---: |
| d129 | 91.85 ms | 46.27 ms | 100.65 ms |
| e271 | 27.67 ms | 25.43 ms | 63.33 ms |

Interpretation:

- `auto_apply` keeps temporal-stack acquisition off by default unless explicitly
  requested, but allows lock-local coast proposals from mature tracks.
- Forced `surface` remains too slow for live Python deployment.
- Heavy temporal-stack proposal generation, not just beam update, is the main
  reason surface mode misses the live budget.
- This supports the plan: use surface extras only after router behavior is
  stable, and do not make them global.

## Artifacts

- `runtime_mode_benchmark.csv` - all mode summaries.
- Each run directory contains `report.json`, `summary.md`, and
  `timing_summary.csv`.
