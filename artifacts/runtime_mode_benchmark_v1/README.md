# Runtime Mode Benchmark v1

Date: 2026-05-29

Purpose: test explicit runtime modes and candidate-local routing using the
pair-rescue detector profile, without heavy surface proposals.

Videos:

- `d129cd72-3228-426d-9252-4e0b3e14927b`
- `aaf1eafd-36d7-43e4-a539-fd79029ddf90`
- `e2711620-6d4e-4f9c-8922-b1b2d1fb74f2`

Each run processed 89 frame pairs.

## Main Result

The candidate-local router is cheap enough to keep testing, especially in
`apply` mode where it replaces the old sky-context pass instead of adding to it.

The current Python beam update is the runtime bottleneck when candidate count is
high:

- d129 baseline: TBD update avg about 32 ms.
- aaf1 baseline: TBD update avg about 39 ms.
- e271 baseline: TBD update avg about 23 ms.

That confirms the embedded plan: Rust should wait until behavior is stable, but
candidate scoring, NMS, and tube update are likely hot-path candidates later.

## Mode Behavior

Pair-rescue profile, avg ms/frame:

| Clip | baseline | auto_log | auto_apply | clean_sky | boundary | surface |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d129 | 51.77 | 53.34 | 43.82 | 37.89 | 37.96 | 51.46 |
| aaf1 | 61.46 | 62.87 | 60.43 | 43.83 | 43.16 | 60.44 |
| e271 | 33.42 | 32.96 | 31.75 | 27.23 | 28.18 | 31.79 |

Interpretation:

- `auto_log` preserves old selection behavior and adds router diagnostics.
- `auto_apply` applies candidate-local penalties and frame-mode candidate caps.
- `clean_sky`/`boundary` reduce candidate count and beam cost.
- `surface` keeps the larger candidate budget, so runtime remains high.

## Concern

The cheap frame router currently over-classifies e271 and aaf1 as `surface`.
That is acceptable for logging, but not good enough to make `auto_apply` a
default. Candidate-local routing is the more reliable control point.

## Artifacts

- `runtime_mode_benchmark.csv` - all mode summaries.
- Each run directory contains `report.json`, `summary.md`, and
  `timing_summary.csv`.
