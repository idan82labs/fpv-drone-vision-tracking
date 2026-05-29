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

The candidate-local router is cheap enough to keep testing, and the first beam
hot-path pass substantially reduced runtime. The latest detector now precomputes
state warp/prediction references once per frame and skips extra diagnostic pair
features unless they are needed.

The current Python beam update is the runtime bottleneck when candidate count is
high:

- d129 baseline: TBD update avg about 7.8 ms.
- aaf1 baseline: TBD update avg about 10.5 ms.
- e271 baseline: TBD update avg about 5.3 ms.

That confirms the embedded plan: Rust should wait until behavior is stable, but
candidate scoring, NMS, and tube update are likely hot-path candidates later.

## Mode Behavior

Pair-rescue profile, avg ms/frame:

| Clip | baseline | auto_log | auto_apply | clean_sky | boundary | surface |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d129 | 28.72 | - | 29.12 | 27.83 | - | - |
| aaf1 | 34.41 | - | 37.49 | 33.09 | - | - |
| e271 | 16.39 | - | 17.78 | 17.53 | - | - |

Interpretation:

- `auto_apply` applies candidate-local penalties and frame-mode candidate caps.
- `clean_sky`/`boundary` reduce candidate count and beam cost.
- `surface` keeps the larger candidate budget, so runtime remains high.

## Concern

Mac-side 30 Hz is now plausible for d129/e271 and borderline for aaf1. This is
still not a Pi 5 claim: capture/decode, p90/p99 timing, and thermal behavior are
not included here.

## Artifacts

- `runtime_mode_benchmark.csv` - all mode summaries.
- Each run directory contains `report.json`, `summary.md`, and
  `timing_summary.csv`.
