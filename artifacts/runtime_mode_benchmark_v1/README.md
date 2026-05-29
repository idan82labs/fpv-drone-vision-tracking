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

The candidate-local router is cheap enough to keep testing, and the beam
hot-path passes reduced runtime without changing selected rates. The latest
detector now precomputes state warp/prediction references once per frame, skips
extra diagnostic pair features unless they are needed, and avoids materializing
candidate-transition states that cannot beat the current per-candidate winner.

The current Python beam update is the runtime bottleneck when candidate count is
high:

- d129 baseline: TBD update avg about 8.6 ms, p90 about 14.2 ms.
- aaf1 baseline: TBD update avg about 10.8 ms, p90 about 15.7 ms.
- e271 baseline: TBD update avg about 5.1 ms, p90 about 9.5 ms.

That confirms the embedded plan: Rust should wait until behavior is stable, but
candidate scoring, NMS, and tube update are likely hot-path candidates later.

## Mode Behavior

Pair-rescue profile, avg ms/frame:

| Clip | baseline | auto_log | auto_apply | clean_sky | boundary | surface |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d129 | 28.20 | - | 28.75 | 27.10 | - | - |
| aaf1 | 33.88 | - | 36.74 | 32.77 | - | - |
| e271 | 14.93 | - | 15.98 | 15.54 | - | - |

Pair-rescue profile, p90 ms/frame:

| Clip | baseline | auto_apply | clean_sky |
| --- | ---: | ---: | ---: |
| d129 | 41.10 | 45.24 | 37.97 |
| aaf1 | 46.17 | 51.11 | 43.66 |
| e271 | 22.29 | 24.07 | 21.78 |

Interpretation:

- `auto_apply` applies candidate-local penalties and frame-mode candidate caps.
- `clean_sky`/`boundary` reduce candidate count and beam cost.
- `surface` keeps the larger candidate budget, so runtime remains high.

## Concern

Average Mac-side 30 Hz is plausible for d129/e271 and borderline for aaf1, but
p90 timing is not yet 30 Hz on the harder clips. This is still not a Pi 5 claim:
capture/decode, p99 timing, and thermal behavior are not included here.

## Artifacts

- `runtime_mode_benchmark.csv` - all mode summaries.
- Each run directory contains `report.json`, `summary.md`, and
  `timing_summary.csv`.
