# Runtime Surface Fallback e271 v1

Date: 2026-05-29

Purpose: validate the committed confidence-gated surface ranker inside the live
`tbd_motion_detector.py` selector, not only as an offline top-tube ranker.

## Detector Profile

All runs use the e271 full clip:

`/Users/idant/Downloads/e2711620-6d4e-4f9c-8922-b1b2d1fb74f2.MP4`

Common detector settings:

- `--downscale 0.5`
- `--beam_width 120`
- `--top_k_candidates 120`
- `--map_peaks --map_radii 2,3,5`
- `--tube_verifier heuristic`
- `--large_dark_peaks --large_dark_top_k 50`
- `--selected_score 6.0`
- no debug frames, no top-tube export

The fallback runs add:

```bash
--surface_ranker_policy confidence_fallback
--surface_ranker_model artifacts/surface_training_v4_multiclip/ranker_loco_v1/extra_trees_surface_xy_ranker.joblib
--surface_ranker_threshold 0.76
```

and sweep `--surface_ranker_top_n`.

## Result

Measured against `artifacts/e271_full_video_vision_labels_v1/e271_full_video_vision_labels_v1.csv`:

| Run | Strict | Loose | Avg ms | P90 ms |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 63.2% | 72.7% | 12.56 | 15.77 |
| Fallback top-20 | 70.2% | 77.8% | 23.92 | 27.40 |
| Fallback top-40 | 70.2% | 77.8% | 24.27 | 28.28 |
| Fallback top-80 | 70.2% | 77.8% | 24.29 | 28.36 |

Top-20 is the best runtime setting in this sweep. It preserves the gain while
staying under 30 Hz on this Mac-side run.

## Honest Read

This is a real runtime selector improvement, not just offline ranking:

- strict +7.0 percentage points over baseline;
- loose +5.2 percentage points over baseline;
- p90 still under 33.3 ms on this machine.

It does **not** solve the late e271 tail (`667..698`): that segment remains
0/32 strict and 1/32 loose. The current surface ranker improves many earlier
surface/ridge frames, but the tail still needs either better features or more
similar hard terrain labels.

Files:

- `eval_summary.csv`: selected-track accuracy and timing summary.
- `baseline_no_top_export/`: baseline selected output.
- `fallback_extra_trees_thr076_top20/`: recommended fallback selected output.
- `fallback_extra_trees_thr076_top40/`, `fallback_extra_trees_thr076_top80/`: runtime cap sweep.
