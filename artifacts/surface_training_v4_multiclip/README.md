# Surface Training v4 Multiclip

Date: 2026-05-29

Purpose: test whether the broader surface-label set improves textured/non-sky
candidate selection without forcing a learned ranker everywhere.

## Inputs

- `surface_textured_xy_labels_v4_plus_d129_full_visible.csv`
- Top-tube source: local `results/surface_ranker_top_tubes_v1`
- Clips represented: `1c`, `529`, `59e`, `7bd`, `aaf1`, `b96`, `d129`, `e271`

The label file has 1,046 rows. The training/eval run uses high and
medium-high visible labels by default, producing 1,030 evaluated frames after
one missing top-tube frame.

## Main Result

The direct learned models are not enough as global replacements:

| Selector | Strict | Loose |
| --- | ---: | ---: |
| Baseline `verified_score` | 80.0% | 85.7% |
| Logistic | 78.6% | 87.0% |
| HistGBDT | 78.1% | 87.5% |
| ExtraTrees | 80.4% | 87.7% |

The useful improvement is a conservative fallback rule:

| Selector | Strict | Loose | Learned frames used |
| --- | ---: | ---: | ---: |
| ExtraTrees only if score >= 0.76, else baseline | 83.3% | 90.1% | 71.4% |

This is the current best multiclip surface-background policy in this artifact.

## Honest Read

This supports a state-machine/fallback selector, not a global learned selector.
The gain comes mostly from e271-like surface frames. aaf1 and d129 do not improve
under this fallback; they remain at the baseline result because the learned model
does not cross the confidence threshold often enough to override baseline.

Next high-ROI work is still more true tree/grass/terrain labels and then
testing whether this confidence fallback holds on new clips.

## Files

- `ranker_loco_v1/loco_summary.csv`: direct model LOCO metrics.
- `ranker_loco_v1/fallback_sweep.csv`: learned-score fallback threshold sweep.
- `ranker_loco_v1/best_fallback_by_clip.csv`: per-clip breakdown for the best fallback.
- `ranker_loco_v1/best_fallback_predictions.csv`: selected candidate per frame for the best fallback.
- `ranker_loco_v1/extra_trees_surface_xy_ranker.joblib`: final direct ExtraTrees model trained on all examples.
