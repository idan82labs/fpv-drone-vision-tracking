# Surface Training v3 Results

Date: 2026-05-29

## What Changed

- Added corrected full-video labels for `e2711620-6d4e-4f9c-8922-b1b2d1fb74f2`, frames `0..698`.
- Replaced older e271 tail snippets because the inherited labels drifted right of the drone around frames `660..698`.
- Evaluated e271 as an all-visible continuity/ranking benchmark.
- Tested a second-order constant-velocity selector in `scripts/evaluate_xy_sequence_ranker.py`.
- Built a combined surface/verified label set with 1,287 visible rows across four clips.

## e271 Full-Visible Benchmark

| Proposal / selector | Oracle@80 | Strict recall | Loose recall | Notes |
| --- | ---: | ---: | ---: | --- |
| Pair-rescue, baseline verified score | 88.3% | 42.6% | 45.7% | Proposal coverage is the main limit. |
| Pair-rescue, learned framewise HGBDT | 88.3% | 60.7% | 77.8% | Better ranking, still lower oracle. |
| Pair-rescue, learned sequence | 88.3% | 68.1% | 80.7% | First-order continuity helps. |
| Hybrid surface proposals, baseline verified score | 96.7% | 62.8% | 73.5% | Hybrid materially improves proposal coverage. |
| Hybrid surface proposals, learned framewise HGBDT | 96.7% | 65.5% | 85.1% | Ranking improves loose recall. |
| Hybrid surface proposals, learned sequence | 96.7% | 78.3% | 92.7% | Best current full-video e271 result. |

`strict` means selected center within 8 detector pixels of the vision label. `loose` means within 16 detector pixels.

## Constant-Velocity Test

The second-order constant-velocity selector did not improve this clip. Best quick CV run was worse than first-order continuity:

| Selector | Strict recall | Loose recall |
| --- | ---: | ---: |
| First-order Viterbi, jump 10, weight 0.75 | 78.3% | 92.7% |
| CV Viterbi, same jump/weight, accel 1.0 | 73.3% | 85.6% |

Interpretation: the FPV chase geometry is not close enough to constant-velocity in image coordinates. Acceleration-aware smoothing is still useful as a feature/hypothesis, but not as a hard selector default.

## Remaining Failure Runs

The best hybrid sequence still fails in concentrated runs:

- `433..459`: faint/low-confidence terrain-backed target; some labels are deliberately lower confidence.
- `682..698`: correct candidate is often present, sometimes rank 3, but the selector jumps to a high-scoring terrain branch.

See `artifacts/e271_full_video_hybrid_surface_sequence_ranker_v1/failure_runs_contact.jpg`.

## Multi-Clip LOCO Check

Combined-label leave-one-clip-out did not beat the existing verified-score baseline:

| Model | Frames | Oracle@80 | Strict recall | Loose recall |
| --- | ---: | ---: | ---: | ---: |
| Baseline verified score | 1,286 | 97.8% | 75.5% | 83.8% |
| Logistic | 1,286 | 97.8% | 71.3% | 82.8% |
| HGBDT | 1,286 | 97.8% | 63.3% | 78.4% |
| ExtraTrees | 1,286 | 97.8% | 66.1% | 82.0% |

Held-out e271 remains weak because the other labeled clips do not yet cover enough similar surface/terrain cases. This confirms the blocker is ranker generalization on hard surface backgrounds, not Rust/runtime or global affine tuning.

## Current Recommendation

Keep the hybrid surface proposal path as a proposal source for surface-backed candidates, but do not make it the final decision layer globally. The next highest-ROI work is more complete surface-backed labels from additional clips, plus feature work aimed at the specific wrong branch pattern: target-vs-background aligned crop stacks and hard-negative branch competitors.
