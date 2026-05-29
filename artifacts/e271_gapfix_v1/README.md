# e271 Reel Gap Investigation and Gap Fix v1

Generated: 2026-05-29

## Trigger

In `next_batch_extended_e2e_reel_v3_tail_extended.mp4`, reel seconds 25-36 map to the
e271 source clip from about frame 23 to frame 573. The drone is plainly visible for
much of this interval, but the v3 reel displays `NO CONFIDENT TRACK`.

## Root Cause

The v3 e271 selection file only contained e271 rows from frame 594 onward:

- e271 v3 rows: frames 594-698 only.
- The visible early/mid e271 segment was not labeled and therefore could not be
  rendered or used as training data.
- The previous backward-template recovery was rejected because it drifted onto
  sky/terrain texture. That rejection was correct for that tracker, but it left a
  large target-visible hole.

Algorithmically, the existing temporal-stack visual top-tubes export also showed
the problem: with the new dense labels, oracle@100 was only 49.0% on all e271
labels and 40.9% on high-confidence labels. The target was often absent from
the candidate pool, not merely ranked low.

## Fix Implemented

Added a `large_dark` proposal source to `scripts/tbd_motion_detector.py`:

- `--large_dark_peaks`
- `--large_dark_top_k`
- `--large_dark_score_floor`
- `--large_dark_nms_px`
- `--large_dark_box_full`
- `--large_dark_bg_sigma`
- `--large_dark_score_weight`

This proposal source is a full-resolution local dark-silhouette detector aimed at
close, clearly visible drones. It is deliberately separate from the tiny/micro
and temporal residual proposal paths.

## Dense Labels Generated

Files:

- `e271_gapfix_dense_top1_labels_v1.csv`
- `e271_gapfix_dense_top1_selection_v1.csv`
- `e271_gapfix_dense_for_xy_ranker_v1.csv`
- `vision_verified_next_batch_selection_v4_e271_gapfix.csv`

Frames:

- 594 new dense rows for e271 frames 0-593.
- Combined selection has 1271 rows.

Confidence tiers:

- high: 275 rows
- medium: 50 rows
- low_review_required: 175 rows
- medium_high: 94 rows

Use `low_review_required` as weak data only until reviewed.

## Before/After Proposal Audit

Existing temporal-stack visual top-tubes on new labels:

- all oracle@100: 49.0%
- high-confidence oracle@100: 40.9%
- best learned strict/loose on all labels: 19.6% / 39.9%
- best learned strict/loose on high labels: 18.0% / 41.2%

New large-dark proposal run:

- runtime: 20.15 ms/frame on this machine
- avg candidates/frame: 16.03
- p90 candidates/frame: 25
- all oracle@100: 92.8%
- high-confidence oracle@100: 85.4%
- ExtraTrees strict/loose on all labels: 77.4% / 79.6%
- ExtraTrees strict/loose on high labels: 83.3% / 85.1%

## Videos

- `next_batch_extended_e2e_reel_v4_e271_gapfix.mp4`
  - Same reel as v3 but with the e271 segment replaced by gap-fixed labels.
- `e271_gapfix_full_0p0-13p98s_v1.mp4`
  - Dense label render, not autonomous detector output.
- `e271_large_dark_extra_trees_allfit_gap_0p4-11p6s_v1.mp4`
  - Learned-ranker render using the new large-dark top-tubes and all-fit model.
  - This is an overfit diagnostic, not a deployment/generalization estimate.

## Honest Read

This was a real blind spot. The drone was visible and the system had no proper
proposal source for this close/large dark-silhouette case. The new proposal
primitive dramatically improves candidate-pool recall on this segment, but the
terrain/horizon frames still need human review before they become strict ground
truth.
