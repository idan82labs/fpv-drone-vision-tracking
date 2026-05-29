# Surface Training v3 With Full e271

This packet combines the corrected full-video e271 labels with the existing verified XY labels.

## Labels

- `surface_e271_full_plus_verified_labels.csv`
- 1,287 visible-target rows across four clips:
  - `e2711620-6d4e-4f9c-8922-b1b2d1fb74f2` from `artifacts/e271_full_video_vision_labels_v1`
  - existing verified rows from `artifacts/labels/vision_verified_next_batch_labels_v3_tail_extended_for_xy_ranker.csv`
  - non-e271 accepted surface rows from `artifacts/surface_training_v2_vision_checked/surface_accepted_labels.csv`

Older e271 snippets are intentionally excluded because the corrected full-video labels supersede them, especially the tail segment where inherited labels had drifted right of the drone.

## Proposal Sources Used For Training

The training run uses a local proxy directory that points:

- e271 -> hybrid surface top-tubes
- other clips -> pair-rescue top-tubes

The proxy itself is not committed because generated `top_tubes.csv` files live under ignored `results/` directories.
