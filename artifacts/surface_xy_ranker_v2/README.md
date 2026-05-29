# Surface Tracking Harness v2

Generated: 2026-05-29

## Purpose

This run targets the next failure class: drone tracks against textured backgrounds
such as trees, grass, terrain, and ridge/skyline clutter.

It deliberately separates surface/background frames from easy clean-sky frames so
accuracy numbers cannot be inflated by the easy cases.

## Inputs

- `results/background_surface_audit_v1/pair_rescue_next_batch_gapfix/frame_background_audit.csv`
- `results/background_surface_audit_v1/aaf1_temporal_stack_fast/frame_background_audit.csv`
- `results/background_surface_audit_v1/surface_textured_xy_labels_v2_plus_aaf1.csv`
- `results/surface_ranker_top_tubes_v1/*/top_tubes.csv`

The label set contains 985 textured/non-sky labeled frames:

- e271: 458
- 1c: 309
- 7bd: 88
- aaf1: 75
- 529: 55

## Harnesses Added

- `scripts/audit_background_splits.py`
  - Classifies dense XY labels by local target background:
    `clean_sky`, `boundary_mixed`, `textured_non_sky`.
  - Reports oracle recall and selected-box accuracy per split.

- `scripts/make_surface_vision_packet.py`
  - Creates clean and diagnostic frame packets for vision/human XY labeling.
  - First packets:
    - `results/surface_vision_packet_v1/textured_failures_top80/`
    - `results/surface_vision_packet_v1/aaf1_textured_failures/`

- `scripts/train_surface_xy_ranker.py`
  - Multi-clip XY tube ranker with leave-one-clip-out evaluation.

## Results

### Pair-rescue next-batch/gapfix split audit

Clean sky:

- strict: 40/40 = 100%

Textured/non-sky:

- oracle@80: 1008/1135 = 88.8%
- strict selected: 679/1135 = 59.8%

Interpretation: textured/non-sky is mainly a ranking/selection problem when the
target is in the alternatives, plus a smaller proposal-miss problem.

### Surface ranker v1, no aaf1

Labels: high + medium_high textured/non-sky from next-batch/gapfix.

- baseline strict: 72.5%
- best LOCO strict: 79.3%
- baseline loose: 77.9%
- best LOCO loose: 89.0%

This was a real improvement on the next-batch/e271-like distribution.

### Surface ranker v2, with aaf1 added

Labels: high + medium_high + a small number of medium textured/non-sky rows.

- baseline strict: 80.3%
- best LOCO strict: 80.1%
- baseline loose: 86.3%
- best LOCO loose: 87.9%

This does **not** justify blindly replacing the current selector. The model
improves e271-like ridge/terrain frames, but it generalizes poorly to the aaf1
surface segment when aaf1 is held out. That means we need more true tree/grass
labels and probably a background-conditioned selector instead of one global
surface ranker.

## Honest Read

The ROI is in the harness and split, not in immediately integrating the v2 model.
The current evidence says:

1. Clean sky is not the bottleneck.
2. Surface frames need their own benchmark.
3. The target is often in top alternatives, so ranking is worth improving.
4. One global learned surface ranker is not stable enough yet.
5. More vision labels should focus on true surface contact frames, especially
   tree/grass backgrounds and not only e271 ridge-line cases.
