# Temporal Stack Proposal Recovery Summary

Date: 2026-05-28

Clip: `aaf1eafd-36d7-43e4-a539-fd79029ddf90`

Label set: `results/manual_dense_xy_aaf1_empty_v1/manual_xy_labels_aaf1_empty_sample_v5_final.csv`

## What Changed

Added an optional past-frame stabilized temporal-stack proposal source in `scripts/tbd_motion_detector.py`:

- `--temporal_stack_peaks`
- full-resolution past-frame warping into the current frame
- short median background residual
- native compact dark/CLAHE scoring
- halo offsets around top temporal peaks to recover nearby target centers

This is deliberately a proposal-recovery source. It does not solve final tube ranking by itself.

## Standalone Proposal Recovery

Best standalone centered-stack upper bound:

| Source | Avg candidates | R@80 | High-confidence R@80 |
| --- | ---: | ---: | ---: |
| old pair-rescue top80 baseline | - | 0.313 | 0.412 |
| `temporal_halo`, centered stack | 120 | 0.761 | 0.912 |
| `temporal_halo`, past-only `-8,-5,-3,-2,-1` | 118 | 0.701 | 0.824 |
| `temporal_halo`, past-only `-5,-3,-2,-1` | 118 | 0.687 | 0.824 |

Interpretation: the temporal-stack halo proposal source is a real recovery improvement. The centered stack is an offline upper bound because it uses future frames; the past-only result is the honest online/low-latency version.

## Integrated Tracker Runs

Manual XY audit uses detector-scale center tolerance `<= 3 px`.

| Run | Proposal recall | High-conf proposal recall | Selected hit | High-conf selected hit | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| old pair-rescue top80 baseline | 0.328 | 0.412 | 0.030 | - | ~18 ms/frame |
| temporal stack, full past `-8,-5,-3,-2,-1` | 0.776 | 0.941 | 0.194 | 0.265 | 188 ms/frame |
| temporal stack fast `-5,-3,-1`, old map kept | 0.687 | 0.794 | 0.209 | 0.265 | 126 ms/frame |
| temporal stack only fast `-5,-3,-1` | 0.672 | 0.794 | 0.224 | 0.294 | 108 ms/frame |
| temporal stack 2-frame `-3,-1` | 0.642 | 0.794 | not selected-audited | not selected-audited | 105 ms/frame |

## Current Read

This is meaningful progress on proposal recovery, not on final ranking. The target now appears in the tracker alternatives much more often, especially on high-confidence manual labels. However, selected-box accuracy is still poor because the current hand-written tube verifier frequently ranks a clutter tube over the recovered target tube.

The full-resolution temporal stack is too slow in Python for real-time use. The result is valuable as a training/proposal-discovery path and as evidence that higher-resolution stabilized temporal evidence helps. To make it runtime viable, the next engineering step should be an ROI-limited or native implementation, or a learned verifier that can use the recovered alternatives without carrying hundreds of candidates into the beam.

