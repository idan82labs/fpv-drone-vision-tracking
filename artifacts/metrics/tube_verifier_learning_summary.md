# Tube Verifier Learning Summary

Date: 2026-05-28

## Baseline To Beat

Current best remains:

`results/tbd_tube_pair_rescue_full`

Sparse checkpoint performance:

| Variant | Gold hit / total | Gold recall | Clutter suppressed / total | Known FP |
|---|---:|---:|---:|---:|
| pair_rescue | 10/18 | 0.556 | 17/18 | 2 |

## Proposal Ceiling

Using current-best top-80 exported tubes:

| Set | Gold found in top-80 | Oracle recall |
|---|---:|---:|
| reviewed gold checkpoints | 14/18 | 0.778 |

Source:

`results/tube_verifier_pairwise_top80_loco/proposal_oracle_gold.csv`

Interpretation: the proposal layer is close to the 80% target. The main problem is selecting/ranking the correct tube.

## Learned Verifier Attempts

### Binary ridge logistic verifier

Output:

`results/tube_verifier_linear_top80_loco`

Leave-one-clip-out:

| Gold hit / total | Gold recall | Clutter suppressed / total | Known FP |
|---:|---:|---:|---:|
| 7/18 | 0.389 | 14/18 | 3 |

In-sample:

| Gold hit / total | Gold recall | Clutter suppressed / total | Known FP |
|---:|---:|---:|---:|
| 8/18 | 0.444 | 17/18 | 0 |

Decision: reject. It underfits positives and does not beat baseline.

### Pairwise ranking verifier

Output:

`results/tube_verifier_pairwise_top80_loco`

Leave-one-clip-out:

| Gold hit / total | Gold recall | Clutter suppressed / total | Known FP |
|---:|---:|---:|---:|
| 7/18 | 0.389 | 11/18 | 8 |

In-sample:

| Gold hit / total | Gold recall | Clutter suppressed / total | Known FP |
|---:|---:|---:|---:|
| 12/18 | 0.667 | 12/18 | 7 |

Decision: reject. It learns same-clip ranking signals but does not generalize.

## Diagnosis

The weak automatic labels are not enough. Several gold frames have no matching target tube in top-80, and many positive labels are tiny boxes with ambiguous loose matching. The model also learns suspicious signs, such as positive weight on support/texture in some runs, which suggests the sparse labels are not separating target tubes from hard terrain/tree/road alternatives.

This does not invalidate the learned-verifier direction. It says we need actual top-tube alternative labels before training.

## Human Review Packet

Prepared:

`results/tube_alternative_review_packet_top16`

Zip:

`results/tube_alternative_review_packet_top16.zip`

Contents:

- `tube_alternatives_to_label.csv`
- `overviews/`: full downscaled frame with reviewed bbox in cyan and top alternatives numbered.
- `crops/`: crop sheets for the same alternatives.

Suggested labels for `human_label`:

- `target`
- `near_target_wrong_center`
- `static_hotspot`
- `line_attached`
- `parallax_edge`
- `boundary_artifact`
- `appearance_blob`
- `terrain_texture`
- `noise`
- `uncertain`

This is the next meaningful data step. Once these labels are filled, retrain the verifier with true hard-negative classes instead of weak checkpoint-level labels.

## Feature-Rich Export For Training

Prepared after the professor's revised guidance:

`results/tbd_tube_pair_rescue_features_top80_full`

This export preserves the current `pair_rescue` checkpoint behavior:

| Variant | Gold hit / total | Gold recall | Clutter suppressed / total | Known FP |
|---:|---:|---:|---:|
| pair_rescue | 10/18 | 0.556 | 17/18 | 2 |
| feature_export | 10/18 | 0.556 | 17/18 | 2 |

It adds feature-only columns for the next learned ranker:

- `competitor_margin`
- `tube_mean_pair_raw`
- `tube_positive_pair_raw_rate`
- `tube_mean_pair_bg`
- `tube_positive_pair_bg_rate`
- `tube_mean_pair_bg_local`
- `tube_mean_align_gain`
- `tube_mean_bg_dist`
- `tube_mean_cv_resid`
- `tube_mean_bg_minus_cv`
- `tube_log_cand_density`

Runtime note: this export averaged about 38.4 ms/frame with a max clip average around 60.3 ms/frame, so it is offline training data, not a live detector setting.

Once `human_label` is filled in `tube_alternatives_to_label.csv`, train with:

```bash
.venv/bin/python scripts/train_tube_verifier.py \
  --labels results/professor_followup_2_package/review_labels_labeled.csv \
  --tube_labels results/tube_alternative_review_packet_top16/tube_alternatives_to_label.csv \
  --results_dir results/tbd_tube_pair_rescue_features_top80_full \
  --out_dir results/tube_verifier_human_labeled_top16_loco \
  --max_rank 16 \
  --train_mode pairwise
```

Do not train against the weak labels again unless it is just a smoke test; both weak-label variants failed leave-one-clip-out.
