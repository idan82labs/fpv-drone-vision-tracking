# Full-Video State Ranker d129 OOF v1

Date: 2026-05-29

Purpose: test the next no-new-label step: train candidate scores from existing
full-video frame labels, then drive the acquire/lock/null state machine with
those scores instead of raw detector `verified_score`.

Input benchmark:

- Clip: `d129cd72-3228-426d-9252-4e0b3e14927b`
- Labels: `artifacts/state_machine_selector_d129_v1/inputs/d129_full_video_vision_labels_v1.csv`
- Candidate source:
  `/Users/idant/Drone-Strike/results/tbd_tube_pair_rescue_features_top80_full/d129cd72-3228-426d-9252-4e0b3e14927b/top_tubes.csv`
- Frames: 250
- Visible target frames: 38
- Invisible/no-target frames: 212
- Candidate examples: 19,618
- Positive candidate examples: 51
- Negative candidate examples: 19,567
- Fold strategy: stratified blocks inside the one clip

## Candidate Ranking Result

The out-of-fold best candidate per frame, before any threshold/state logic:

| Score source | Visible strict best-candidate rate |
| --- | ---: |
| Baseline native `verified_score` | 26/38 = 68.4% |
| OOF logistic | 38/38 = 100.0% |
| OOF HistGBDT | 37/38 = 97.4% |
| OOF ExtraTrees | 38/38 = 100.0% |

## State-Machine Result

Best ExtraTrees-driven state-machine config:

- acquire threshold: `0.90`
- track threshold: `0.02`
- acquire hits: `1`
- max misses: `0`
- max jump: `12 px`

| Selector | All-frame accuracy | Visible strict | Invisible no-box | Selected frames |
| --- | ---: | ---: | ---: | ---: |
| Current actual pair-rescue selected boxes | 64.0% | 76.3% | 61.8% | n/a |
| Prior learned HistGBDT state artifact | 90.0% | 68.4% | 93.9% | 51 |
| Native `verified_score` state negative control | 86.8% | 26.3% | 97.6% | 15 |
| OOF ExtraTrees + state machine | 99.6% | 38/38 = 100.0% | 211/212 = 99.5% | 39 |

The single false positive is frame 211, immediately before the first visible
target label. The first strict target frame is 212, so strict acquisition latency
is 0 frames on this label set.

## Honest Read

This is the strongest no-new-label result so far, but it is not deployment proof.
It is out-of-fold inside one clip, not leave-one-clip-out across independent
videos. Because the visible target appears in one continuous tail segment, the
model still benefits from neighboring positive examples in other folds.

The result is still useful: it shows the state-machine architecture can work if
the score is null-aware and candidate-level, and it gives a concrete integration
target. The next validation step is to build the same full-video OOF table on at
least one more complete clip before making this selector a default.

## Files

- `model_summary.csv` - OOF best-candidate summary.
- `metadata.json` - training/eval metadata and feature list.
- `oof_best_per_frame_*.csv` - one best candidate per frame for state sweeps.
- `state_machine_sweep_extra_trees.csv` - state sweep for ExtraTrees scores.
- `best_config_*.json` - best state config per model.
- `best_frame_predictions_extra_trees.csv` - frame-level output for the best
  ExtraTrees state config.

Reproduce:

```bash
python scripts/train_full_video_state_ranker.py \
  --labels artifacts/state_machine_selector_d129_v1/inputs/d129_full_video_vision_labels_v1.csv \
  --top_tubes /Users/idant/Drone-Strike/results/tbd_tube_pair_rescue_features_top80_full/d129cd72-3228-426d-9252-4e0b3e14927b/top_tubes.csv \
  --clip d129cd72-3228-426d-9252-4e0b3e14927b \
  --out_dir /tmp/d129_full_video_state_ranker_oof_v1 \
  --max_rank 80 \
  --folds 5 \
  --fold_strategy stratified_blocks

python scripts/evaluate_lock_state_machine.py \
  --labels artifacts/state_machine_selector_d129_v1/inputs/d129_full_video_vision_labels_v1.csv \
  --candidates /tmp/d129_full_video_state_ranker_oof_v1/oof_best_per_frame_extra_trees.csv \
  --score_column oof_extra_trees_score \
  --max_rank 80 \
  --acquire_thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9 \
  --track_thresholds 0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8 \
  --acquire_hits 1,2,3 \
  --max_misses 0,1,2 \
  --max_jump_px 12,18,24,32,48 \
  --out_dir /tmp/d129_state_ranker_selector_extra_trees
```
