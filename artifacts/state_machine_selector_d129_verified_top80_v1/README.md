# State-Machine Selector d129 Verified-Score Top80 v1

Date: 2026-05-29

Purpose: test whether the same acquire/lock/null selector can use the detector's
native `verified_score` directly from exported `top_tubes.csv`, before spending
more labeling effort.

Input benchmark:

- Clip: `d129cd72-3228-426d-9252-4e0b3e14927b`
- Frames: 250
- Visible target frames: 38
- Invisible/no-target frames: 212
- Candidate source:
  `/Users/idant/Drone-Strike/results/tbd_tube_pair_rescue_features_top80_full/d129cd72-3228-426d-9252-4e0b3e14927b/top_tubes.csv`
- Score column: `verified_score`
- Candidate cap: top 80 rows per frame

## Result

Best all-frame configuration:

- acquire threshold: `28.0`
- track threshold: `28.0`
- acquire hits: `3`
- max misses: `0`
- max jump: `12 px`
- tentative output: off
- coast output: off

| Selector | All-frame accuracy | Visible strict | Invisible no-box | First strict frame |
| --- | ---: | ---: | ---: | ---: |
| State machine over native `verified_score` | 86.8% | 26.3% | 97.6% | 240 |

The best configuration with at least 50% visible strict recall reached 82.4%
all-frame accuracy, 52.6% visible strict recall, and 87.7% invisible no-box.
No swept configuration reached 65% visible strict recall.

## Honest Read

This is useful as a negative control. The acquire/lock/null state machine is
promising, but the detector's native `verified_score` is not enough to drive it:
high thresholds suppress hallucinations but acquire the visible drone too late,
while lower thresholds bring back clutter.

Do not integrate this as a default selector. The state-machine path should use
learned/out-of-fold candidate scores or a null-aware verifier before becoming
part of the main tracking pipeline.

## Files

- `state_machine_sweep.csv` - parameter sweep.
- `best_config.json` - best all-frame config.
- `best_frame_predictions.csv` - frame-by-frame output for the best config.

Reproduce:

```bash
python scripts/evaluate_lock_state_machine.py \
  --labels artifacts/state_machine_selector_d129_v1/inputs/d129_full_video_vision_labels_v1.csv \
  --candidates /Users/idant/Drone-Strike/results/tbd_tube_pair_rescue_features_top80_full/d129cd72-3228-426d-9252-4e0b3e14927b/top_tubes.csv \
  --score_column verified_score \
  --max_rank 80 \
  --acquire_thresholds 12,14,16,18,20,22,24,26,28,30,32,34,36,40 \
  --track_thresholds 6,8,10,12,14,16,18,20,22,24,26,28,30 \
  --acquire_hits 1,2,3 \
  --max_misses 0,1,2 \
  --max_jump_px 12,18,24,32,48 \
  --out_dir /tmp/d129_lock_state_verified_top80
```
