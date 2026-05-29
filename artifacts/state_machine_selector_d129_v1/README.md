# State-Machine Selector d129 v1

Date: 2026-05-29

Purpose: test whether the existing learned per-frame candidate score can be
turned into better complete-video behavior with an acquire/lock/null state
machine before collecting more labels.

Input benchmark:

- Clip: `d129cd72-3228-426d-9252-4e0b3e14927b`
- Frames: 250
- Visible target frames: 38
- Invisible/no-target frames: 212
- Candidate source: `learned_hist_gbdt_best_per_frame.csv`

## Baselines From Existing Full-Video Eval

| Selector | All-frame accuracy | Visible strict | Invisible no-box |
| --- | ---: | ---: | ---: |
| Current actual pair-rescue selected boxes | 64.0% | 76.3% | 61.8% |
| Simple learned threshold, best prior point | 86.8% | 68.4% | 90.1% |

## New State-Machine Sweep

Best all-frame configuration:

- acquire threshold: `0.85`
- track threshold: `0.45`
- acquire hits: `3`
- max misses: `0`
- max jump: `12 px`
- tentative output: off
- coast output: off

Result:

| Selector | All-frame accuracy | Visible strict | Invisible no-box | First strict frame |
| --- | ---: | ---: | ---: | ---: |
| State machine, best all-frame | 90.0% | 68.4% | 93.9% | 224 |

Best point that preserves the current 76.3% visible strict recall:

- acquire threshold: `0.80`
- track threshold: `0.65`
- acquire hits: `1`
- max misses: `0`
- max jump: `12 px`

Result:

| Selector | All-frame accuracy | Visible strict | Invisible no-box | First strict frame |
| --- | ---: | ---: | ---: | ---: |
| State machine, recall-preserving | 84.8% | 76.3% | 86.3% | 221 |

## Honest Read

This is a real no-new-label improvement path. State beats the old actual tracker
on null suppression by a lot, and it can either maximize all-frame correctness or
preserve the current visible recall while still suppressing many hallucinations.

It is not solved. The high-accuracy config locks 12 frames after first visible
label, so acquisition is too conservative. The recall-preserving config is a
better candidate for demos/control, but still needs validation on more complete
clips before becoming default.

## Files

- `inputs/d129_full_video_vision_labels_v1.csv` - frame-level visible/no-target labels.
- `inputs/learned_hist_gbdt_best_per_frame.csv` - per-frame best learned candidate.
- `state_machine_sweep.csv` - parameter sweep.
- `best_config.json` - best all-frame config.
- `best_frame_predictions.csv` - frame-by-frame output for best all-frame config.

Reproduce:

```bash
python scripts/evaluate_lock_state_machine.py \
  --labels artifacts/state_machine_selector_d129_v1/inputs/d129_full_video_vision_labels_v1.csv \
  --candidates artifacts/state_machine_selector_d129_v1/inputs/learned_hist_gbdt_best_per_frame.csv \
  --out_dir /tmp/d129_lock_state
```
