# Hybrid Surface Proposal Experiment v1

Generated: 2026-05-29

## What Was Added

`scripts/tbd_motion_detector.py` now has two experimental hybrid additions:

1. `--hybrid_coast_proposals`
   - Uses the current beam states as cheap predicted proposal boxes.
   - This is inspired by the simple Kalman/state-machine tracker, but it only
     creates proposals. It does not decide final lock by itself.

2. `--scenario_balance`
   - Keeps separate candidate quotas for:
     - `sky`
     - `surface`
     - `boundary`
     - `large`
     - `coast`
   - Goal: avoid a single global top-K list crowding out surface/ground/grass
     candidates.

## AAF1 Result

Baseline: `results/background_surface_audit_v1/aaf1_temporal_stack_fast/`

Hybrid: `results/hybrid_surface_v1/aaf1_temporal_stack_hybrid_audit/`

Textured/non-sky frames:

- baseline strict: 65/75 = 86.7%
- hybrid strict: 70/75 = 93.3%
- baseline loose: 65/75 = 86.7%
- hybrid loose: 74/75 = 98.7%

Clean sky frames:

- baseline strict: 347/351 = 98.9%
- hybrid strict: 327/351 = 93.2%

Runtime:

- baseline temporal-stack run: 126.5 ms/frame
- hybrid temporal-stack run: 203.7 ms/frame

Interpretation: the hybrid helps this aaf1 surface segment, but it hurts easy sky
selection and is too slow in the heavy temporal-stack configuration.

## E271 Result

Baseline: `results/hybrid_surface_v1/e271_large_dark_base_audit/`

Hybrid: `results/hybrid_surface_v1/e271_large_dark_hybrid_audit/`

Textured/non-sky frames:

- baseline strict: 409/683 = 59.9%
- hybrid strict: 393/683 = 57.5%
- baseline loose: 440/683 = 64.4%
- hybrid loose: 436/683 = 63.8%

Runtime:

- baseline large-dark run: 20.2 ms/frame
- hybrid large-dark run: 70.2 ms/frame

Interpretation: this hybrid configuration hurts the e271 large-drone/ridge case.
Do not make it default.

## Honest Read

The idea is still worth keeping as an experiment, but only behind explicit flags.

What seems useful:

- Scenario-balanced candidate pools are the right harness for surface work.
- Coast proposals can rescue some surface frames where current selection jumps to
  appearance clutter.

What is not ready:

- The current hybrid degrades clean sky.
- It hurts e271 under the large-dark setup.
- It adds significant runtime.

Next test should be narrower: use scenario-balanced proposal export for surface
training, then let a learned ranker choose among alternatives. Do not let
coast proposals carry high score without stronger surface/background evidence.
