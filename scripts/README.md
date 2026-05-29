# Scripts

This folder contains both detector hot-path code and desktop lab tools.

## Runtime Core

- `tbd_motion_detector.py` - main OpenCV detector, proposal generation, beam
  tracking, router flags, and export/report modes.
- `motion_detector_v2.py` - supporting motion-detector path used by the Pi
  bundle and experiments.

These are the only `scripts/` files currently included in the minimal Pi
bundle.

## Evaluation and Selection

- `evaluate_tracking_run.py` - compare selected boxes against frame labels.
- `apply_surface_sequence_selector.py` - offline/deferred continuity selector.
  Supports experimental `hmm`, `adaptive_hmm`, and `joint_hmm` routing/state
  probes, but those modes are not default runtime paths.
- `evaluate_surface_selector_modes.py` - batch selector-mode evaluation.
- `evaluate_selector_router_policy.py` - offline diagnostic for routing between
  permissive Viterbi and conservative HMM/null selector outputs.
- `analyze_selector_disagreements.py` - extract frame-level disagreements
  between two selector outputs for router training, failure review, and hard
  example mining.
- `evaluate_mode_supervisor.py` - leave-one-clip-out offline supervisor probe
  that learns when to choose Viterbi vs HMM from disagreement examples, with
  optional sustained-HMM and continuous-Viterbi guardrail sweeps. This is a lab
  harness, not a promoted runtime router.
- `evaluate_lock_state_machine.py` - lock/acquisition/null state-machine
  evaluation.
- `evaluate_xy_sequence_ranker.py` - sequence-ranking utilities.

## Training and Mining

- `train_surface_xy_ranker.py` - train/check surface candidate rankers.
- `train_crop_stack_verifier.py` - offline hard-alternative target/background
  crop-stack verifier probe, including diagnostic pairwise ranking and optional
  candidate source/geometry feature modes.
- `apply_crop_stack_verifier.py` - apply a trained crop-stack verifier to
  exported top-tube rows for offline selector evaluation.
- `train_acquisition_null_ranker.py` - train/check acquisition/null filtering.
- `augment_top_tubes_alignment_features.py` - add offline candidate-local
  background-alignment (`clba_*`) columns to exported top tubes.
- `augment_top_tubes_competition_features.py` - add frame-local competitor
  normalization (`comp_*`) columns on top of CLBA/top-tube rows.
- `make_tracking_miss_review_packet.py` - generate review packets for misses
  and hard negatives.

## Labeler and Rendering

- `tube_labeling_server.py` - local/Fly labeler server for `web/tube_labeler/`.
- Rendering/demo helpers produce local files under ignored `artifacts/`.

When adding a new script, decide whether it belongs to the runtime core or the
desktop lab. Runtime-core scripts need tests, bounded runtime behavior, and Pi
bundle consideration. Desktop lab scripts can be heavier but should write only
to ignored output directories.
