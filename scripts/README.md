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
  Supports experimental `adaptive_hmm` routing, but that mode is not a default
  runtime path.
- `evaluate_surface_selector_modes.py` - batch selector-mode evaluation.
- `evaluate_selector_router_policy.py` - offline diagnostic for routing between
  permissive Viterbi and conservative HMM/null selector outputs.
- `analyze_selector_disagreements.py` - extract frame-level disagreements
  between two selector outputs for router training, failure review, and hard
  example mining.
- `evaluate_lock_state_machine.py` - lock/acquisition/null state-machine
  evaluation.
- `evaluate_xy_sequence_ranker.py` - sequence-ranking utilities.

## Training and Mining

- `train_surface_xy_ranker.py` - train/check surface candidate rankers.
- `train_acquisition_null_ranker.py` - train/check acquisition/null filtering.
- `make_tracking_miss_review_packet.py` - generate review packets for misses
  and hard negatives.

## Labeler and Rendering

- `tube_labeling_server.py` - local/Fly labeler server for `web/tube_labeler/`.
- Rendering/demo helpers produce local files under ignored `artifacts/`.

When adding a new script, decide whether it belongs to the runtime core or the
desktop lab. Runtime-core scripts need tests, bounded runtime behavior, and Pi
bundle consideration. Desktop lab scripts can be heavier but should write only
to ignored output directories.
