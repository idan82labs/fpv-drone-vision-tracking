# Architecture

The current system has four layers.

## 1. Motion Proposal Generation

`scripts/motion_detector_v2.py` and `scripts/tbd_motion_detector.py` generate candidate boxes from frame-to-frame motion residuals, compact dark/blob cues, multiscale map evidence, and native-resolution micro candidates.

The goal of this layer is high recall. It is allowed to generate clutter as long as the true drone is present in the candidate pool.

## 2. Track-Before-Detect Beam Search

`scripts/tbd_motion_detector.py` keeps a short-window beam of candidate tubes. The transition model uses velocity, acceleration, misses, pair evidence, and candidate scores to keep plausible paths alive.

This is the correct architecture for the present data because many real targets are only a few pixels wide and are not separable from clutter in a single frame.

## 3. Tube Verifier / Ranker

The current learning direction is tube-level ranking over exported `top_tubes.csv` alternatives.

Relevant scripts:

- `scripts/train_tube_verifier_sklearn.py`
- `scripts/apply_sklearn_tube_verifier.py`
- `scripts/train_xy_tube_ranker.py`
- `scripts/augment_top_tubes_visual_features.py`
- `scripts/calibrate_learned_tube_verifier.py`

The strongest next direction is not another broad threshold sweep. It is target-aligned versus background-aligned visual evidence, hard-negative mining, and null/clutter calibration.

## 4. Labeling and Review

`scripts/tube_labeling_server.py` serves `web/tube_labeler/`. The UI supports:

- video plus candidate overview,
- selected candidate crop,
- candidate labels,
- frame-level target marking,
- visible/not-visible frame state,
- notes,
- CSV persistence with backups.

## Known Failure Modes

- Persistent terrain, tree, cloud, or skyline points win the tube score.
- Near-stationary drones collapse the motion evidence and become a tiny-object recognition problem.
- Broad silhouette rescue improves recall but revives clutter.
- More proposal candidates improve oracle recall, but can make autonomous selection worse without a better ranker.

