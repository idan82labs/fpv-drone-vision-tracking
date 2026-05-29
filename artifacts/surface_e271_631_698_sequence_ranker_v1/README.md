# e271 Surface Sequence Ranker v1

Purpose: test whether learned candidate scores plus a simple continuity/Viterbi
selector can stop surface-texture jumps on the visually checked e271 terrain
segment.

Input:

- Labels: `artifacts/surface_training_v2_vision_checked/e271_surface_631_698_labels.csv`
- Candidates: `/Users/idant/Drone-Strike/results/surface_ranker_top_tubes_v1/e2711620-6d4e-4f9c-8922-b1b2d1fb74f2/top_tubes.csv`
- Frames: e271 631-698, 68 visible target frames

Result:

- Top-tube oracle: 68/68 strict.
- Baseline `verified_score`: 20/68 strict.
- OOF logistic framewise ranker: 64/68 strict.
- OOF logistic + Viterbi continuity: 68/68 strict.

Best swept config:

- `max_jump_px=8`
- `transition_weight=0.05`

Caveat: this is a single-segment continuity result, not deployment proof. It is
strong evidence that surface-mode selection needs sequence continuity over
learned scores before promotion into the live detector.
