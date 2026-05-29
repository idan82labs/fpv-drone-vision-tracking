# Data and Artifacts

This repository includes only curated data.

## Included

- `deploy_assets/tube_hard_negative_review_packet_thr060_top8/`
  - Review packet used by the labeling website.
- `deploy_assets/videos/`
  - Small compressed review clips used by the labeler.
- `artifacts/labels/`
  - Current curated CSV labels and weak vision-assisted labels.
- `artifacts/metrics/`
  - Cross-validation summaries, threshold sweeps, and proposal-recovery summaries.
- `artifacts/current_demo/`
  - Selected demo videos and contact sheets.
- `artifacts/diagnostics/`
  - A few diagnostic sheets used to justify vision-assisted gap filling.
- `artifacts/state_machine_selector_d129_v1/`
  - Complete-video d129 acquisition/null selector sweep and its input CSVs.
- `artifacts/full_video_oof_state_eval_d129_v2/`
  - Reproducible full-video d129 out-of-fold candidate scores plus
    acquire/track/null state-machine sweeps.
- `artifacts/surface_training_v2_vision_checked/`
  - Strictly promoted non-sky/surface-backed labels from visual review. This is
    intentionally smaller than the router `surface_backed` set.
- `artifacts/surface_e271_631_698_xy_ranker_v1/`
  - e271 terrain-tail candidate-ranker cross-validation artifact.
- `artifacts/surface_e271_631_698_sequence_ranker_v1/`
  - e271 terrain-tail learned-score plus continuity/Viterbi selector artifact.
- `artifacts/runtime_e271_router_surface_tail_v1/`
  - e271 full-clip runtime benchmark for baseline, auto router, and forced
    surface mode.

## Excluded

- Full local experiment sweeps under `results/`.
- Virtual environments and caches.
- Local Fly password files.
- Uncurated screenshots, temporary frames, and Playwright logs.

## Label Strength

Use label sources carefully:

- `target` or reviewed hard-negative labels: strongest labels.
- `vision_assisted`: useful but weaker; use for model guidance and visual checking.
- `vision_assisted_gapfill`: weakest; these rows were filled after visual inspection and should be manually reviewed before being treated as strict ground truth.

## Current Demo Caveat

`artifacts/current_demo/e271_vision_assisted_clean_9p0-12p8s_v2.mp4` is intended for explaining the desired target track. It is not a claim that the autonomous detector currently holds that lock.
