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

