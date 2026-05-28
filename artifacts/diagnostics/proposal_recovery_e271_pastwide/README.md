# Proposal Recovery Experiment


Video: `/Users/idant/Downloads/e2711620-6d4e-4f9c-8922-b1b2d1fb74f2.MP4`
Labels: `results/vision_review_next_batch/tail_extension_v1/vision_verified_next_batch_labels_v3_tail_extended.csv`
Hit threshold: center distance <= 6.0 original px

| Source | Avg cand | R@20 | R@50 | R@80 | R@120 | R@200 | R@500 | High R@80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native_dark` | 481.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.236 | 0.0 |
| `clahe_dark` | 486.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.236 | 0.0 |
| `temporal_dark` | 416.7 | 0.258 | 0.416 | 0.539 | 0.573 | 0.73 | 0.91 | 0.755 |
| `temporal_combo` | 481.0 | 0.135 | 0.348 | 0.438 | 0.494 | 0.551 | 0.787 | 0.714 |
| `temporal_halo` | 700.0 | 0.135 | 0.258 | 0.337 | 0.404 | 0.517 | 0.82 | 0.143 |
| `combined` | 1095.8 | 0.079 | 0.225 | 0.281 | 0.393 | 0.539 | 0.775 | 0.184 |

Failure sheets for `temporal_dark` top-80 are in `failure_sheets_top80/`.