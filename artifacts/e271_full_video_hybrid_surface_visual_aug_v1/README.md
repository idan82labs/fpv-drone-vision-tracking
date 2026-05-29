# e271 Hybrid Surface Visual Augmentation v1

This experiment used `scripts/augment_top_tubes_visual_features.py` to add offline visual stack features to the e271 hybrid surface `top_tubes.csv`.

The generated augmented `top_tubes.csv` was about 25 MB and is intentionally not committed. Regenerate it with:

```bash
python scripts/augment_top_tubes_visual_features.py \
  --video /Users/idant/Downloads/e2711620-6d4e-4f9c-8922-b1b2d1fb74f2.MP4 \
  --top_tubes /Users/idant/Drone-Strike/results/hybrid_surface_v1/e271_large_dark_hybrid_wrapped/e2711620-6d4e-4f9c-8922-b1b2d1fb74f2/top_tubes.csv \
  --out artifacts/e271_full_video_hybrid_surface_visual_aug_v1/top_tubes.csv \
  --downscale 0.5 \
  --max_rank 80 \
  --offsets -12,-9,-7,-5,-3,-2,-1 \
  --model auto
```

The downstream compact results are stored in:

- `artifacts/e271_full_video_hybrid_surface_visual_aug_xy_ranker_v1`
- `artifacts/e271_full_video_hybrid_surface_visual_aug_sequence_ranker_logistic_v1`
- `artifacts/e271_full_video_hybrid_surface_visual_aug_sequence_ranker_sizejump_v1`
