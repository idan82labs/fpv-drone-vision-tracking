# Vision-Checked Surface Training v2

Purpose: keep only non-sky/surface-backed labels that survived visual review.

Included now:

- `e271` frames 631-698: main terrain/road/grass-backed continuity segment, visually checked from contact sheets and crops.
- `7bd` frames 583-588: short close surface-backed segment, useful as a small sanity sample.

Excluded for this pass:

- `1c` router `surface_backed` ranges: visually looked mostly skyline/near-horizon, so they are not clean surface training data.

Files:

- `surface_accepted_labels.csv`: promoted labels for training.
- `e271_surface_631_698_labels.csv`: e271-only contiguous training/eval slice.
- `surface_router_review_manifest.csv`: router rows with visual review status.
- `*_contact_*.jpg`: compact visual audit sheets.
