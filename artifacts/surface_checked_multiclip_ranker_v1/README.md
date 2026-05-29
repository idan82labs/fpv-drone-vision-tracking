# Surface XY Tube Ranker

Leave-one-clip-out evaluation for the strict vision-checked surface labels.

Best LOCO model: `hist_gbdt`

Important caveat: this is not a deployable model. The set has 68 e271 frames
and only 6 7bd frames, so LOCO is underdetermined and performs badly:
baseline strict is 24/74, while the learned LOCO models are 0-5/74 strict.
Keep this artifact as negative evidence that we need another real surface clip
before claiming cross-clip generalization.

See `loco_summary.csv`, `loco_predictions.csv`, and `metadata.json`.
