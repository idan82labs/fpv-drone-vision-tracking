# Data and Artifacts

The source repository should stay small and reproducible. Generated data is
kept locally by default.

## Tracked

Tracked files should be:

- source code;
- tests;
- documentation;
- small config examples;
- small manifests or summaries that explain a reproducible result.

## Ignored Local Directories

`artifacts/`

- Generated experiment output: videos, contact sheets, CSV sweeps, telemetry,
  review packets, benchmark directories, trained model dumps, and temporary
  reports.
- Keep large files local or publish them externally.
- Promote only concise conclusions into `docs/STATUS.md`.

`deploy_assets/`

- Local labeling-server inputs: review CSVs, crops, overviews, compressed review
  videos, backups, and Fly volume seed data.
- These are working assets, not source.

`models/`

- Promoted model files and calibration assets that may be used by runtime
  bundles later.
- The files are ignored unless deliberately published outside Git or added with
  an explicit release process.

`raw_videos/`, `downloads/`, `results/`, `tmp/`, `scratch/`

- Local-only footage, downloads, and intermediate work.

## Label Strength

Use label sources carefully:

- Human-reviewed target and hard-negative labels are strongest.
- Dense continuity labels are useful but should keep their provenance.
- `vision_assisted` labels are model/human-assisted guidance, not strict truth.
- `vision_assisted_gapfill` rows are weakest and need manual review before
  being treated as strict ground truth.

## Promotion Rules

Before a local artifact affects the production/Pi path:

1. Summarize the evidence in `docs/STATUS.md`.
2. Include held-out or leave-one-clip-out behavior when applicable.
3. Include null/no-target behavior.
4. Include runtime timing with p95/p99/max tails for deployment-relevant modes.
5. Record the source artifact path and model/checksum if a model is promoted.

## Sharing Large Outputs

Use one of these instead of committing large files:

- GitHub release assets;
- iCloud/Drive packet;
- Fly volume seed data for labeler deployments;
- local `artifacts/` paths referenced from docs.

The Pi bundle tool intentionally excludes `artifacts/`, `deploy_assets/`, raw
videos, review packets, and training outputs.
