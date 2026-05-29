# Local Artifacts

`artifacts/` is for generated experiment output: videos, contact sheets, CSV
sweeps, trained models, review packets, telemetry dumps, and benchmark runs.

These files stay local by default and are ignored by Git. Promote only small,
stable summaries into `docs/` when they explain project state. If a large
artifact needs to be shared, publish it as a release asset, cloud-drive file, or
external packet instead of committing it to the source repo.

Recent local artifact families that may exist on the development machine:

- `e6_sequence_failure_analysis_v1/`
- `aaf1_surface_mining_v1/`
- `surface_selector_mode_eval_v1/`
- `pi_runtime_bundle/`
- `pi_runtime_sweep_v1/`
- `tracking_demo_sequence_batch_2026_05_29/`

Do not copy this directory into the Raspberry Pi deployment bundle.
