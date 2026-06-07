# Publication Artifact Scope

Branch: `paper/pi-tbd-ground-tracking-artifact`

This branch is the staging area for a slim, reproducible artifact around the
Pi-bounded tiny-drone track-before-detect work. The research repository remains
the lab notebook; this branch should only keep code, configs, tests, and
documentation that support a defensible paper artifact.

## Core Claim Under Audit

A monocular, no-ML-in-the-core track-before-detect pipeline can recover much of
the against-ground offline/deferred performance with bounded candidate source
promotion suitable for Pi-class deployment.

Current hard-segment evidence under the experimental ID58 path:

```text
e271 f654-f698 no-sky ground segment
source table:                 38/45 strict, 43/45 loose
selected strict-head replay:  38/45 strict, 38/45 loose
previous runtime-selected:     0/45 strict
```

The branch should not claim production readiness until real Pi camera ingest,
latency, and held-out terrain clips are validated.

## Publishable Slice

Keep and polish:

```text
raspberry_pi_runtime/
scripts/motion_detector_v2.py
scripts/tbd_motion_detector.py
scripts/apply_true_ground_profile_mux.py
scripts/audit_srps_id58_source_promoter_provenance.py
scripts/build_srps_id58_runtime_source_promoter_dataset.py
scripts/train_srps_id58_runtime_source_promoter.py
tests/test_apply_true_ground_profile_mux.py
tests/test_srps_id58_source_promoter_pipeline.py
docs/SRPS_ID58_PI_BOUNDED_SOURCE_PROMOTER_DISTILLATION_REAL_TRAINING_PLAN_2026_06_07.md
```

Potentially keep after audit:

```text
scripts/render_against_ground_status_clip.py
scripts/evaluate_tracking_run.py
scripts/selector_core.py
models/README.md
deploy_assets/README.md
```

## Lab-Only Material

Do not promote directly:

```text
bulk SRPS plan backlog
professor dropoff packets
giant artifacts and rendered review sheets
all-fit/interleaved-only training results
prototype gates without held-out validation
scripts that depend on private absolute paths without a manifest
```

## Artifact Requirements

Before this becomes a public repo or paper artifact, it needs:

1. Frozen commands for the ID58 source-promoter dataset, training, replay, and
   focused tests.
2. A compact sample-data manifest or downloadable artifact manifest.
3. A baseline table: previous runtime, offline/deferred sequence, ID58 bounded
   source-promoter path.
4. Ablations for registration, source promotion, temporal selection, and force
   release behavior.
5. Timing table for development machine and Pi/Pi-like bounded runtime.
6. Clear limitations: not production-ready, not closed-loop flight validated,
   and no claim beyond the tested hard segment until held-out clips pass.

## Next Cleanup Pass

1. Commit this branch as a staging branch.
2. Reduce docs to paper-facing docs only.
3. Move lab scripts out of the eventual public repo, or keep them in a
   `research/` appendix if needed for reproducibility.
4. Replace absolute-path assumptions with manifest-driven paths.
5. Build a small reproducibility package rather than publishing the full
   artifact tree.
