# Pi-Bounded Ground Tracking Artifact Branch

This worktree is the clean staging branch for the publishable slice of the
project:

```text
branch: paper/pi-tbd-ground-tracking-artifact
worktree: /Users/idant/fpv-drone-vision-tracking-publication
```

The main research repo remains the lab notebook. This branch is for turning the
current against-ground tracking result into a reproducible paper artifact.

## Current Artifact Focus

The first scoped artifact is the ID58 Pi-bounded source-promoter path for the
hard e271 no-sky ground segment.

Current checked claim:

```text
e271 f654-f698 no-sky ground segment
source table:                 38/45 strict, 43/45 loose
selected strict-head replay:  38/45 strict, 38/45 loose
previous runtime-selected:     0/45 strict
```

This is not a production-readiness claim. It is a bounded experimental path
that needs held-out terrain validation and real Pi camera/timing validation.

## Focused Verification

Run:

```bash
pytest tests/test_apply_true_ground_profile_mux.py \
       tests/test_srps_id58_source_promoter_pipeline.py
```

Expected current result:

```text
30 passed
```

## Scope Document

See:

```text
docs/PUBLICATION_ARTIFACT_SCOPE.md
```

That document defines what belongs in the future public artifact and what should
stay in the research repo.
