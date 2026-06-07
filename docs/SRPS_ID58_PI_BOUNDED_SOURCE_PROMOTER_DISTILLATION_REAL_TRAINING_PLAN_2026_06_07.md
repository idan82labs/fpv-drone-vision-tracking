# SRPS ID58 Pi-Bounded Source Promoter Distillation Real Training Plan

## Summary

ID57 produced the first meaningful selected-output recovery on the e271 no-sky ground segment, but only with diagnostic source parity:

```text
baseline/current runtime-shaped selected replay:
  strict: 0/45
  loose:  0/45

ID57 external source-parity hybrid:
  strict: 31/45
  loose:  43/45
```

This is a real capability jump, but it is not Pi-ready. The passing branch uses:

```text
artifacts/srps_id51_balanced_source_label_packet_full_aaf1_v1/review_candidates.csv
```

as an external bridge. That file is a review/source-promoter artifact, not a live runtime source.

The next work is therefore not more switch-threshold tuning. ID58 must distill the external/source-promoter candidate source into a bounded, Pi-shaped runtime candidate materializer.

## Decision

Build:

```text
ID58:
  Pi-bounded source-promoter materializer
  trained from external/source-parity positives
  evaluated by candidate availability and selected replay
  integrated behind flags only
```

Professor guidance is not needed before starting. Use the existing ID56 dropoff only if ID58 fails in a contradictory way, for example if the same source candidates are present in a Pi-computable row table but cannot be materialized without a broad K explosion.

## Implementation Status: 2026-06-07

Implemented.

New scripts:

```text
scripts/audit_srps_id58_source_promoter_provenance.py
scripts/build_srps_id58_runtime_source_promoter_dataset.py
scripts/train_srps_id58_runtime_source_promoter.py
```

Mux integration:

```text
scripts/apply_true_ground_profile_mux.py
  --target_source_promoter_model
  --target_source_promoter_threshold
  --target_source_promoter_source_table
  --target_source_promoter_top_k
  --target_source_promoter_trace
```

Focused tests:

```text
tests/test_srps_id58_source_promoter_pipeline.py
tests/test_apply_true_ground_profile_mux.py
```

Verification:

```text
30 passed
```

Real artifacts:

```text
artifacts/srps_id58_source_promoter_provenance_v1/
artifacts/srps_id58_runtime_source_promoter_dataset_v1/
artifacts/srps_id58_runtime_source_promoter_loose_v1/
artifacts/srps_id58_runtime_source_promoter_strict_v1/
artifacts/srps_id58_source_promoter_replay_f654_698_v3/
artifacts/srps_id58_source_promoter_replay_f654_698_eval_v3/
artifacts/srps_id58_source_promoter_replay_f654_698_loose_v2/
artifacts/srps_id58_source_promoter_replay_f654_698_loose_eval_v2/
```

Key results:

```text
Phase A provenance:
  external positive rows: 80
  matched Pi-computable rows: 80/80
  matched runtime rows before ID58 materializer: 1/80
  gate: pass

Phase B runtime source table:
  focus frames: 45
  strict source availability: 38/45
  loose source availability: 43/45
  leakage columns: 0
  gate: pass

Phase C source-promoter heads:
  strict head source top1: 38/45 strict, 43/45 loose
  loose head source top1: 31/45 strict, 43/45 loose
  both retain the two-frame loose gap because the source table has only 43/45 loose availability.

Selected replay:
  strict head + corrected release handoff:
    38/45 strict
    38/45 loose

  loose head + corrected release handoff:
    26/45 strict
    43/45 loose
```

Important implementation correction:

```text
The first selected replay failed at 1/45 because `is_external_positive_source`
accidentally entered the trainable feature schema. That provenance-only field is
now forbidden. The corrected model still reaches 38/45 strict source top1 and
the selected strict-head replay reaches 38/45 strict.
```

Current read:

```text
ID58 converts the previous 0/45 runtime-shaped selected output into 38/45 strict
on the e271 f654-f698 no-sky ground segment using Pi-computable source rows.
It is a meaningful capability jump.

It does not meet the old 44/45 loose selected gate because this specific source
table has only 43/45 loose-positive frames. Reaching 44/45+ loose now requires
candidate-source recovery for the remaining two loose-missing frames, not another
handoff threshold.
```

## Findings ID58 Builds On

### ID57 Runtime/Top-Tube Source Still Fails

Artifact:

```text
artifacts/srps_id57_runtime_reacquisition_rows_f654_698_v1/
```

Result:

```text
runtime alternatives:
  strict available: 0/45
  loose available:  0/45

top-tube source bridge:
  strict available: 9/45
  loose available:  12/45

phase_a_gate_pass: 0
```

### ID57 External Source Parity Works

Same artifact with external bridge:

```text
external source bridge:
  strict available: 38/45
  loose available:  43/45

phase_b_gate_pass: 1
```

Runtime switch model on those candidates:

```text
artifact:
  artifacts/srps_id57_runtime_switch_model_f654_698_external_v1

best model:
  extratrees

frame_top1_target: 43/45
frame_top3_target: 43/45
longest_wrong_switch_run: 0
phase_d_runtime_switch_gate_pass: 1
```

Selected replay:

```text
loose model:
  strict: 25/45
  loose:  43/45

strict model:
  strict: 30/45
  loose:  32/45

source/action hybrid:
  strict: 31/45
  loose:  43/45
```

Interpretation:

```text
The handoff/switch classifier can work.
The missing piece is the runtime candidate source that exposes the same family of candidates.
```

## Phase A: Source Provenance Audit

### Goal

Identify exactly which upstream candidate generator produced the external source-parity positives.

### New Script

```text
scripts/audit_srps_id58_source_promoter_provenance.py
```

### Inputs

```text
artifacts/srps_id51_balanced_source_label_packet_full_aaf1_v1/review_candidates.csv
artifacts/srps_id50_pi_source_distillation_rows_topk_export80_v1/candidate_rows.csv
artifacts/srps_id52_multiclass_source_promoter_bootstrap_full_aaf1_v1/prediction_rows.csv
artifacts/srps_id57_runtime_reacquisition_rows_f654_698_v1/runtime_reacquisition_rows.csv
```

### Outputs

```text
artifacts/srps_id58_source_promoter_provenance_v1/
  provenance_matches.csv
  positive_source_summary.csv
  missing_runtime_source_summary.csv
  README.md
```

### Required Report

For every external positive in f654-f698:

```text
frame
candidate_id
source_family
source
rank
box
strict/loose label
which upstream table contains this candidate
which runtime table does not contain it
candidate generator/provenance fields
Pi-computable feature availability
```

### Phase A Gate

Pass only if:

```text
>= 90% of external strict/loose positives can be traced to a Pi-computable upstream candidate table.
>= 90% have no trainable leakage fields required for reproduction.
top missing source families are identified.
```

If this fails, ID58 cannot proceed as a source-promoter distillation task. We need proposal generation work instead.

## Phase B: Runtime Source-Promoter Training Dataset

### Goal

Train on candidate rows that can exist at runtime, not on review-packet-only rows.

### New Script

```text
scripts/build_srps_id58_runtime_source_promoter_dataset.py
```

### Inputs

```text
provenance_matches.csv
candidate_rows.csv from the Pi-computable source table
e271 f654-f698 labels in original detector coordinates
aaf1/e6/d129 regression source-label packets
ID57 external bridge selected-output misses/wins
```

### Labels

For each candidate:

```text
PROMOTE_STRICT   center <= 8 px
PROMOTE_LOOSE    center <= 16 px
REJECT_WRONG     center > 24 px
AMBIGUOUS_NEAR   16-24 px, ignored for training
NO_CANDIDATE     frame has no safe candidate in this source
```

### Required Row Families

```text
e271 f654-f698:
  all candidates from the traced source table
  all external source-parity positives
  current runtime/top-tube negatives

aaf1:
  ground/terrain false locks
  target-like positives when available

e6:
  branch/terrain regression positives and safe current cases

d129:
  no-target / null hard negatives
```

### Trainable Features

Allowed:

```text
rank
score
verified_score
source/source_family
box size/aspect/area
local contrast / dark compactness if present
large_dark / map / appearance provenance
source recurrence count
crop/logit scores available at runtime
router/background bucket
motion residuals available causally
```

Forbidden:

```text
true box coordinates
distance_to_true
strict/loose flags as features
candidate_role such as nearest_gt
review_role
review_reason
manual review notes
future-frame features
external bridge label provenance
```

### Outputs

```text
artifacts/srps_id58_runtime_source_promoter_dataset_v1/
  source_promoter_rows.csv
  frame_candidate_availability.csv
  trainable_feature_columns.json
  summary.json
  README.md
```

### Phase B Gate

Pass only if:

```text
e271 f654-f698:
  strict candidate availability in source rows >= 35/45
  loose candidate availability in source rows  >= 42/45

regression data:
  aaf1/e6/d129 rows present
  no-target/hard-negative rows >= positive rows

leakage audit:
  trainable leakage columns = 0
```

If source rows cannot expose `>= 42/45` loose candidates, do not train. The source table is not sufficient.

## Phase C: Train Pi-Bounded Source Promoter

### New Script

```text
scripts/train_srps_id58_runtime_source_promoter.py
```

### Models

Start small and Pi-feasible:

```text
logistic regression
HistGradientBoosting
ExtraTrees for offline diagnostic only
```

The production candidate should be a small model that can be ported or approximated:

```text
logistic / shallow tree rules / compact GBDT
```

Do not start with CNN/TCN here. ID57 already showed the issue is candidate source materialization, not visual switch scoring.

### Training Targets

Train two heads or two binary models:

```text
loose source promoter:
  target = PROMOTE_STRICT or PROMOTE_LOOSE

strict source promoter:
  target = PROMOTE_STRICT
```

Use blocked validation:

```text
e271 blocks:
  654-668
  669-683
  684-698

regression:
  aaf1/e6/d129 held-out blocks
```

### Metrics

```text
frame top1 strict/loose
frame top3 strict/loose
candidate precision/recall
wrong promotion rejection
longest wrong promotion run
source family confusion
background bucket confusion
```

### Phase C Gate

Loose promoter:

```text
e271 blocked:
  frame top1 loose >= 38/45
  frame top3 loose >= 42/45
  longest wrong promotion run <= 1
```

Strict promoter:

```text
e271 blocked:
  frame top1 strict >= 25/45
  frame top3 strict >= 35/45
```

Regression:

```text
d129 no-target wrong promotion rejection >= 95%
e6 wrong promotion run <= 1
aaf1 no new long terrain-lock pattern
```

If loose promoter cannot pass, do not integrate. If strict fails but loose passes, integrate loose only and use recenter/tightening later.

## Phase D: Runtime Materializer Integration

### Mux Changes

Extend `apply_true_ground_profile_mux.py` with a production-shaped source promoter, separate from the diagnostic external CSV bridge.

Flags:

```text
--target_source_promoter_model
--target_source_promoter_threshold
--target_source_promoter_top_k
--target_source_promoter_source_table
--target_source_promoter_trace
```

Behavior:

```text
1. Build candidate rows from the traced Pi-computable source table.
2. Score with the ID58 source promoter.
3. Materialize only top_k candidates per frame.
4. Add them as source_promoter candidates for ID57 reacquisition scoring.
5. Do not globally widen detector K.
6. Do not use external review_candidates.csv in production-shaped runs.
```

Eligibility:

```text
only active when:
  current track is released or absent
  router/background is ground/surface/mixed/unknown
  current source confidence is low or wrong-source-prone
```

Initial diagnostic setting:

```text
top_k = 5
```

Production-shaped setting:

```text
top_k <= 3
```

### Trace

```text
target_source_promoter_trace.csv
  frame
  source_candidate_count
  promoted_count
  best_promoter_score
  best_candidate_box
  best_candidate_source
  materialized
```

### Phase D Gate

Source materialization:

```text
e271 f654-f698:
  materialized strict availability >= 30/45
  materialized loose availability  >= 42/45
```

If materialized loose availability is below `42/45`, stop. The promoter is not exposing enough candidates.

## Phase E: Selected Replay With ID57 Switch

### Replay Config

Use:

```text
ID58 source promoter materializer
ID57 loose runtime switch model
ID57 hybrid strict tightening only if both loose/strict outputs are source-promoter reacquisitions
current release policy:
  diagnostic release-all first
  then trained current-trust only after positive trust rows exist
```

### Required Outputs

```text
artifacts/srps_id58_source_promoter_replay_f654_698_v1/
  sequence_selected_tracks.csv
  target_source_promoter_trace.csv
  target_reacquisition_trace.csv
  summary.json
  visible_strict_misses.csv
  worst_false_locks.csv
```

### Phase E Gate

Minimum:

```text
e271 f654-f698:
  strict >= 25/45
  loose  >= 42/45
```

Target:

```text
e271 f654-f698:
  strict >= 31/45
  loose  >= 43/45
```

Regression:

```text
e6 strict regression <= 1 pp
d129 false boxes/min does not increase materially
aaf1 no new long wrong-lock run
easy/sky clips promoter gate rate <= 3%
```

Do not claim production readiness until regression clips pass.

## Phase F: Current Trust Real Training

ID57 could not train a real current-trust model from f654-f698 because every current row was `RELEASE_CURRENT`:

```text
current_label_counts:
  RELEASE_CURRENT: 45
```

ID58 should not fake this. Build a separate current-trust dataset only after collecting rows where current output is genuinely correct.

### New Packet

```text
artifacts/srps_id58_current_trust_training_rows_v1/
```

Required positives:

```text
e6/e271/easy/sky frames where current selected output is strict/loose correct
stable T-state or selected track should be retained
```

Required negatives:

```text
e271 f654-f698 wrong current rows
aaf1 terrain locks
d129 no-target false current rows
```

Gate:

```text
positive TRUST_CURRENT rows >= 200
negative RELEASE_CURRENT rows >= 200
blocked release recall >= 80%
blocked trust recall >= 85%
trusted wrong current <= 1 on f654-f698
```

Until this passes, production-shaped source-promoter replay should use conservative release rules, not a learned current-trust model.

## Real Training Discipline

Every ID58 artifact must include:

```text
trainable_feature_columns.json
leakage audit
blocked-validation metrics
selected replay metrics
exact command/config metadata
```

Do not promote:

```text
all-fit results
interleaved-only folds
external review_candidates.csv as a live source
threshold sweeps without candidate availability gain
```

## Stop Rules

### Stop ID58 Source Distillation

Stop if:

```text
external positives cannot be traced to Pi-computable upstream rows.
```

Then the next task is proposal/source generation, not model training.

### Stop Selector Work

Stop selector/switch tuning if:

```text
source promoter materialized loose availability < 42/45.
```

The selector cannot pick candidates it does not see.

### Stop Scalar Source Promoter

Stop scalar/tree training if:

```text
source rows expose candidates,
but blocked frame top1/top3 target metrics fail.
```

Then add crop-stack/tensor logits as source-promoter features.

### Ask Professor

Ask professor only if:

```text
the source provenance audit says the external positives are Pi-computable,
but the runtime source promoter cannot materialize them with bounded top_k.
```

Package should include:

```text
ID57 selected replay summaries
ID58 provenance_matches.csv
ID58 source materialization summary
source-positive frame cards
trace snippets for missing frames
```

## Expected Outcome

ID58 should not be judged by candidate AUC. It should be judged by source materialization and selected replay.

The first meaningful production-shaped win is:

```text
e271 f654-f698:
  strict >= 25/45
  loose  >= 42/45
```

The target is to reproduce the diagnostic source-parity result without using the external review packet as a live source:

```text
strict >= 31/45
loose  >= 43/45
```

If ID58 reaches that while preserving e6/d129/easy regressions, then the ground-backed path is meaningfully closer to Pi production.
