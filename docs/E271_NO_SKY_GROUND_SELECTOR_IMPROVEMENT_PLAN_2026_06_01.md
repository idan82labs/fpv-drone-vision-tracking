# E271 No-Sky Ground Selector Improvement Plan - 2026-06-01

## Scope

This plan targets the flagged `e2711620` segment where the drone is visible in a
tight target-local crop against terrain/vegetation/road texture, not sky.

Primary review artifacts:

- `artifacts/real_ground_no_sky_video_2026_06_01/e271_normal_speed_review/e271_ground_tight_no_sky_source_speed_f654_698_h264.mp4`
- `artifacts/real_ground_no_sky_video_2026_06_01/e271_normal_speed_review/e271_ground_tight_no_sky_source_speed_f654_698_sheet.jpg`
- `artifacts/real_ground_no_sky_video_2026_05_31/e271_ground_candidate_review/e271_real_no_sky_oracle_tight_f654_698_tight_crop_metrics.csv`

The current best evidence says this is not a proposal-availability problem.
The oracle/nearest candidate table has:

- frames `654-698`
- `44/45` strict hits
- `45/45` loose hits
- one loose-only frame at `691`

The current selected tracker does not hold the same target-local path. It sticks
to high-score wrong/background-ish tracks, especially track `6574` through
frames `665-694`, then track `9241` near the tail.

## Diagnosis

The selector gives too much authority to accumulated path score once a wrong
large-dark/background path is established. Target-local recovery currently adds
or fills candidates, but it does not reliably override an existing selected box
when that selected box is far from the recent target-local prediction.

The failure is therefore:

```text
candidate exists -> selector keeps wrong high-score identity
```

not:

```text
candidate missing -> need a bigger proposal sweep
```

## Phase 1 - Offline Target-Local Replacement Gate

Keep this offline only.

Use the new opt-in mux replacement gate in
`scripts/apply_true_ground_profile_mux.py`:

- `--target_local_recovery_replace_existing_error`
- `--target_local_recovery_replace_improvement_px`
- `--target_local_recovery_replace_min_side`
- `--target_local_recovery_replace_top_k`
- `--target_local_recovery_replace_raw_min`

Intent:

```text
If the current selected box is far from the recent target-local motion
prediction, and a current-frame top-tube candidate is much closer to that
prediction, replace the selected box.
```

Important guardrails:

- requires at least two recent emitted boxes;
- does not create acquisition from nothing;
- uses only candidate geometry, candidate score, rank, and recent selected
  history;
- does not use labels or oracle distance at runtime;
- replacement candidates should have a minimum side length so the gate does not
  snap to tiny terrain specks.
- replacement gets its own deeper `top_k` and lower raw-score floor because the
  e271 tail candidates are deep/weak, while normal gap recovery must stay
  conservative.

### Phase 1 Score Gate

Pass only if all are true:

```text
e271 f654-698:
  strict >= 35/45
  loose >= 42/45
  selected frames >= 42/45

audited true-ground packet, 7bd+e6:
  strict regression <= 1 frame versus previous_anchor12
  loose regression == 0 frames
  selected_wrong_loose_frames == 0

visual:
  no new obvious branch/terrain point lock in the generated contact sheet
```

Fail interpretation:

- If e271 improves but 7bd/e6 regress, routing/guards are too broad.
- If e271 does not improve, the replacement pool is the wrong candidate table
  or the target-local prediction is not using the right seed.

### Phase 1 Implementation Audit Requirements

The offline mux gate is not production-shaped until these are true:

```text
absolute frame age is used for recovery horizon
replacement counts are reported by clip/source/frame range
prediction error before/after is exported
runtime and offline candidate streams can be compared on the same segment
```

The first item is implemented in the mux probe. The remaining items are
required before accepting an offline score as meaningful runtime evidence.

Recommended first e271-only probe:

```bash
.venv/bin/python scripts/apply_true_ground_profile_mux.py \
  --clip e2711620-6d4e-4f9c-8922-b1b2d1fb74f2 \
  --out_dir artifacts/true_ground_profile_mux_v1/e271_replace_k80_rawneg2_emit405_v1 \
  --min_emit_score 4.05 \
  --target_local_recovery_frames 12 \
  --target_local_recovery_top_k 8 \
  --target_local_recovery_max_error 10 \
  --target_local_recovery_raw_min 5 \
  --target_local_recovery_replace_existing_error 9 \
  --target_local_recovery_replace_improvement_px 6 \
  --target_local_recovery_replace_min_side 7 \
  --target_local_recovery_replace_top_k 80 \
  --target_local_recovery_replace_raw_min -2
```

## Phase 2 - Runtime Selector Override Probe

Implement the same logic inside `tbd_motion_detector.py`, behind an explicit
flag. This must be selection-side, not just proposal-side.

Suggested flag names:

```text
--target_local_state_select
--target_local_state_select_error_px
--target_local_state_select_improvement_px
--target_local_state_select_min_side
--target_local_state_select_top_n
```

Placement:

```python
states = tbd.update(...)
baseline = tbd.best()
selected = target_local_state_override(baseline, states, target_local_seed, args)
```

This is required because the offline mux applies recovery after sequence
selection, while runtime currently adds target-local candidates before beam
selection. A replay-only win can be fake unless the runtime selector can make
the same replacement decision.

The override should consider current-hit states near the target-local seed
prediction and choose the best no-label local candidate when:

```text
baseline is missing
OR baseline center is far from predicted target-local path
OR local candidate improves prediction error by configured margin
```

Do not use it in stable clean-sky `T` unless the selected state becomes
locally implausible.

### Phase 2 Score Gate

Run on existing videos with frozen flags:

```bash
.venv/bin/python scripts/tbd_motion_detector.py \
  deploy_assets/videos/e2711620-6d4e-4f9c-8922-b1b2d1fb74f2.MP4 \
  --output_dir artifacts/e271_target_local_state_select_probe_v1 \
  --downscale 0.5 \
  --top_k_candidates 120 \
  --export_top_tubes 80 \
  --target_local_recovery_proposals \
  --target_local_state_select
```

Then evaluate frames `654-698` against:

```text
artifacts/real_ground_no_sky_video_2026_05_31/e271_ground_candidate_review/e271_real_no_sky_oracle_tight_f654_698_tight_crop_metrics.csv
```

Pass only if:

```text
e271 f654-698 selected strict >= 30/45
e271 f654-698 selected loose >= 40/45
runtime selected path does not jump to a static terrain point for >5 frames
```

Regression checks:

```text
7bd true-ground frames 583-588: no loose regression
e6 true-ground packet: strict regression <= 1 frame
d129 null windows: false boxes/min not worse than current gated baseline
new Pi clips: no reintroduction of terrain/tree false locks
```

## Phase 3 - Data Gate Before Claiming General Ground Tracking

Do not claim production-grade against-ground tracking from e271 alone. The
current corpus still lacks a long, clean, different no-sky target-against-ground
clip.

Minimum new label packet:

```text
one new no-sky terrain/vegetation/road-backed clip
>= 150 visible target frames
>= 50 null/surface frames
distributed over the whole segment
frame-level true boxes
top candidate taxonomy: T/S/E/H/G for false alternatives
```

Promotion gate:

```text
held-out no-sky ground clip:
  strict >= base + 20 percentage points
  loose >= base + 25 percentage points
  longest wrong-lock run lower than base
  false boxes/min not worse on null/surface windows
```

## Current Read

The immediate useful implementation is selector-side target-local replacement,
not another global ranker, not a new full-frame proposal branch, and not
claiming the corrected e271 oracle crop as live tracking.

The corrected e271 source-speed crop is valuable because it proves the target
can be represented continuously by the candidate pool. The production problem
is choosing that candidate path causally while suppressing the wrong
background/terrain identities.
