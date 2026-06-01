# Against-Ground Plan Implementation Audit

Date: 2026-05-31

Plan audited:

```text
docs/AGAINST_GROUND_SCORE_GATED_PLAN_2026_05_31.md
```

## Bottom Line

The implementation now covers the evidence-hygiene gate, the first true-ground
seed packet, a Phase 2 oracle-by-backdrop smoke gate, and most of the first
production-shaped routing fixes. It does **not** yet satisfy the full plan.

The important result is correct and useful: the project now refuses to count
the previously promoted aaf1 288-640 / 480-640 skyline segment as
`true_ground`. A conservative true-ground seed packet now exists, but it is
only 58 visible rows from 2 clips. That is enough to audit scoring honestly; it
is not enough to satisfy the plan's training/eval gate.

The next blocker remains two-part:

1. expand the true-ground label packet from clips where the target pixels are
   backed by vegetation, terrain, road, or mixed ground;
2. finish offline/live/runtime selector parity so future routed branch scores
   cannot be offline-only wins.

## Verification Summary

| Plan item | Status | Evidence |
| --- | --- | --- |
| Phase 0.1 backdrop correction manifest | Done | `configs/target_backdrop_corrections_2026_05_31.csv` marks aaf1 288-640 as `skyline_above_terrain` and excludes it from true-ground. |
| Phase 0.2 schema extension | Done for the ground evaluator | `scripts/evaluate_ground_profile_continuity.py` carries `target_backdrop`, `frame_context`, `audit_status`, `label_provenance`, `evidence_class`, `exclude_from_true_ground` into label and frame-eval CSVs. |
| Phase 0 true-ground filtering | Done | `--view true_ground` requires a backdrop manifest and filters to audited target-backed-by-ground rows only. |
| Phase 0 evidence_class on all promoted metrics | Done for the ground continuity evaluator | `profile_summary.csv`, `by_clip_summary.csv`, and frame-level eval rows now carry `evidence_class`, target backdrops, audit statuses, and evaluation view. The evaluator also writes `metadata.json` and `label_manifest.json` with input checksums. |
| Phase 1 true-ground label packet | Seed only / gate not met | `artifacts/true_ground_packet_2026_05_31/true_ground_labels_v1.csv` has 58 conservative true-ground rows from e6 and 7bd. The required `>=300` visible true-ground frames across `>=3` clips are not present. |
| Phase 2 oracle@K by backdrop | Smoke implemented / full gate not met | `scripts/evaluate_oracle_by_backdrop.py` evaluates oracle@K by target backdrop. On the 58-row seed, current runtime candidates reach strict@5 `56/58 = 96.55%` and strict@80 `58/58 = 100%`. The full Phase 2 gate is still blocked by Phase 1 label coverage. |
| Phase 3.1 shared surface/null-risk policy | Extracted for offline branch, runtime parity still pending | `scripts/selector_core.py` now owns `surface_gate_low_confidence` and `surface_rescue_risk`; `scripts/apply_gated_surface_branch.py` calls that shared policy. Runtime detector telemetry has not yet been wired to the same policy fixture. |
| Phase 3.2 no-backfill production replay | Done | `evaluate_xy_sequence_ranker.py` has `--no_viterbi_backfill`, propagates it into selector-core Viterbi, and records it in metadata. |
| Phase 3.3 delayed surface-ranker fallback retention / selected-no-box parity | Mostly covered at unit level | Runtime delayed selector falls back to verified-ranked candidates when surface-ranker rows are absent/empty. `raspberry_pi_runtime/verified_sequence_selector.py` now has an opt-in `--emit_no_box_rows` mode, `tests/test_pi_verified_sequence_selector.py` verifies no-backfill selected/no-box parity against `selector_core.select_viterbi_sequence`, and `tests/test_delayed_sequence_selector.py` verifies live delayed selector selected/no-box parity against the same core on a restart stream. Full artifact-level replay parity over real detector telemetry is still pending. |
| Phase 3.4 learned-logit null priors | Done | `learned_logits` now parses explicit `--null_priors` and passes them into `router_priors`. Unit test covers surface null prior behavior. |
| Phase 4 routed recentered branch v2 | Not promoted | Branch artifacts exist, but the plan requires held-out true-ground validation, which cannot happen before Phase 1/2. |
| Phase 5 multi-class observation upgrade | Existing experimental path, not promoted | JS1 learned-logit interface supports `T/S/E/H/G`, but promotion gates are not met on true-ground held-out data. |
| Phase 6 validation packet/release gate | Partial | The ground continuity evaluator emits `metadata.json` and `label_manifest.json` with checksums, and the oracle evaluator emits `oracle_summary.csv`, by-backdrop summaries, frame eval, and metadata. The full release packet still lacks timing, model checksums, failure sheets, and promotion-grade evidence class. |
| Phase 7 Pi gate | Not started for this plan | No actual Pi 5 live gate should run before Phase 4/5 pass frozen offline validation. |

## Evidence Checked

### Phase 0

Implemented:

```text
configs/target_backdrop_corrections_2026_05_31.csv
scripts/evaluate_ground_profile_continuity.py
tests/test_evaluate_ground_profile_continuity.py
```

Key verified behavior:

```text
--view true_ground
```

requires a backdrop manifest and keeps only:

```text
target_backdrop in vegetation/tree_canopy/terrain/road/mixed_ground
audit_status in visual_confirmed/contact_sheet_reviewed
visible == 1
```

Smoke outputs:

```text
artifacts/true_ground_score_gate_2026_05_31/profile_summary.csv
artifacts/core_ground_score_gate_2026_05_31/profile_summary.csv
```

Earlier result:

```text
true_ground labels: 0
core_ground legacy labels: 973
```

This was expected and proved the aaf1 skyline correction was being enforced.

Current conservative seed packet:

```text
artifacts/true_ground_packet_2026_05_31/true_ground_labels_v1.csv
artifacts/true_ground_packet_2026_05_31/profile_eval/profile_summary.csv
artifacts/true_ground_packet_2026_05_31/profile_eval/metadata.json
artifacts/true_ground_packet_2026_05_31/profile_eval/label_manifest.json
```

Result:

```text
true_ground labels: 58
clips: 2
backdrop: mixed_ground
evidence_class: conservative_true_ground_seed
best full-coverage profile: cs_js2_loco_label_frames_cont_fast
strict: 32/58 = 55.17%
loose: 34/58 = 58.62%
wrong-loose: 3/58 = 5.17%
longest strict run: 26 frames, e6 frames 1775-1800
largest strict miss gap: 22 frames, e6 frames 2278-2299
```

This is a seed benchmark, not production evidence. It misses the Phase 1
coverage gate by clip count and label count.

### Phase 2

Done:

```text
scripts/evaluate_oracle_by_backdrop.py
tests/test_evaluate_oracle_by_backdrop.py
artifacts/oracle_by_backdrop_2026_05_31/true_ground_seed_v1/
```

Seed oracle result:

```text
current_runtime true_ground seed:
  strict@1  = 35/58 = 60.34%
  strict@5  = 56/58 = 96.55%
  strict@80 = 58/58 = 100.00%
  loose@80  = 58/58 = 100.00%

surface_stack_teacher true_ground seed:
  strict@5  = 41/58 = 70.69%
  strict@80 = 57/58 = 98.28%
  loose@80  = 58/58 = 100.00%
```

Interpretation: on the current conservative seed packet, candidate availability
is not the blocker. The next blocker is selector/routing identity plus
hard-null suppression. This conclusion must be re-tested after Phase 1 expands
labels beyond the 58-row seed.

### Phase 3

Done:

```text
scripts/evaluate_xy_sequence_ranker.py --no_viterbi_backfill
tests/test_xy_sequence_ranker.py
```

Done:

```text
scripts/evaluate_explicit_state_selector.py learned_logits + --null_priors
tests/test_explicit_state_selector.py::test_learned_logits_uses_explicit_surface_null_prior
```

Partial:

```text
scripts/selector_core.py::surface_gate_low_confidence
scripts/selector_core.py::surface_rescue_risk
scripts/tbd_motion_detector.py surface ranker / delayed sequence fallback
raspberry_pi_runtime/verified_sequence_selector.py --emit_no_box_rows
tests/test_pi_verified_sequence_selector.py::test_viterbi_matches_selector_core_no_backfill_selected_no_box_parity
tests/test_delayed_sequence_selector.py::test_delayed_sequence_matches_core_no_backfill_selected_no_box_stream
```

The branch gate has repeated low-confidence plus local surface-risk tests, and
the branch script now calls shared selector-core policy functions. Runtime has
surface-ranker fallback protections. The Pi verified selector now has a
full-frame no-box output mode and a no-backfill parity test against
`selector_core`. The live delayed selector now also has a selected/no-box
parity test against `selector_core`. The remaining parity gap is artifact-level
coverage against real `tbd_motion_detector.py` delayed/live telemetry and the
gated surface-branch merged candidate stream.

## Test Results

Focused tests:

```text
.venv/bin/python -m unittest -v tests.test_evaluate_ground_profile_continuity
.venv/bin/python -m unittest -v tests.test_evaluate_oracle_by_backdrop
.venv/bin/python -m unittest -v tests.test_xy_sequence_ranker
.venv/bin/python -m unittest -v tests.test_explicit_state_selector
.venv/bin/python -m unittest -v tests.test_selector_core tests.test_apply_gated_surface_branch
.venv/bin/python -m unittest -v tests.test_pi_verified_sequence_selector tests.test_selector_core
.venv/bin/python -m unittest -v tests.test_delayed_sequence_selector tests.test_selector_core tests.test_pi_verified_sequence_selector
```

Result:

```text
8 + 3 + 6 + 24 + 14 + 14 + 24 tests passed
```

Related selector/runtime tests:

```text
.venv/bin/python -m unittest -v tests.test_selector_core tests.test_apply_gated_surface_branch tests.test_runtime_router
```

Result:

```text
32 tests passed
```

Full suite:

```text
.venv/bin/python -m unittest discover -s tests -v
```

Result:

```text
blocked by environment: this venv does not have pytest, while
tests/test_apply_true_ground_profile_mux.py imports pytest directly.

PYTHONPATH=. pytest tests/test_apply_true_ground_profile_mux.py tests/test_evaluate_oracle_by_backdrop.py -q
15 tests passed

PYTHONPATH=. pytest -q
blocked by environment: system pytest can import pytest but lacks cv2, while
the repo venv has cv2.
```

## Honest Gaps

1. There is currently no validated true-ground training/eval set.
   The 58-row true-ground seed packet is useful for smoke scoring, but it is
   far below the `>=300` visible rows / `>=3` clips gate.

2. The project still has legacy `core_ground` numbers based on `bg_split`.
   These can be useful regression views, but they cannot be called
   against-ground performance.

3. Surface-risk gating is now shared between selector-core and offline branch
   replay, but runtime detector telemetry has not yet been parity-tested
   against it.

4. The no-backfill selector option exists and the Pi verified selector now has
   an opt-in full-frame no-box output mode. The remaining score gates still
   need to be rerun in production-shaped no-backfill mode on the expanded
   true-ground packet.

5. Delayed surface-ranker fallback retention appears protected in runtime code,
   but still lacks the exact live-runtime plan gate:

```text
production replay vs tbd_motion_detector delayed/live telemetry selected/no-box match: 100%
```

6. Phase 4/5 cannot be honestly evaluated until Phase 1/2 are done.

## Next Gate To Run

Do not tune aaf1 as true-ground. The next implementation step should be:

1. Expand the true-ground label packet from e6/7bd/59e/b96/new Pi clips.
2. Require every row to include `target_backdrop`, `audit_status`,
   `frame_context`, and `evidence_class`.
3. Rerun:

```text
.venv/bin/python scripts/evaluate_ground_profile_continuity.py \
  --view true_ground \
  --out_dir artifacts/true_ground_score_gate_<date>
```

4. Only if there are enough labels, run oracle@K by backdrop.
5. Only if oracle@K is high, run routed recentered branch v2 and multi-class
   JS1 replay.

Until then, any "against-ground score" is not production evidence.
