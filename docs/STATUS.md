# Status

## Honest Read

The system has made progress, but the remaining problem is still hard.

Current capability:

- Proposals can often include the visible drone.
- Temporal-stack proposal recovery improved oracle recall on the hard e271 tail.
- The new `large_dark` proposal path recovers the close, visibly centered e271 gap that was absent from the v3 reel.
- The labeler and review workflow now support hard-negative mining and frame-level drone marking.

Current blocker:

- The autonomous ranker still picks plausible clutter too often.
- More candidate count alone is not a reliable improvement.
- Surface backgrounds need their own benchmark; clean-sky numbers hide the tree/grass/terrain failure mode.

Repo/runtime cleanup update:

- Generated artifacts, review videos, crop sheets, model dumps, and Fly labeler
  seed data are now local/ignored rather than source-controlled.
- `docs/REPO_LAYOUT.md` defines the split between the desktop lab and the
  Raspberry Pi runtime path.
- The production/Pi path remains a bounded Python/OpenCV runtime scaffold, not
  a separate Rust rewrite.
- This cleanup improves repo hygiene and deployment boundaries. It does not
  change the honest tracker capability rating.

## Recent Numbers

From the v3 tail-extended labels:

- Baseline strict: 465/677 = 68.7%
- Baseline loose: 576/677 = 85.1%
- Logistic strict: 477/677 = 70.5%
- Logistic loose: 594/677 = 87.7%
- ExtraTrees strict: 491/677 = 72.5%
- ExtraTrees loose: 587/677 = 86.7%

From the v4 e271 proposal-recovery diagnostic:

- Temporal-stack proposal layer, best standalone source: `temporal_dark`
- R@80 = 0.539 overall
- R@200 = 0.730 overall
- R@500 = 0.910 overall
- High-confidence R@80 = 0.755

Interpretation: the target is often present somewhere in the proposal pool, but the ranker does not reliably select it.

From the e271 reel-gap investigation:

- v3 reel seconds 25-36 mapped to e271 frames about 23-573.
- v3 e271 selection rows only covered frames 594-698, leaving a visible-drone hole.
- Existing temporal-stack top-tubes on the new dense labels: oracle@100 = 49.0% all labels, 40.9% high-confidence labels.
- New `large_dark` proposal run: oracle@100 = 92.8% all labels, 85.4% high-confidence labels.
- ExtraTrees diagnostic ranker on `large_dark` top-tubes: 77.4% strict / 79.6% loose on all labels, 83.3% strict / 85.1% loose on high-confidence labels.

Interpretation: the immediate failure was partly data coverage and partly proposal coverage. `large_dark` is the correct short-term recovery primitive for close, clear dark silhouettes, but it still needs leave-one-clip-out validation and review of low-confidence gap labels.

From the surface-background harness:

- Pair-rescue/gapfix split audit: clean sky strict = 40/40, textured/non-sky strict = 679/1135 = 59.8%.
- In that same textured/non-sky split, oracle@80 = 1008/1135 = 88.8%, so many failures are ranking failures rather than pure proposal failures.
- A first surface ranker improves the next-batch/e271-like distribution, but the combined v2 model with aaf1 does not beat baseline under leave-one-clip-out.

Interpretation: do not blindly replace the selector with the surface ranker yet. Keep the harness, collect more true tree/grass labels, and use a background-conditioned selector.

From the hybrid proposal/coast experiment:

- Added experimental `--hybrid_coast_proposals` and `--scenario_balance` flags.
- On aaf1 textured/non-sky frames, strict improved from 86.7% to 93.3%.
- On aaf1 clean sky frames, strict dropped from 98.9% to 93.2%.
- On e271 large-dark textured/non-sky frames, strict dropped from 59.9% to 57.5%.
- Runtime also worsened materially.

Interpretation: keep hybrid as an explicit surface-mode experiment only. It is not ready as a global default.

From the first GPT vision surface-label expansion:

- Added `scripts/make_surface_continuity_packet.py` to build continuous non-sky review packets with full-frame and crop sheets.
- Added `scripts/profile_target_background_router.py` to split immediate target background from wider scene context.
- Visually reviewed aaf1/e6/1c/7bd/529/b96/59e/d129 packets.
- Important correction: the automatic `textured_non_sky` split is over-broad. Many "textured" labels are still sky/cloud/horizon-backed, not true tree/grass/terrain-backed.
- Promoted 23 high-confidence b96/59e horizon-surface labels.
- Held out d129 tree-line candidates for human review because visual confidence was below 0.8.
- Apples-to-apples LOCO against `surface_ranker_top_tubes_v1`: baseline strict = 80.4%, best learned strict = 79.4%.
- A conservative state-machine selector, using ExtraTrees only for `surface_backed` router frames and baseline elsewhere, improved strict to 81.0% and loose to 88.6%.

Interpretation: the new labels improve coverage, but they do not justify a global learned selector. A state-conditioned selector is more promising: learned ranking helps true surface-backed frames and hurts skyline-adjacent sky frames.

From the multiclip surface v4 fallback sweep:

- Added `artifacts/surface_training_v4_multiclip/` with high/medium-high
  textured/non-sky rows from `1c`, `529`, `59e`, `7bd`, `aaf1`, `b96`, `d129`,
  and `e271`.
- Direct LOCO models still do not justify global replacement:
  - baseline `verified_score`: 80.0% strict, 85.7% loose.
  - ExtraTrees: 80.4% strict, 87.7% loose.
- A conservative fallback policy is materially better:
  - use ExtraTrees only when learned score >= 0.76, otherwise baseline.
  - result: 83.3% strict, 90.1% loose over 1,030 evaluated frames.
  - learned model used on 71.4% of frames.
- Per-clip read: the gain is mostly e271-like surface frames; aaf1 and d129 stay
  at baseline under the confidence gate.

Interpretation: this is the first multiclip surface policy that improves without
blindly replacing the baseline. It supports the state-machine/fallback direction.
It is not a final surface tracker; the next data need is still true
tree/grass/terrain labels from additional clips.

From the first live runtime integration of the surface fallback:

- Added explicit `tbd_motion_detector.py` flags:
  - `--surface_ranker_model`
  - `--surface_ranker_policy confidence_fallback`
  - `--surface_ranker_threshold`
  - `--surface_ranker_top_n`
- The integration keeps the baseline scorer as the default and only lets the
  learned ranker override selection when its confidence clears the threshold.
- On full e271, using the large-dark profile and no top-tube export:
  - baseline: 63.2% strict, 72.7% loose, 12.56 ms/frame average, 15.77 ms p90.
  - fallback top-20: 70.2% strict, 77.8% loose, 23.92 ms/frame average, 27.40 ms p90.
  - fallback top-40/top-80 did not improve accuracy over top-20 and were slightly slower.
- The late e271 tail `667..698` remains essentially unsolved:
  0/32 strict and 1/32 loose under the fallback.

Interpretation: confidence-gated surface fallback now works inside the live
selector and stays Mac-side 30 Hz at top-20. The remaining tail failure is not
fixed by selector integration; it needs new terrain-tail features or more
similar hard labels.

Audit update:

- Expert review found that the v1 runtime fallback was not computing all
  pair/alignment feature columns used by the trained ranker when top-tube export
  was disabled. This has been fixed in `tbd_motion_detector.py`.
- The v1 fallback model was also an all-fit model that included e271 labels, so
  the e271 runtime gain was in-sample integration evidence, not clean
  generalization.
- New artifact: `artifacts/runtime_surface_fallback_e271_audit_v2/`.
  - baseline rerun: 63.2% strict, 72.7% loose, 12.47 ms/frame average.
  - all-fit fallback after feature parity: 71.1% strict, 78.1% loose,
    25.17 ms/frame average.
  - e271-held-out fallback: 65.8% strict, 77.4% loose, 25.24 ms/frame average.
  - e271-held-out tail `667..698`: 28.1% strict, 37.5% loose.

Interpretation: the selector/ranker direction still has signal, especially for
loose continuity and terrain-tail recovery, but the honest held-out strict gain
is modest. Do not present the all-fit result as baseline progress. Next clean
validation should use nested threshold selection or a separate holdout clip.

Nested threshold-selection update:

- Added nested fallback evaluation to `train_surface_xy_ranker.py`: for each
  held-out clip, choose model/threshold using only the other clips, then score
  the held-out clip.
- Reran `artifacts/surface_training_v4_multiclip/ranker_e271_heldout_runtime_audit/`.
- Optimistic fixed-threshold fallback remains 83.3% strict / 90.1% loose.
- Nested fallback is 77.8% strict / 85.8% loose.
- Baseline top-tube replay is 80.0% strict / 85.7% loose.

Interpretation: global learned fallback is not a validated default. The ranker
is useful as a state-specific/terrain-tail tool, but strict global selection
needs better state routing or more representative hard surface labels before it
should replace baseline behavior.

State-gate follow-up:

- Added nested gated fallback replay and an explicit runtime
  `--surface_ranker_gate` flag.
- Nested gated replay improves over ungated nested replay:
  - nested ungated: 77.8% strict / 85.8% loose.
  - nested gated: 80.2% strict / 86.1% loose.
- Runtime initially applied the gate before scoring, while replay scored first
  and gated only the best learned candidate. After fixing runtime semantics, the
  live e271 high-support result improved:
  - held-out ungated fallback: 65.8% strict / 77.4% loose.
  - held-out high-support gate, threshold 0.00: 69.2% strict / 80.5% loose.
  - held-out high-support gate, threshold 0.76: 69.1% strict / 80.4% loose.
  - the high-support gate does not recover the late e271 tail; ungated fallback
    still owns that specific failure mode.

Interpretation: `high_support` is now a plausible state-specific policy for
the main e271 body, but it is not a global default because it trades away tail
recovery. Keep `--surface_ranker_gate` experimental until another held-out
surface clip confirms the same behavior.

From the first complete-video non-sky benchmark:

- Built a full-video d129 vision-label set: 250 total frames, 38 visible target frames, 212 invisible/no-target frames.
- Current actual pair-rescue tracker: 160/250 all-frame correctness = 64.0%.
- Current actual visible strict hit: 29/38 = 76.3%.
- Current actual invisible no-box rate: 131/212 = 61.8%.
- Top-80 oracle on visible d129 frames: 38/38 = 100%.
- HistGBDT visible-frame ranking with d129 held out: 34/38 = 89.5%.
- Simple learned null threshold can reach 86.8% all-frame correctness, but visible strict drops to 26/38 = 68.4%.

Interpretation: d129 is now a useful complete-video benchmark. The target is in the candidate pool, and learned ranking can pick it when visible. The blocker is acquisition/null logic: the current tracker hallucinates before target appearance, while a blunt learned threshold suppresses too much target recall.

From the embedded runtime candidate-router implementation:

- Added explicit runtime modes and candidate-local routing inside `scripts/tbd_motion_detector.py`.
- Default remains conservative: `--runtime_mode baseline --candidate_router off`.
- `auto` can log frame/candidate router decisions without changing behavior.
- Candidate-local router states are now exported in candidates and tube features.
- Per-frame timing is exported in `report.json` and `timing_summary.csv`.
- Added `scripts/benchmark_runtime_modes.py`.
- Fixed audit issues:
  - `candidate_router=log` no longer gates surface ranker behavior.
  - `candidate_router=off` is respected even in fixed runtime modes.
  - scenario-balanced candidate lists now honor the effective runtime cap.
  - final router counts use the final candidate set.
  - `hybrid_coast` is no longer treated as a surface-only source.
- Beam hot-path pass:
  - precomputes state warp/prediction references once per frame;
  - skips diagnostic pair/background/alignment features unless needed.
  - skips materializing candidate-transition states that cannot beat the
    current per-candidate winner.
- Pair-rescue profile, 89-frame slices:
  - d129 baseline 28.20 ms/frame, auto_apply 28.75 ms/frame, clean_sky 27.10 ms/frame.
  - aaf1 baseline 33.88 ms/frame, auto_apply 36.74 ms/frame, clean_sky 32.77 ms/frame.
  - e271 baseline 14.93 ms/frame, auto_apply 15.98 ms/frame, clean_sky 15.54 ms/frame.
  - p90 timing still misses 30 Hz on d129/aaf1 slices, so average timing alone
    is not enough for deployment confidence.
- Heavy surface extras remain too slow as explicit `surface` mode: d129 86.87 ms/frame, e271 53.27 ms/frame on 45-frame slices.

Interpretation: the router infrastructure is now cleaner and the Python beam
update is no longer the only dominant cost in normal pair-rescue mode. Mac-side
average 30 Hz is plausible for d129/e271 and borderline for aaf1, but this is
not yet a Pi-ready runtime profile.

From the current top-tube / temporal-stack profile:

- New artifact: `artifacts/surface_stack_profile_2026_05_29/`.
- Current no-stack aaf1 visible-frame oracle@8 is only 17/75 = 22.7%; selected@8 is 6/75 = 8.0%.
- Enabling temporal-stack proposals raises aaf1 oracle@8 to 74/75 = 98.7%, but selected@8 only reaches 7/75 = 9.3%.
- d129 remains a selection/null failure: oracle@8 is 38/38, selected@8 is 0/38.
- e271 does not benefit from global stack in the current top-80 profile: selected@8 drops from 442/699 = 63.2% to 420/699 = 60.1%.
- All-stack ExtraTrees LOCO improves aggregate visible strict replay from 59.0% to 69.0%, but held-out aaf1 is still only 6/60 = 10.0% and held-out d129 is 0/34.
- All-fit separability on aaf1 is 55/60 = 91.7% strict, so the feature set can use temporal-stack positives after seeing similar examples.
- Runtime is not acceptable for global stack: 107-146 ms/frame average on the current machine.

Interpretation: aaf1 is no longer primarily a visibility/proposal problem when
temporal stack is available; it is a ranker generalization problem. Full-frame
temporal stack should remain an offline teacher / surface-branch experiment, not
a default runtime path. The next data need is more aaf1-like tree/road/grass
positive labels from another clip, plus null labels for similar clutter.

From the first acquisition/null state-machine sweep:

- Added `scripts/evaluate_lock_state_machine.py`.
- Evaluated the existing d129 complete-video labels against the learned
  HistGBDT per-frame best candidate table.
- Best all-frame state config: 225/250 = 90.0% all-frame correctness, 26/38 =
  68.4% visible strict, 199/212 = 93.9% invisible no-box.
- Recall-preserving state config: 84.8% all-frame correctness while keeping the
  current 29/38 = 76.3% visible strict recall and improving invisible no-box to
  86.3%.
- Probe `--mask_selected_for_motion_model`: d129 visible strict improved
  29/38 -> 30/38, but all-frame correctness fell 160/250 -> 152/250 because
  hallucinations increased. Keep it as an explicit experiment, not a default.
- Negative control over native detector `verified_score` from top-80 tubes:
  best all-frame was 86.8%, but visible strict collapsed to 10/38 = 26.3%.
  No swept config reached 65% visible strict recall.
- Export hygiene fix: top-tube and selected-feature rows now mark whether
  `cand_*` fields are from a current-frame candidate. Coasting/missed states no
  longer export stale candidate fields as if they were fresh detections.

Interpretation: acquisition/null state is a better no-new-label path than more
threshold tuning. It can materially suppress hallucinations, but the
high-accuracy config acquires late, and raw detector scores are not sufficient
for selection. The next integration should support two modes: strict search/null
before lock and continuity-backed tracking after lock, driven by learned or
out-of-fold candidate scores rather than native `verified_score` alone.

From the first full-video OOF state-ranker harness:

- Added `scripts/train_full_video_state_ranker.py`.
- It labels top-tube candidates from full-video frame labels: visible target
  candidates are positives, far visible candidates and no-target-frame
  candidates are negatives.
- On d129, the OOF best-candidate strict rate improved from baseline
  `verified_score` 26/38 = 68.4% to:
  - logistic: 38/38 = 100.0%
  - HistGBDT: 37/38 = 97.4%
  - ExtraTrees: 38/38 = 100.0%
- ExtraTrees OOF scores plus the state machine reached 249/250 = 99.6%
  all-frame correctness, 38/38 visible strict, and 211/212 invisible no-box.
- This is a one-clip stratified OOF result, not leave-one-clip-out proof. It is
  strong evidence that null-aware candidate scores are the right selector input,
  but it should not become a default until reproduced on another complete clip.

From the reproducible full-video OOF state-eval driver:

- Added `scripts/run_full_video_oof_state_eval.py` to generate OOF candidate
  scores and run acquire/track/null sweeps from one command.
- Rebuilt the d129 complete-video benchmark as
  `artifacts/full_video_oof_state_eval_d129_v2/`.
- The artifact now keeps all-candidate OOF score tables:
  `oof_candidate_scores_<model>.csv`, not only best-per-frame selections.
- Best OOF ExtraTrees + state machine:
  - 249/250 all-frame correct = 99.6%.
  - 38/38 visible strict = 100.0%.
  - 211/212 invisible/no-box = 99.5%.
  - selected frames = 39.
- OOF logistic + state machine also kept 38/38 visible strict, but with more
  false positives: 236/250 all-frame correct and 198/212 invisible/no-box.
- Native `verified_score` negative control, using the correct raw-score
  threshold grid, reached 217/250 all-frame correct but only 10/38 visible
  strict. This confirms that the win is not the state machine alone; it needs
  null-aware learned candidate scores.

Interpretation: the next integration target is now concrete: pass learned
candidate scores into the acquire/lock/null selector. The same caveat remains:
this is one-clip stratified OOF, not cross-video proof.

From the second vision-checked surface pass:

- Re-profiled the next-batch labels with the target-local router.
- Strictly promoted only visually checked surface-backed labels:
  - e271 frames 631-698: 68 terrain/road/grass-backed visible frames.
  - 7bd frames 583-588: 6 short close surface-backed frames.
- Rejected the 1c router-surface ranges for this pass because visual review
  showed they are mostly skyline/near-horizon, not true tree/grass/terrain
  training data.
- On e271 631-698, top-80 oracle was 68/68 = 100%.
- Native `verified_score` selected only 20/68 strict = 29.4%.
- OOF logistic framewise ranker selected 64/68 strict = 94.1%.
- The four remaining misses were continuity/ranking failures, including two
  road/texture jumps while the correct target tube was still present.
- Added `scripts/evaluate_xy_sequence_ranker.py`.
- OOF logistic scores plus a simple Viterbi continuity selector reached
  68/68 strict = 100% on that segment.
- Runtime over the full e271 clip, including the hard tail:
  - baseline pair-rescue: 15.70 ms/frame average, 24.55 ms p90.
  - auto router/apply: 16.97 ms/frame average, 26.27 ms p90.
  - forced surface mode: 17.14 ms/frame average, 26.81 ms p90.

Interpretation: this is the strongest evidence so far that the surface problem
is not just proposal recovery. In the checked e271 terrain segment, the target
is already in the top-tube pool; the failure is selecting a coherent target
sequence over intermittent road/terrain clutter. The result is still
single-segment and should not be promoted as a global default until repeated on
another true tree/grass clip with no-target frames.

From the e6 surface-mining pass:

- Mined `e6a60298-989b-4c37-9b8f-79e0af4d62a0` and promoted 72
  high-confidence visible labels:
  - 60 skyline-surface frames.
  - 12 textured/non-sky frames.
- Kept 27 ambiguous tree/terrain rows out of training for human review.
- All-stack e6 top-tube read:
  - oracle@8 = 67/72 = 93.1%.
  - selected@8 = 57/72 = 79.2%.
  - frames 1775-1800 are useful ranking failures: the target is present in
    top-tube alternatives while the selected box is more than 120 px away.
- Added the e6 labels to the all-stack ExtraTrees LOCO ranker:
  - aggregate strict recall improved from 68.97% to 71.67%.
  - aggregate loose recall improved from 82.00% to 84.42%.
  - d129 held-out strict improved from 0/34 to 6/34.
  - aaf1 held-out strict stayed stuck at 6/60.
  - e6 held-out strict was 62/72 = 86.1%.
- Reran the acquisition/null diagnostic with the e6 labels:
  - baseline selected: 50.10% all-frame correctness, 57.69% visible strict,
    1.42% invisible no-select.
  - nested logistic: 36.79% all-frame correctness, 42.38% visible strict,
    0.94% invisible no-select.
- Optimized `scripts/train_acquisition_null_ranker.py` so threshold sweeps reuse
  cached per-frame scores rather than recomputing model predictions for every
  threshold.

Interpretation: e6 labels are a real but limited ranker improvement. They do not
solve aaf1-style tree/grass generalization, and the null-ranker diagnostic is a
negative result. The next data need is still more true target-over-tree/grass
clips plus more varied no-target/clutter frames.

From the aaf1/e6 hard-null pass:

- Reviewed the 27 ambiguous e6 selected-candidate rows and promoted them as
  high-confidence no-target/hard-null labels, not positives. They read as
  tree/terrain/skyline clutter.
- Inspected aaf1 frame samples and zoomed 480-640 gap crops. No new positive
  labels were promoted because the local dark-component tracker drifted between
  tiny sky points, cloud specks, and tree/terrain texture.
- Added 28 high-confidence aaf1 pre-acquisition no-target labels from frames
  0-270.
- New merged label set:
  `artifacts/aaf1_surface_mining_v1/aaf1_e6_existing_plus_null_labels_v1.csv`
  with 1,359 visible frames and 267 invisible/null frames.
- Logistic acquisition/null diagnostic:
  - before aaf1 nulls: 31/239 invisible no-select = 13.0%.
  - after aaf1 nulls: 58/267 invisible no-select = 21.7%.
  - aaf1 pre-acquisition null: 27/28 no-select.
  - e6 null: 14/27 no-select.
  - d129 null remains poor: 17/212 no-select.
  - visible strict remains weak: 590/1359 = 43.4%.

Interpretation: this is useful hard-null data, but not a deployable null model.
The model can learn local aaf1/e6 no-target clutter suppression but does not
generalize to d129. More varied null labels are needed, and positive aaf1 gap
labels should not be auto-promoted without human review.

From the multiclip null-mining / accept-gate pass:

- Reviewed no-label gap contact sheets across `1c`, `529`, `59e`, `7bd`, and
  `b96`.
- Rejected unsafe null ranges where the drone was visible or likely still
  visible.
- Promoted 37 high-confidence no-target frames:
  - `1c`: 5 early frames.
  - `59e`: 10 mid-clip terrain frames.
  - `7bd`: 12 early/late road/field frames.
  - `b96`: 10 early haze/road frames.
- New merged label set:
  `artifacts/null_mining_multiclip_v1/aaf1_e6_existing_plus_multiclip_null_labels_v1.csv`
  with 1,359 visible frames and 304 invisible/null frames.
- Select-best null ranker remains wrong:
  - nested logistic: 40.1% all-frame correctness, 42.9% visible strict,
    27.3% null no-select.
- Added `--decision_mode gate_selected` to `scripts/train_acquisition_null_ranker.py`.
  This scores only the detector-selected row and accepts/rejects it instead of
  reselecting a different candidate.
- Accept/reject gate result:
  - baseline selected: 47.6% all-frame, 57.7% visible strict, 2.6% null no-select.
  - optimistic fixed logistic threshold 0.05: 51.9% all-frame, 51.9% visible
    strict, 51.6% null no-select.
  - nested ExtraTrees gate: 47.0% all-frame, 46.0% visible strict, 51.3% null
    no-select.

Interpretation: the accept/reject gate is the right architecture for acquisition
null suppression, but the learned gate is not a global default. It should be
used as a state-machine acquisition gate candidate, with stricter policy around
when the system is not already locked.

State-machine accept-gate follow-up:

- Fixed two evaluator issues before trusting the numbers:
  - frame `0` in `train_acquisition_null_ranker.py` was being converted to
    `-1` by truthiness-based parsing;
  - `evaluate_lock_state_machine.py` now supports `--clip` filtering and a
    separate `--track_score_column`, so acquisition can use learned accept
    probability while locked tracking still uses raw `verified_score`.
- New artifact:
  `artifacts/null_mining_multiclip_v1/acquisition_gate_state_eval_v1/`.
- Per-clip best configs look good but are optimistic:
  - baseline selected/verified: 56.8% all-frame, 57.1% visible strict,
    55.3% null no-box.
  - ExtraTrees acquire gate + verified tracking: 64.9% all-frame, 57.3%
    visible strict, 99.0% null no-box.
- One shared config across all clips is the honest default-read:
  - baseline selected/verified: 54.2% all-frame, 55.6% visible strict,
    48.4% null no-box.
  - ExtraTrees acquire gate + verified tracking: 54.3% all-frame, 55.6%
    visible strict, 48.7% null no-box.
  - logistic acquire gate + verified tracking: 54.1% all-frame, 52.9%
    visible strict, 59.5% null no-box.

Interpretation: the acquisition-only gate is architecturally useful, but it is
not a meaningful global improvement yet. The impressive per-clip gain is mostly
threshold tuning to each video. With a single config it is basically baseline
plus one frame. The shared-config failure export confirms the misses are still
visible-frame failures on e271, aaf1, 7bd, 529, d129, and 1c, not primarily null
false positives. The next valuable work is more state-specific labels and/or a
better router, not promoting this gate as default.

From the dense surface-label expansion:

- Added `artifacts/surface_dense_label_expansion_v1/`.
- Promoted 228 continuity labels by interpolation between existing
  high-confidence reviewed anchors:
  - `529`: 57 rows.
  - `7bd`: 66 rows.
  - `b96`: 85 rows.
  - `59e`: 20 rows.
- No aaf1 gap labels were promoted; that segment remains visually ambiguous and
  should not be trained as a positive without human review.
- Fixed `train_surface_xy_ranker.py` parsing so frame `0` and zero-valued
  coordinates are not lost through truthiness checks.
- Apples-to-apples ranker comparison:
  - original labels, fixed parser: ExtraTrees LOCO 71.67% strict / 84.42%
    loose; nested gated fallback 69.08% strict / 80.25% loose.
  - dense v1 labels: ExtraTrees LOCO and nested gated fallback both 75.21%
    strict / 87.96% loose.

Interpretation: this is meaningful offline progress for the learned surface
selector. The dense labels increase the hard-frame denominator and make the
learned selector generalize better across the currently labeled clips. It does
not solve aaf1: held-out aaf1 remains 6/60 strict, so true tree/foliage
acquisition is still the main unsolved visual regime.

Acquisition/null gate check on the dense label set:

- baseline selected: 45.43% all-frame, 53.62% visible strict, 2.63% null
  no-select.
- nested logistic gate: 41.78% all-frame, 38.37% visible strict, 59.54% null
  no-select.
- nested ExtraTrees gate: 36.70% all-frame, 36.42% visible strict, 38.16% null
  no-select.

Interpretation: positive continuity labels help the surface selector, not the
selected-row accept/reject gate. Do not promote the acquisition gate.

Held-out e271 runtime/sequence check:

- Added `scripts/apply_surface_sequence_selector.py`.
- The dense model trained with e271 excluded improves the live e271 fallback to
  71.1% strict / 84.1% loose at 27.61 ms/frame, compared with the previous
  held-out fallback at 65.8% / 77.4%.
- The high-support gate is worse on this run: 70.1% strict / 83.4% loose, so it
  should not be promoted.
- Exported top-80 alternatives show oracle strict@80 = 94.4% and loose@80 =
  99.9%. Frames 400-499 are the main failure block: oracle loose@80 is 100%,
  but selected loose is only 50%.
- A held-out sequence selector over top-20 candidates reaches 81.4% strict /
  98.9% loose on e271 with `max_jump_px=10` and `transition_weight=0.5`.

Interpretation: the current e271 bottleneck has moved from proposal recovery to
sequence-level selection. This is the first honest e271 result crossing 80%
strict while excluding e271 from the ranker training set. Caveat: the selector
is offline/deferred over exported candidates; it still needs a short sliding
window implementation before it is a live Pi-5 candidate.

CLBA score-modifier check:

- Added `scripts/sweep_clba_score_adjustment.py`.
- This keeps the proven acquire/track state machine and only sweeps direct
  score modifiers from candidate-local background alignment features.
- Added an optional CLBA adjustment path to
  `scripts/apply_surface_sequence_selector.py`, so the offline delayed
  sequence selector can now be compared with and without CLBA-adjusted candidate
  probabilities when augmented top-tube rows are available.
- It is a narrower test than the explicit A/P/T/S/E/C selector, which is still
  not validated.
- Best aaf1 ExtraTrees CLBA-adjusted run:
  - 65/75 visible strict = 86.7%.
  - 28/28 invisible no-box = 100%.
  - 93/103 all-frame correctness = 90.3%.
  - Previous comparable all-stack CLBA state-machine run was 62/75 strict with
    28/28 no-box.
- Best e6 HistGBDT CLBA-adjusted run:
  - 66/72 visible strict = 91.7%.
  - 26/27 invisible no-box = 96.3%.
  - 92/99 all-frame correctness = 92.9%.
  - Previous comparable all-stack CLBA state-machine run was 65/72 strict with
    26/27 no-box.

Interpretation: the professor's target/background-alignment primitive has a
measurable but modest selector benefit when used as a calibrated modifier on
top of OOF candidate scores. Strong hand-designed clutter penalties are still
not validated; the winning weights are small and clip/model-specific. Keep this
as an offline calibration harness and feature source, not a runtime default yet.

CLBA sequence-selector check:

- Added `scripts/sweep_clba_sequence_selector.py`.
- This evaluates OOF candidate scores with optional CLBA score modifiers under
  framewise, rolling-window, and full-window Viterbi selection.
- Added sequence-path acquisition/keep hysteresis because plain sequence
  continuity improved visible recall while hallucinating through null frames.
- Tested lower acquisition thresholds with multi-hit acquisition. This did not
  improve the current best aaf1 result: it preserved null safety in some
  configs but dropped visible strict from 69/75 to 67/75. The remaining aaf1
  misses are mostly target-birth score/ranking issues, not just acquisition
  hysteresis issues.
- Best aaf1 ExtraTrees OOF result:
  - 69/75 visible strict = 92.0%.
  - 28/28 invisible no-box = 100%.
  - 97/103 all-frame correctness = 94.2%.
  - config: full-window Viterbi, jump 10 px, acquire 0.90, keep 0.30,
    attached penalty 0.10, no lost patience.
- Best aaf1 zero-CLBA result in the same harness:
  - 56/75 visible strict = 74.7%.
  - 27/28 invisible no-box = 96.4%.
  - 83/103 all-frame correctness = 80.6%.
- Best e6 HistGBDT OOF result:
  - 67/72 visible strict = 93.1%.
  - 26/27 invisible no-box = 96.3%.
  - 93/99 all-frame correctness = 93.9%.
  - best e6 config is effectively framewise/no-CLBA; sequence/CLBA do not add
    useful signal there.

Interpretation: on aaf1-like hard surface/background labels, the combination
of continuity + acquisition/keep hysteresis + a small CLBA attached penalty is
the first offline selector result that improves visible recall and null
suppression together. On e6, the learned per-frame score is already sufficient
and sequence/CLBA should not be forced. This supports a router/state-machine
direction: sequence mode should be applied only when the local background state
needs it, not globally.

Runtime delayed-sequence integration:

- Added non-default `tbd_motion_detector.py` delayed-sequence acquire/keep
  hysteresis flags:
  - `--delayed_sequence_acquire_threshold`
  - `--delayed_sequence_acquire_hits`
  - `--delayed_sequence_keep_threshold`
  - `--delayed_sequence_lost_patience`
- Added `--delayed_sequence_score_source surface_ranker`, so delayed sequence
  can use learned surface-ranker probabilities instead of only detector
  `verified_score`.
- Guardrail: `surface_ranker` scoring now fails fast unless a surface-ranker
  model/policy is explicitly configured.
- Smoke-tested aaf1 first 360 frames:
  - pair-rescue baseline: 32.37 ms/frame average, 45.09 ms p90.
  - delayed verified-score hysteresis: 36.94 ms/frame average, 50.56 ms p90.
  - surface-extras + learned-ranker delayed sequence: 161.87 ms/frame average,
    197.36 ms p95.

Interpretation: the runtime plumbing is now present and tested, but the heavy
surface/temporal-stack runtime path is much too slow as a global mode. The
offline aaf1 gain still needs a bounded, router-gated implementation before it
is a Pi candidate.

Runtime bounded-surface follow-up:

- Fixed `scripts/benchmark_runtime_modes.py` surface-extras invocation so
  negative temporal offsets are passed as `--temporal_stack_offsets=-5,-3,-1`
  instead of being parsed as flags.
- Reduced learned-ranker runtime overhead:
  - delayed mode now uses a raw best state only for motion-model masking;
  - surface-ranker extra pair/background/alignment diagnostics are computed
    only when a scoped path can actually enter the learned ranker.
- Added experimental candidate-local temporal stack:
  - `--temporal_stack_candidate_local`
  - `--temporal_stack_seed_top_k`
  - `--temporal_stack_local_halo_limit`
- Avoided recomputing candidate router/support context for candidates already
  routed in the cheap proposal pass.
- Added `--hybrid_coast_min_evidence` and stopped treating coast state prior as
  `map_score`; coast is no longer counted as a surface-source tube feature.
- Added `--surface_ranker_scope surface_context`, which includes
  `surface_backed` plus `boundary_mixed` / `sky_target_near_surface` tubes.

Measured on aaf1 first 360 frames:

- Previous auto surface-extras learned sequence: 100.13 ms/frame average,
  164.54 ms p95.
- Scoped ranker diagnostics: 73.27 ms/frame average, 134.33 ms p95.
- Candidate-local stack: 69.76 ms/frame average, 128.31 ms p95.
- Bounded candidate budget + de-duplicated context: 52.39 ms/frame average,
  73.82 ms p95.
- Coast-evidence/surface-context top-tube audit:
  - true target in top-80 for 31/54 visible aaf1 labels under frame 360;
  - selected strict is still only 1/54 in that window.

Interpretation: the runtime path is substantially lighter, but this is not a
tracking-quality win yet. The remaining aaf1 failure is a state/identity issue:
the selector acquires a smooth false map tube before the target window and does
not switch when the true target appears in nearby top alternatives. Next work
should be explicit false-lock/null state handling or retraining on current
top-tube hard alternatives, not more broad threshold/runtime tuning.

Current-runtime top-tube retrain and aaf1 proposal-recovery pass:

- Exported current bounded-runtime top tubes for all 9 dense-labeled clips to
  `artifacts/current_runtime_top_tubes_v2`.
- Aggregate current-top-tube audit, high/medium-high labels:
  - oracle@80: 92.5%;
  - baseline `verified_score` strict: 69.6%, loose: 74.5%;
  - best direct LOCO ranker (`hist_gbdt`) strict: 70.7%, loose: 76.8%.
- Per-clip result shows the aggregate hides the real blocker:
  - e6 is now healthy enough as a top-tube/ranking problem:
    oracle@80 100%, baseline strict 90.3%, loose 93.1%.
  - aaf1 is not healthy under bounded candidate-local stack:
    oracle@80 61.7%, baseline strict/loose 11.7%.
  - d129 and e271 have good-to-moderate oracle but poor selection, so they are
    still selector/null-calibration problems.
- aaf1 proposal-recovery sweep:
  - bounded candidate-local stack variants did not recover aaf1 oracle
    (roughly 60-62% oracle@80).
  - full-frame temporal stack + large-dark proposals recovers aaf1 oracle:
    old all-stack reference oracle@80 98.3%, but avg runtime 146 ms/frame.
  - bounded full-stack + large-dark top60/beam70 gives oracle@20 96.7% and
    baseline loose 60.0%, but is still too slow at 125.8 ms/frame average.
  - candidate-local stack seeded by large-dark did not recover the same oracle
    and was slower than expected.
- Ranker transfer result:
  - A ranker trained on other clips does not transfer to aaf1 even when the
    full-stack candidates are present.
  - The same `hist_gbdt` feature set trained including aaf1 ranks the recovered
    aaf1 candidates well: 83.3% strict and 100% loose on labeled aaf1 frames
    when selecting from top candidates.
  - Running the detector with that all-fit ranker gives 86.7% strict and 100%
    loose on the labeled aaf1 frames, but this is explicitly not a held-out
    result and the runtime is not deployable (194 ms/frame average with
    delayed sequence and ranker scoring).

Interpretation: the next jump is not more generic thresholding. aaf1 needs more
aaf1-like surface labels or an explicit target-vs-background alignment feature
that transfers; current scalar features can rank the clip once trained on it,
but they do not generalize from other videos. Runtime-wise, the useful oracle
source is full-frame temporal stack + large-dark proposals, and the next code
task is to convert that into a cheaper candidate/local/teacher pathway without
losing the recovered oracle.

CLBA augmentation and partial-label generalization pass:

- Augmented the mixed current/full-stack top-tube rows with offline
  candidate-local background-alignment features:
  `artifacts/mixed_aaf1_full_ld_top60_clba_v1`.
- Adding those CLBA columns to the global LOCO ranker gives only a small
  aggregate gain:
  - non-CLBA top60 mixed `hist_gbdt`: strict 74.8%, loose 83.1%;
  - CLBA top60 mixed `hist_gbdt`: strict 75.6%, loose 82.6%.
- Held-out aaf1 still does not transfer from other clips:
  - non-CLBA `hist_gbdt`: 15.0% strict / 16.7% loose;
  - CLBA `hist_gbdt`: 18.3% strict / 18.3% loose.
- Feature-level drilldown on aaf1 says the CLBA primitive is real but not
  enough by itself:
  - `clba_gain_norm` AUC true-vs-negative: about 0.79;
  - top false selected branch/terrain boxes have low CLBA gain;
  - raw `verified_score` still strongly over-scores some false locks.
- Direct CLBA score adjustment on aaf1 improves the local state-machine result:
  - best zero-CLBA aaf1 state-machine baseline: 9.3% strict visible recall,
    34.7% loose, 100% invisible no-box;
  - best direct CLBA aaf1 sweep: 45.3% strict, 56.0% loose, 89.3% invisible
    no-box.
  - fixed aaf1 CLBA weights are not globally safe: they hurt e6 and e271, so
    this must be a routed hard-surface branch, not a default selector.
- Added `scripts/evaluate_partial_clip_generalization.py` to test whether
  adding partial labels from a hard clip actually generalizes to held-out
  frames from the same clip.
- Partial-label checks:
  - aaf1 interleaved labels generalize strongly: held-out alternating-frame
    strict recall around 80-90%, loose around 93-100%.
  - aaf1 chronological split with only 6 target-training frames fails, which
    means the labels must be distributed across the segment, not clustered.
  - d129 interleaved `hist_gbdt` reaches up to 82.4% strict on held-out
    alternating frames.
  - e271 interleaved `hist_gbdt` reaches about 71-73% strict and about 79-80%
    loose on held-out alternating frames.

Interpretation: targeted distributed labels from the same surface mode are now
proven high ROI. The model can learn these domains once it sees representative
surface examples, but current cross-clip transfer remains weak. The next data
collection should focus on distributed true-target and hard-negative labels in
aaf1/d129/e271-like tree/grass/terrain footage. The next algorithm work should
make the CLBA/direct-adjustment branch router-specific and keep it out of e6/e271
unless the local state says it is in the same hard-surface regime.

## Next Meaningful Work

1. Add explicit false-lock/null state handling or train a current-top-tube
   ranker on the aaf1/e6 hard alternatives; verify it can switch away from the
   early aaf1 false map tube when the true target appears.
2. Turn the aaf1 full-stack + large-dark oracle source into a cheaper path:
   first as an offline teacher/hard-negative miner, then as a candidate-local
   or low-confidence surface branch. Do not make the 125-194 ms/frame path a
   runtime default.
3. Add more true surface labels from clips that resemble aaf1, not just skyline
   or clean-sky cases. The transfer failure says the current training set does
   not cover this domain.
4. Use `scripts/evaluate_partial_clip_generalization.py` after each new labeling
   packet. A useful packet should improve held-out alternating-frame recall,
   not just all-fit/demo recall.
5. Reproduce the CLBA+hysteresis sequence result on at least one more true
   tree/grass/terrain segment; 1c-style skyline-adjacent rows should stay out
   unless visually verified.
6. Keep improving the frame/candidate router so delayed sequence selection only
   pays the extra cost in surface-backed states.
7. Reproduce the full-video OOF state-ranker harness on at least one more
   complete clip, then integrate null handling only if the null/visible tradeoff
   survives.
8. Train null-aware tube rankers with explicit no-target/hallucination
   negatives.

Latest hard-surface label/ranker loop:

- Built a conservative Codex-reviewed aaf1 teacher packet from the all-fit
  full-stack selector:
  `artifacts/aaf1_teacher_surface_packet_v1/teacher_labels_vision_confirmed.csv`.
  I did not accept the full teacher packet. Branch/tree jump boxes around the
  ambiguous early gap were excluded; only 32 medium-high pseudo labels were
  promoted into
  `artifacts/aaf1_teacher_surface_packet_v1/labels_plus_codex_aaf1_teacher_v1.csv`.
- Important harness fix: the older CLBA table did not cover those new frames.
  Re-augmented aaf1 from the full top60 export into
  `artifacts/mixed_aaf1_full_ld_top60_clba_codex_v1`, producing 9,380 aaf1
  candidate rows across 135 merged label frames.
- With the corrected CLBA table, aaf1 partial-label generalization now shows
  the new supervision is useful:
  - early-to-late: `hist_gbdt` 92.6% strict / 96.3% loose;
  - late-to-early: `extra_trees` 67.7% strict / 73.9% loose;
  - interleaved splits remain strong, around 78-80% strict and 89-96% loose.
- Added `--include_null_frames` to `scripts/train_surface_xy_ranker.py` so
  visible=0 frames can contribute hard negative top-tube examples. This is
  necessary because the previous ranker ignored null labels and gave no-target
  temporal-stack candidates scores overlapping real target scores.
- Null-aware aaf1 score distributions improved, but selection is still a
  tradeoff:
  - no null training, threshold 0: 84.1% strict / 96.3% loose, only 3.6%
    invisible no-box;
  - null-aware, threshold 0.45: 74.8% strict / 80.4% loose, 46.4% invisible
    no-box;
  - null-aware, threshold 0.50: 72.9% strict / 77.6% loose, 57.1% invisible
    no-box;
  - null-aware hysteresis acquire 0.70 / keep 0.45: 63.6% strict / 67.3%
    loose, 92.9% invisible no-box.

Interpretation: the professor's direction is validated in part. Candidate-local
target-vs-background evidence plus distributed surface labels moves aaf1
substantially once the candidate table actually covers the new frames. The
remaining blocker is state calibration: null suppression and visible recall
still fight each other. The next implementation should make null/background
state handling first-class rather than relying on one global score threshold.

Offline null-state HMM selector pass:

- Added `--selector hmm` to `scripts/apply_surface_sequence_selector.py`. This
  keeps the existing Viterbi selector as the default, but adds an explicit
  candidate-HMM test path with:
  - `A`: absent/no box;
  - `T`: acquired target, emits a candidate;
  - `C`: lost/coast, emits no box and can reacquire near the prior box.
- Best conservative aaf1 setting so far:
  `--hmm_score_scale 1.0 --hmm_birth_penalty 0.6 --hmm_miss_penalty 0.35
  --hmm_track_bonus 0.15 --hmm_max_coast 1 --hmm_clutter_weight 0.3`.
  Result on the merged aaf1 label set:
  - 77.6% strict / 83.2% loose visible recall;
  - 78.6% invisible no-box;
  - 97 selected frames out of 134 candidate frames.
- This beats the best simple null-aware threshold point tested in this loop:
  threshold 0.45 gave 74.8% strict / 80.4% loose with 46.4% invisible no-box.
- A softer HMM setting improves visible recall but weakens null suppression:
  `--hmm_clutter_weight 0.15` gives 79.4% strict / 84.1% loose with 57.1%
  invisible no-box.
- Regression check: the conservative HMM setting is not globally safe. It works
  reasonably on aaf1/d129/e6-style null-heavy or hard-surface cases, but hurts
  e271 badly because it emits no box on too many visible-only frames
  (e271 strict about 38.6%, loose about 47.4% in the quick regression run).

Interpretation: explicit null/coast state handling is a real improvement for
the aaf1 hard-surface/null problem, but it must stay behind a routed hard-surface
or null-risk policy. It should not replace the global selector. The next useful
work is to add a router/state criterion for when to use HMM null mode and when
to keep the more permissive visible-continuity selector.

Selector-router probe:

- Added HMM evidence score modes to `scripts/apply_surface_sequence_selector.py`
  for analysis:
  - `logit`: previous default;
  - `centered`: score evidence is `scale * (score - center)`;
  - `raw`: score evidence is `scale * score`.
- The score-mode sweep did not produce a safe global selector:
  - centered modes can improve aaf1 visible recall, but e271 remains too weak;
  - raw modes recover more e271 continuity, but collapse no-target suppression.
  This confirms the problem is routing/state calibration, not just a score
  transform.
- Added `scripts/evaluate_selector_router_policy.py`, an offline diagnostic for
  combining existing selector outputs by CLBA background-lock risk.
- Current CLBA clip-risk statistic is median rank-1
  `clba_bg_static_likelihood - clba_target_likelihood`:
  - aaf1: `0.911`;
  - d129: `0.812`;
  - e271: `0.315`;
  - e6: `-0.966`;
  - other clips are strongly negative.
- Routing HMM only when risk > `0.5` selects HMM for aaf1/d129 and Viterbi for
  the other seven clips:
  - global HMM: 64.9% strict / 74.2% loose / 97.7% invisible no-box;
  - global Viterbi: 79.7% strict / 88.2% loose / 3.3% invisible no-box;
  - routed risk > `0.5`: 79.4% strict / 87.4% loose / 79.6% invisible no-box.
- Robustness check: this routing signal is sensitive to the statistic. Mean
  risk and p75 risk overreact to high-risk tails in otherwise normal clips;
  median rank-3 risk underreacts to aaf1/d129 null risk. The best current probe
  is specifically median rank-1 risk, which means the runtime version should use
  a recent top-candidate/window signal rather than a broad candidate-population
  statistic.

Interpretation: CLBA static-vs-target risk is a promising selector-router
signal. It is not yet a runtime solution, because this probe uses clip-level
median risk from exported top tubes. The next implementation should make the
same decision candidate/window-local: use HMM/null behavior only when the recent
candidate stream looks background-locked, otherwise keep the permissive
continuity selector.

Adaptive window-router test:

- Added explicit offline `--selector adaptive_hmm` to
  `scripts/apply_surface_sequence_selector.py`. It computes a causal rolling
  median of rank-limited CLBA static-lock risk, then chooses between rolling
  Viterbi and conservative HMM per frame.
- This first local router is not good enough:
  - risk threshold `0.50`: 72.3% strict / 82.5% loose / 65.1% invisible
    no-box;
  - threshold `0.75`: 73.1% strict / 83.3% loose / 60.5% invisible no-box;
  - threshold `1.00`: 73.5% strict / 83.8% loose / 56.6% invisible no-box;
  - threshold `1.50`: 74.2% strict / 84.7% loose / 46.4% invisible no-box;
  - threshold `2.00`: 74.8% strict / 85.7% loose / 38.2% invisible no-box;
  - threshold `3.00`: 76.1% strict / 87.0% loose / 25.3% invisible no-box.
- Failure read: naive per-frame switching fires HMM too often inside
  continuous-visible clips such as e271, so visible continuity drops before
  null suppression reaches the clip-level router result.

Interpretation: the runtime router should not be a raw per-frame threshold.
The next useful router experiment needs state/hysteresis around the routing
decision itself: enter HMM/null mode only after sustained background-lock
evidence, exit it quickly when target-like continuity dominates, and avoid
switching inside continuous visible tracks.

Adaptive router hysteresis check:

- Added `--adaptive_risk_acquire_threshold`, `--adaptive_risk_keep_threshold`,
  `--adaptive_risk_hits`, and `--adaptive_risk_release`.
- Fast sweep over acquire/keep/hit/release settings did not beat the clip-level
  diagnostic route. The best balanced setting found was:
  `acquire=0.75`, `keep=0.25`, `hits=1`, `release=3`.
- Full apply/eval verification for that setting:
  - 71.5% strict / 81.8% loose;
  - 73.0% invisible no-box;
  - 1,571 selected frames.

Interpretation: hysteresis helps no-box relative to raw adaptive thresholding,
but it still sacrifices too much continuous visible tracking. Do not promote
`adaptive_hmm` to default. The useful next direction is not more thresholding;
it is a single state model that scores target, null, and background-lock states
together instead of switching between two separately optimized selectors.
