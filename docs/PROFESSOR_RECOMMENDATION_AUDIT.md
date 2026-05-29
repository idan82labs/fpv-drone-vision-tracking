# Professor Recommendation Audit

Date: 2026-05-29
Commit audited: `98b3fde`

Update after later implementation passes: the repo now includes an offline
`A/P/T/S/E/C` explicit-state selector harness in
`scripts/evaluate_explicit_state_selector.py`, plus an experimental
`--global_quarantine` option for beam-wide `S/E` lock suppression. The harness
is still not validated as a default: hard global quarantine improves null
suppression but loses too much visible target recall, while soft quarantine does
not beat the sidecar baseline. The main open issue remains observation
calibration, not state bookkeeping.

This checks the project against the professor's CLBA-1 recommendation:

> candidate-local target-aligned minus background-aligned tube likelihood,
> explicit null/background/attached states, fixed-lag candidate Viterbi/HMM,
> routed surface use only, and full-video evaluation.

## Executive Read

We have not fully exhausted the recommendation. We have tested many adjacent
pieces, and the evidence supports the professor's diagnosis, but the strongest
version has not been implemented yet.

What was tried:

- offline target-aligned vs background-aligned CLBA features;
- local distractor/control normalization;
- direct CLBA score modifiers;
- learned rankers with and without CLBA columns;
- null-aware acquisition/ranker gates;
- candidate-local router infrastructure;
- candidate-local temporal-stack probes;
- explicit absent/target/coast HMM selector;
- CLBA-risk router, adaptive HMM router, hysteretic router, and learned
  Viterbi/HMM mode supervisor;
- full-video metrics across aaf1/e6/d129/e271 plus regression clips.

What is still missing:

- true joint `A/P/T/S/E/C` fixed-lag model;
- explicit `S` static/background-lock and `E` attached-edge/tree lock states;
- quarantine of background-lock anchors;
- CLBA observation used as `O_T - logsumexp(O_S,O_E,O_H,O_N)` inside the state
  model instead of as a score modifier or selector-family router;
- soft physical motion prior over range bins;
- per-router maximum-null-window threshold calibration;
- causal runtime implementation of CLBA-1 at bounded `K <= 40`, `L <= 5`.

## Recommendation Checklist

| Professor item | Status | Evidence | Honest read |
| --- | --- | --- | --- |
| Candidate-based fixed-lag Viterbi/HMM, not dense full-video Viterbi | Partial | `scripts/apply_surface_sequence_selector.py` has rolling Viterbi and candidate HMM; `scripts/evaluate_mode_supervisor.py` tests mode routing. | We avoided dense image Viterbi. The HMM is not yet the requested full state model. |
| Start with lag `L=5`, test `L=7/9` offline | Partial | CLBA augmentation defaults to `window_radius=4` past frames; sequence/hysteresis windows were swept. | We tested windowed/deferred selectors, but not a clean fixed-lag `A/P/T/S/E/C` implementation with L=5/7/9. |
| States `A/P/T/S/E/C`, optional `H` | Not complete | Current HMM has only `A/T/C`; other scripts emulate null/acquire/lock but not explicit `P/S/E/H`. | This is the largest implementation gap. |
| S/E should not poison future; quarantine anchors | Not implemented | No real stabilized-coordinate quarantine mechanism in runtime/selector. | We diagnosed false-lock poisoning but only mitigated with selector resets/routers. |
| Observation `O_T - logsumexp(O_S,O_E,O_H,O_N)` | Partial | `augment_top_tubes_alignment_features.py` computes target/background/static/attached CLBA fields; `tbd_motion_detector.py` has heuristic likelihood terms. | We have the features, but not a principled joint observation in the tracker. |
| Candidate rank/old score only weak proposal priors | Partial | Learned ranker still often dominates; CLBA score modifiers are additive. | This remains a failure mode: raw/learned score can preserve tree/terrain locks. |
| Real MISS/null candidate every frame | Partial | Null-aware HMM and acquisition/null ranker tested; selected-track outputs can emit no box. | Done in offline selectors, not in a full runtime state model with P/S/E/C semantics. |
| Target-aligned minus background-aligned tube score | Tried | `profile_tube_alignment_features.py`, `augment_top_tubes_alignment_features.py`; CLBA feature drilldown. | Real signal: `clba_gain_norm` AUC about `0.79` on aaf1 hard examples, but not enough alone. |
| Local distractor-normalized margin | Tried, partial | Deterministic annulus controls in `augment_top_tubes_alignment_features.py`. | Useful, but controls are not yet same-router/same-texture buckets as recommended. |
| Static/background-lock likelihood | Tried, partial | `clba_bg_static_likelihood`; router-risk policy; HMM clutter weight. | Good clip-level signal, weak frame-local signal. |
| Attached-edge/tree support as clutter likelihood, not hard kill | Partial | `clba_attached_likelihood`, attached support, line context; router line-attached class. | Implemented mostly as penalties/features; not clean alternate `E` state likelihood. |
| Soft physical motion prior over range bins | Not implemented | Current code uses max-jump, transition cost, static/jump penalties. | We have image-plane gates, not the recommended range-bin Student-t prior. |
| Candidate-local router | Implemented infrastructure | `tbd_motion_detector.py` candidate router states; runtime mode logging/apply flags. | Good infrastructure, but not calibrated enough for production decisions. |
| Tiny CNN/TCN only after labels | Deferred intentionally | We stayed with engineered features / logistic / GBDT. | Correct deferral; label volume is not enough for a safe CNN. |
| Full-frame temporal stack as teacher, not runtime | Tried and rejected as runtime | aaf1 full-stack oracle high, runtime 107-146 ms/frame; candidate-local versions did not recover oracle. | The professor's warning was correct. Keep full-stack as offline teacher/miner. |
| Sequential log-odds / null calibration | Partial | Acquisition/null gates, HMM, hysteresis, mode supervisor. | We tested many forms, but not per-router max-null-window calibration. |
| Per-router thresholds/null distributions | Not complete | Router metrics exist; no calibrated `P(max_window_score>x | no target)` by router. | Needed before production router. |
| Evaluation protocol | Mostly done | Full-video strict/loose/no-box, oracle@K, LOCO, timing, demos, worst cases in docs/artifacts. | Missing consistent per-router metrics, ECE/reliability, and worst-case sheets for every idea. |
| Pi/Rust path after algorithm moves | Followed | Rust deferred; Pi/runtime profiling done. | Correct: algorithm still blocks production more than implementation language. |

## Key Results So Far

### CLBA feature/score tests

- Direct CLBA score adjustment improved local aaf1 state-machine strict recall
  from `9.3%` to `45.3%`, with invisible no-box at `89.3%`.
- CLBA-enhanced global LOCO ranker barely moved aggregate performance:
  `75.6%` strict / `82.6%` loose.
- Held-out aaf1 still did not generalize from other clips:
  `18.3%` strict / `18.3%` loose.

Read: CLBA is real signal, but scalar CLBA features do not solve transfer or
state identity by themselves.

### Temporal stack

- Full-frame temporal stack plus large-dark proposals recovered aaf1 oracle,
  but at roughly `107-146 ms/frame`.
- Candidate-local stack did not recover the same oracle and remained slower
  than expected.

Read: keep full-stack as offline teacher/hard-example miner, not runtime.

### Explicit null/coast HMM

- Best conservative HMM on aaf1:
  `77.6%` strict / `83.2%` loose / `78.6%` invisible no-box.
- It hurts e271 badly:
  about `38.6%` strict / `47.4%` loose in the quick regression.

Read: explicit null/coast is useful for hard-surface/null clips, unsafe
globally.

### CLBA clip-level router

- Global HMM: `64.9%` strict / `74.2%` loose / `97.7%` invisible no-box.
- Global Viterbi: `79.7%` strict / `88.2%` loose / `3.3%` invisible no-box.
- Clip-level CLBA-risk route: `79.4%` strict / `87.4%` loose / `79.6%`
  invisible no-box.

Read: routing is the right production shape, but clip-level risk is not a live
solution.

### Frame-local/adaptive routers

- Raw adaptive HMM: best points around `72-76%` strict, no-box degrades as
  visible recall improves.
- Hysteretic adaptive route: `71.5%` strict / `81.8%` loose / `73.0%` no-box.
- Learned mode supervisor:
  - OOF AUC `0.693`;
  - no guardrail threshold `0.2`: `73.1%` strict / `82.0%` loose / `84.5%`
    no-box;
  - stateful/protected variants still did not beat clip-level route.

Read: current scalar top-tube/CLBA features cannot safely decide Viterbi vs HMM
frame-by-frame.

## What We Have Not Exhausted Yet

The professor's highest-value experiment is still only partially done. The next
strict version should be:

1. Build a real candidate fixed-lag state model with `A/P/T/S/E/C`.
2. Treat `S` and `E` as negative/background identity states, not selector
   outputs.
3. Add local quarantine for `S/E` anchors in stabilized coordinates.
4. Compute `O_T`, `O_S`, `O_E`, `O_H`, `O_N` from CLBA target/background,
   attached support, skyline/parallax, null prior, and weak proposal prior.
5. Add a soft motion likelihood over range bins instead of max-jump-only gates.
6. Calibrate thresholds from max-null-window distributions per router bucket.
7. Evaluate with the professor's required split:
   - aaf1 selected@8 and no-target no-box;
   - e6 strict/loose/no-box;
   - d129 acquisition/no-box;
   - e271 regression;
   - oracle@K separately;
   - p50/p90/p95/p99 timing;
   - worst false locks and worst misses.

## Recommendation

Do not ask the professor whether to keep threshold sweeping; that is answered.
The next professor question should be narrower:

- Given that CLBA is a real but insufficient scalar feature, should we implement
  the full `A/P/T/S/E/C` joint model next, or is the evidence now strong enough
  that engineered CLBA should be killed in favor of crop-stack target-vs-
  background learning?
- If we implement the joint model, what exact `O_T/O_S/O_E/O_H/O_N` terms and
  transition priors should be used first to avoid another tuning explosion?
- How should `S/E` quarantine work in stabilized coordinates when the camera
  registration is imperfect and the target may pass close to tree/terrain
  clutter?
