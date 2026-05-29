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
still not a Pi 5 claim. Surface acquisition remains a separate, explicit
experiment because disabling it improves runtime but risks missing hard surface
targets.

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

## Next Meaningful Work

1. Integrate the acquisition/null selector behind explicit flags in
   `tbd_motion_detector.py`, then evaluate it on complete videos.
2. Improve the frame router so it does not over-classify e271/aaf1-like ridge
   or horizon clips as surface.
3. Keep collecting true target-over-tree/grass/terrain labels; route ambiguous
   d129-like frames to human review.
4. Keep the state-machine selector path: learned ranker only in states where
   LOCO shows benefit.
5. Train null-aware tube rankers with explicit no-target/hallucination
   negatives.
