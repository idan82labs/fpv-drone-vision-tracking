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

## Next Meaningful Work

1. Collect or generate true target-over-tree/grass/terrain labels; route ambiguous d129-like frames to human review.
2. Improve the background router until it cleanly separates sky, skyline, cloud texture, and true surface-backed targets.
3. Keep the state-machine selector path: learned ranker only in states where LOCO shows benefit.
4. Train a tube-level ranker on hard top-tube alternatives after the surface-positive dataset is less biased.
5. Add target-aligned versus background-aligned crop-stack features.
