# Against-Ground Score-Gated Recovery Plan

Date: 2026-05-31

Purpose: turn the current multi-agent critique into a concrete loop for real
against-ground tracking. This plan is intentionally stricter than previous
"ground" artifacts because the latest visual audit corrected a major mistake:
the strong aaf1 288-640 segment is mostly sky/skyline-backed, not true
ground-backed.

## Executive Read

The current project has useful surface/skyline tracking evidence, but not a
validated against-ground tracker.

The biggest gaps are:

1. Label truth is mixed: `bg_split` often describes scene context, not the
   pixels behind the target.
2. aaf1 288-640 / 480-640 must be moved to `skyline_above_terrain`, not
   `true_ground`.
3. The strongest aaf1 branch is a teacher/proof path: direct surface/ranker
   evidence is high on aaf1 but collapses on e6 when used globally.
4. Offline and runtime selection still drift: surface routing, Viterbi
   backfill, learned-logit priors, and delayed ranker candidate retention are
   not yet one shared production path.
5. Renders are demos only. Promotion requires raw per-frame eval rows, exact
   commands, leakage class, label hashes, and timing.

## Hard Definitions

Every promoted label row must separate:

```text
frame_context:
  sky / skyline / tree_line / vegetation / terrain / road / mixed / unknown

target_backdrop:
  clean_sky
  cloud_sky
  skyline_above_terrain
  vegetation
  tree_canopy
  terrain
  road
  mixed_ground
  unknown

audit_status:
  visual_confirmed
  contact_sheet_reviewed
  interpolated
  weak_vision_assisted
  rejected
```

For "true against-ground" metrics:

```text
target_backdrop in {vegetation, tree_canopy, terrain, road, mixed_ground}
audit_status in {visual_confirmed, contact_sheet_reviewed}
visible == 1
```

Explicit exclusion:

```text
aaf1 frames 288-640 and 480-640 are skyline_above_terrain unless re-audited
frame-by-frame and proven otherwise.
```

## Phase 0: Evidence Hygiene Gate

Goal: prevent another false "ground" claim before optimizing.

Actions:

1. Add a backdrop correction manifest for known mislabeled artifacts.
2. Extend label/eval schema with `target_backdrop`, `frame_context`,
   `audit_status`, and `label_provenance`.
3. Add an evidence class to every new artifact:

```text
all_fit
interleaved_same_clip
blocked_same_clip
one_clip_oof
loco
nested_loco
frozen_holdout
pi_live
```

Gate:

```text
0 known aaf1 288-640 / 480-640 rows included in true_ground metrics
100% promoted metrics include evidence_class
100% promoted label rows include target_backdrop
0 critical backdrop mismatches in promoted contact sheets
<= 2% box-center QA failures in promoted label packets
```

Kill rule:

If a result cannot say what the target is backed by, it is not allowed to be
called against-ground.

## Phase 1: True-Ground Label Packet

Goal: build an actually representative target-backed-by-ground set before
training or tuning.

Highest-value sources:

```text
e6 true tree/terrain frames
7bd / 59e / b96 ground recovery clips
new Pi camera clips with target over road/trees/terrain
e271 only if the target box itself is ground-backed
```

Immediate packet work:

1. Finish labels for:

```text
artifacts/current_router_failure_packets_v1/aaf1_null_false_top_tubes_review/tube_alternatives_to_label.csv
artifacts/current_router_failure_packets_v1/e271_visible_miss_top_tubes_review/tube_alternatives_to_label.csv
```

2. Add a true-ground positive packet:

```text
>= 300 visible target-backed-by-ground frames
>= 3 clips
no single clip > 40% of visible true-ground rows
>= 150 null/hard false-lock surface frames
```

3. For each visible frame, label top alternatives:

```text
T  true target
S  static background hot spot
E  attached branch/tree/terrain/grass/edge
H  skyline/horizon/cloud/parallax boundary
G  generic clutter
UNK ignore for subclass CE
```

Gate:

```text
coverage: >= 3 clips, no clip > 40%
visible true_ground rows: >= 300
hard null/surface rows: >= 150
taxonomy fill rate on top alternatives: >= 85%
visual contact sheets: present for every packet
```

Kill rule:

If no true-ground positive frames are found in a candidate clip, move that clip
to skyline/null/regression, not ground training.

## Phase 2: Proposal Availability Gate

Goal: determine whether the target exists in candidates before touching rankers.

Run oracle@K separately for:

```text
true_ground
skyline_above_terrain
hard_null_surface
```

Candidate streams to compare:

```text
current_runtime top tubes
surface-stack teacher
recentered surface branch
bounded candidate-local approximation
prop/offaxis augmented diagnostics
```

Gate before selector work:

```text
true_ground top-80 oracle >= 80% strict
true_ground top-80 oracle >= 95% loose
skyline_above_terrain reported separately
hard_null candidate explosion does not increase selected false alternatives
```

If current runtime oracle is low but surface-stack oracle is high:

```text
problem = proposal recovery / branch admission
next = bounded recentered-surface branch
```

If oracle is high but selection is poor:

```text
problem = observation/routing
next = multi-class JS1/logit selector
```

Kill rule:

If a feature improves ranking but not oracle or selected recall on true-ground,
keep it as diagnostic only. Current prop/offaxis features fall here until proven
otherwise.

## Phase 3: Production-Shaped Routing Fixes

Goal: make offline wins correspond to runtime behavior.

Implementation fixes to do before claiming a new score:

1. **Shared surface/null-risk policy**
   - Extract the surface-risk gate currently duplicated between
     `scripts/apply_gated_surface_branch.py` and runtime detector logic.
   - Offline replay and runtime telemetry must produce the same gate decisions
     from the same trace/candidate rows.

2. **No-backfill production replay**
   - Offline production-mode evaluation must use streaming/no-backfill semantics.
   - Do not score disconnected Viterbi prefixes as if live output would emit
     them.

3. **Delayed surface-ranker fallback**
   - When surface-ranker scoring is active, retain fallback verified-score
     candidates instead of dropping non-surface-scope candidates.

4. **Learned-logit null priors**
   - Ensure explicit `--null_priors` affect learned-logit observation mode.

Gate:

```text
offline/live gate decisions identical on fixture: 100%
production replay vs verified_sequence_selector selected/no-box match: 100%
fallback candidate coverage does not regress e271/clean-sky recall by > 2 pp
learned-logit null-prior unit test passes
focused unit tests pass
```

Score gate after fixes:

```text
true_ground strict improves over current production-shaped baseline by >= 10 pp
true_ground wrong-loose <= 5%
hard_null no-box >= 90%
non-targeted clip strict/loose regression <= 2 pp
```

Kill rule:

If the direct surface branch only wins in offline backfilled mode, it remains a
teacher/mining path, not a runtime path.

## Phase 4: Routed Recentered Surface Branch v2

Goal: admit the high-oracle surface/recenter candidates only when the state
machine needs them.

Branch eligibility:

```text
router/state says surface/line/boundary/unknown risk
state in {A, P, C, S, E}
or T has low margin / high clutter logit for >= 2 frames
local surface risk passes threshold
```

Do not globally replace the normal path.

Inputs:

```text
normal top tubes
surface/recentered alternatives
multi-class T/S/E/H/G logits when available
base selector trace as routing context only
```

Gate:

```text
held-out true-ground strict >= base + 20 pp
held-out true-ground loose >= base + 25 pp
selected strict captures >= 50% of oracle gap
e6 strict regression <= 1 pp
e6 loose regression <= 1 pp
easy/sky gate rate < 5%
hard_null no-box regression <= 2 pp
```

Kill rule:

If held-out oracle@K is high but selected recall stays poor, stop scalar/halo
ranker tuning and escalate observation to crop-stack visual logits.

## Phase 5: Multi-Class Observation Upgrade

Goal: replace scalar targetness with a real clutter-vs-target observation.

Model interface:

```text
O_T true moving drone
O_S static background hot spot
O_E attached branch/tree/terrain/edge
O_H skyline/horizon/cloud/parallax
O_G generic clutter
O_N no-target/null window
```

Use inside JS1/shared selector:

```text
E_T = O_T - logsumexp(O_S, O_E, O_H, O_G, O_N)
```

Inputs:

```text
crop-stack features
CLBA features
surface/recenter metadata
router/context features
prop/offaxis confirmation features as auxiliary only
```

Diagnostic gate:

```text
false-lock rows: max(S,E,H,G) > T in >= 80%
true-ground T rows: T > max(S,E,H,G) in >= 75%
pairwise true-vs-selected-false win >= 75%
no-target max-window target score separable by PR curve
```

Replay gate:

```text
aggregate strict >= 70%
aggregate loose >= 82%
hard_null no-box >= 70%
no catastrophic clip:
  true_ground clip strict >= 55%
  e271 strict >= 70%
  e6 strict >= 80%
```

Promotion gate:

```text
aggregate strict >= 80%
aggregate loose >= 90%
hard_null no-box >= 90%
true_ground held-out improvement passes Phase 4 gate
```

Kill rule:

If multi-class logits help all-fit but fail LOCO/held-out, collect more
distributed true-ground clips before increasing model size.

## Phase 6: Validation Packet and Release Gate

Every candidate promotion must create:

```text
metadata.json
summary.csv
by_clip_summary.csv
by_backdrop_summary.csv
by_router_summary.csv
oracle_summary.csv
selected_frame_eval.csv
state_trace.csv
failures.csv
worst_false_locks.csv
worst_misses.csv
timing_summary.csv
README.md
exact_commands.txt
model_checksums.json
label_manifest.json
```

Required score lines:

```text
strict / loose visible recall
no-box on invisible/null
false boxes per minute
oracle@K vs selected@K
longest strict run
longest miss gap
longest wrong-lock dwell
timing p50/p90/p95/p99/max
```

Promotion gate:

```text
evidence_class in {loco, nested_loco, frozen_holdout, pi_live}
not all_fit
not interleaved_same_clip
not unlabeled
```

Render rule:

Renders can illustrate only rows already present in `selected_frame_eval.csv`.
Interpolated/smoothed presentation renders are demos, not validation evidence.

## Phase 7: Pi Gate

Run only after Phase 4/5 passes offline frozen validation.

Pi requirements:

```text
actual Raspberry Pi 5, not Mac proxy
real camera/decode path
causal latency <= 5-9 frames
p95/p99 under 33.3 ms
max under 100 ms
CPU/memory/temp/dropped-frame telemetry
same quality gate on recorded and live-capture inputs
```

Gate:

```text
quality regression vs Mac frozen replay <= 3 pp strict/loose
hard_null no-box regression <= 3 pp
dropped frames reported and below configured cap
thermal throttling absent or explicitly bounded
```

Kill rule:

If Pi fails timing but Mac quality passes, optimize hot loops. If Pi quality
fails while timing passes, do not port more code; return to observation/routing.

## Immediate Next Execution Order

1. Add `target_backdrop` manifest and exclude corrected aaf1 skyline rows from
   true-ground metrics.
2. Patch `evaluate_ground_profile_continuity.py` to require a label manifest
   for `true_ground`.
3. Fix production-shaped replay drift:
   - no-backfill replay gate;
   - shared surface-risk policy;
   - delayed surface-ranker fallback retention;
   - learned-logit null priors.
4. Build a true-ground label packet from e6/7bd/59e/b96/new Pi clips.
5. Run proposal oracle@K by backdrop.
6. Only then run routed recentered surface branch v2 and multi-class JS1 replay.

## Current Standing

The old aaf1 skyline/above-terrain result is useful but must be renamed and
excluded from true-ground claims. The most credible route to real
against-ground progress is not more threshold sweeping. It is:

```text
backdrop-correct labels
+ production-shaped replay parity
+ routed surface/recenter branch
+ multi-class T/S/E/H/G observation
+ frozen held-out validation
```

