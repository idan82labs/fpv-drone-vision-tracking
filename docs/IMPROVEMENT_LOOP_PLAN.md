# Improvement Loop Plan

Date: 2026-05-30

Purpose: convert the current weaknesses into a concrete experiment loop. This
plan is intentionally strict: every idea must either improve measured behavior,
produce better labels for the next pass, or get killed.

## Target State

The system is not "ideal" until it can run as a routed live tracker with:

| Requirement | Target |
| --- | ---: |
| Full-video strict visible hit rate | >= 80% |
| Full-video loose visible hit rate | >= 90% |
| No-target / not-visible no-box rate | >= 90% |
| Continuous-visible e271-style recall | no regression below current best routed baseline |
| Hard-surface aaf1/d129-style recall | large positive move from current weak baseline |
| False branch/tree/terrain locks | visibly reduced in worst-case sheets |
| Live latency at 30 Hz | p95/p99 under 33.3 ms, with measured stage timing |
| Control latency | <= 5-9 frame causal window before any production claim |
| Pi readiness | real Pi 5 camera/decode/thermal run, not Mac proxy only |

These are not marketing targets. If a method improves one line while destroying
another, it stays experimental and routed.

## Current Weaknesses

| Weakness | What the evidence says | What we do next |
| --- | --- | --- |
| Surface clutter beats target | Tree/grass/terrain/cloud/skyline candidates still rank above true drone in hard clips. | Improve target-vs-background observation and collect typed hard negatives. |
| One global selector is unsafe | HMM/no-box behavior helps null-heavy hard clips but hurts continuous-visible e271. | Build candidate-local router and state-conditioned selector, not a global default. |
| Crop-stack signal transfers unevenly | HGBDT crop-stack verifier is strong overall but weak on held-out aaf1/d129. | Mine pair failures and add representative labels/features before new model classes. |
| Pairwise logistic did not fix it | Same-frame pairwise objective collapsed on aaf1/e271 with current features. | Keep pairwise as a future objective only after richer features/labels. |
| Source/box-size cues did not fix it | Source/geometry was descriptive but flat/down in LOCO. | Do not promote as default; use it only as diagnostic context. |
| Null calibration is score-limited | Router null thresholds trade recall for no-box because score is not calibrated enough. | Improve observation first, then rerun null calibration. |
| Pi path is scaffolded, not proven | Proxy timings pass on some modes; no real Pi camera/thermal/soak run. | Algorithm first, then Pi profile gate on actual hardware. |

## Loop 0: Freeze the Baseline Before Each New Idea

Goal: make regression visible before optimizing.

Required output for every experiment:

```text
artifact/
  metadata.json
  summary.csv
  by_clip_summary.csv
  failures.csv
  README.md
```

Minimum report:

1. full-video strict/loose selected-box metrics;
2. no-target/no-box rate;
3. oracle@K split from selected@K;
4. per-clip results for e271, aaf1, d129, e6, and regression clips;
5. timing p50/p90/p95/p99/max if runtime-facing;
6. visual sheet of worst misses and false locks;
7. exact command/config/model/artifact path.

Promotion gate:

```text
Promote only if aggregate improves AND e271 continuous-visible behavior does
not collapse AND null/no-target behavior does not collapse.
```

Kill rule:

```text
If an idea only improves a highlight video or one tuned clip, it is a mining
tool, not a runtime path.
```

## Loop 1: Typed Failure Mining

Goal: make the remaining false competitors explicit enough to train against.

Starting artifact:

```text
artifacts/crop_stack_pair_failure_review_v1/review_packet_top16/
```

Label every top-tube competitor in the hard failure packet with:

```text
target
near_target_wrong_center
static_hotspot
line_attached
parallax_edge
boundary_artifact
appearance_blob
terrain_texture
cloud_speck
large_dark_false
noise
uncertain
```

For each checkpoint, also preserve:

```text
true target box if visible
visibility confidence
router context: clean_sky / skyline / surface / attached / unknown
selected false competitor rank/source
```

Pass gate:

```text
At least 300 new hard-negative rows and at least 100 true/near-target rows
across aaf1, d129, and e271-style failures, with typed clutter labels.
```

Next branch:

- If the true target is often missing from top alternatives, go to Loop 2A
  proposal recovery.
- If the true target is present but ranked below clutter, go to Loop 2B
  observation model.

## Loop 2A: Proposal Recovery

Run only when oracle@K is the bottleneck.

Hypothesis: target is visible but not present in top alternatives often enough.

Experiments:

1. candidate-local temporal stack around top-K proposals;
2. higher-resolution ROI rescoring;
3. bounded large-dark/close-target proposal bank;
4. local residual matched filter for tiny motion events.

Hard limits:

```text
K <= 40 for runtime-facing branch
L <= 5 for live candidate branch
patch <= 31x31
no full-frame temporal stack in live path
```

Pass gate:

```text
oracle@20 or oracle@40 improves materially on hard-surface visible frames
without increasing no-target false alternatives enough to destroy selection.
```

Kill rule:

```text
If oracle improves but selected@K falls, keep the method only as an offline
teacher/miner until Loop 2B improves observation.
```

## Loop 2B: Target-vs-Clutter Observation

Run when target exists in top alternatives but loses to clutter.

Current best direction:

```text
score = O_target - logsumexp(O_static, O_attached, O_boundary, O_noise)
```

Feature blocks to add or improve:

| Block | Concrete feature | Why |
| --- | --- | --- |
| Target/background stack | `Q(target_path) - Q(background_path)` | Tests moving compact target vs stabilized clutter. |
| Attached edge likelihood | connected dark/edge support through/under box | Penalizes branches/poles as an alternate explanation. |
| Boundary/parallax likelihood | one-sided horizon gradient + local residual instability | Handles skyline and ridge artifacts. |
| Local distractor margin | candidate score vs same-context annulus controls | Prevents smooth terrain texture from winning by raw score. |
| Native-resolution compactness | dark-ring / LoG / PSF-like score in ROI | Helps 3-10 px target recognition. |
| Typed clutter priors | learned weights from Loop 1 labels | Converts labels into explicit null explanations. |

First concrete experiment:

```text
Train a typed-clutter observation model using the Loop 1 labels.
Compare:
  A. current HGBDT crop-stack verifier
  B. HGBDT + typed clutter labels
  C. typed O_target - logsumexp(O_clutter_k)
  D. same model inside selector/HMM
```

Pass gate:

```text
On held-out clips:
  pairwise true-vs-false win rate improves on aaf1 and d129;
  e271 does not regress;
  full selector strict/loose and no-box improve together.
```

Kill rule:

```text
If typed clutter labels improve offline pairwise but not selector behavior, the
state model/null calibration is the bottleneck; move to Loop 3.
```

## Loop 3: Explicit State Model

Run when observation is better but the selected track still locks, coasts, or
drops incorrectly.

Target model:

```text
A = absent / no box
P = present but not acquired
T = acquired target
S = static/background lock
E = attached edge/tree/terrain lock
C = coast/lost
```

Candidate observation:

```text
obs(c) =
  weak_proposal_prior(c)
  + target_likelihood(c)
  - logsumexp(static_bg, attached_edge, boundary, noise)
```

State rules:

```text
emit box only in T
allow A/P to birth fresh tracks
allow T -> C on short misses
send repeated static/attached evidence to S/E
quarantine S/E anchors briefly, but allow fresh birth elsewhere
never let one S/E false lock poison the future target identity
```

Pass gate:

```text
Same observation model, selector-only change:
  visible recall improves or holds;
  no-target no-box improves;
  e271 continuous-visible acquisition delay stays acceptable.
```

Kill rule:

```text
If S/E quarantine improves null but kills target recall, quarantine is too
strong or observation is still not target-aware enough.
```

## Loop 4: Production Router

Run only after Loops 2/3 produce a useful hard-surface branch.

Router states:

```text
clean_sky_core
cloud_or_sky_texture
skyline_boundary
surface_texture
attached_linear_structure
high_parallax_boundary
unknown
```

Router controls:

```text
candidate budget
whether surface extras are allowed
which selector family is used
null threshold prior
whether HMM/no-box behavior is allowed
```

Pass gate:

```text
Routed selector beats both global permissive and global conservative selectors:
  keeps e271-style continuous visible recall;
  improves aaf1/d129 hard surface/null behavior;
  does not route skyline/cloud into surface behavior too broadly.
```

Kill rule:

```text
If routing decisions are unstable frame-to-frame, add hysteresis/state memory
before touching detector thresholds again.
```

## Loop 5: Raspberry Pi Runtime Gate

Run only after a profile has real algorithm value.

Profile requirements:

```text
pi_light_live: fast baseline, lower recall accepted
pi_balanced_live: best live candidate, must stay under 30 Hz budget
surface branch: optional, only when router allows and total budget survives
```

Gate commands:

```bash
python raspberry_pi_runtime/run_pi_detector.py camera:0 \
  --output_dir /tmp/fpv-tracker-smoke \
  --profile pi_light_live \
  --max_frames 120 \
  --selected_jsonl /tmp/fpv-tracker-selected.jsonl \
  --telemetry_jsonl /tmp/fpv-tracker-telemetry.jsonl \
  --stream_only

cp /tmp/fpv-tracker-telemetry.jsonl /tmp/fpv-tracker-smoke/telemetry.jsonl
python raspberry_pi_runtime/production_gate.py \
  --run_dir /tmp/fpv-tracker-smoke
```

Pass gate:

```text
actual Pi 5, target camera mode:
  p95/p99 <= 33.3 ms;
  no unbounded report/video writes in stream mode;
  selected JSONL and telemetry JSONL emitted;
  thermal/CPU/memory/dropped-frame soak recorded.
```

Kill rule:

```text
Do not port to Rust until branch behavior is stable and profiling identifies
specific hot paths.
```

## Loop Order

Current priority order:

1. Loop 1: label typed pair-failure packet.
2. Loop 2B: train typed target-vs-clutter observation.
3. Loop 3: insert observation into explicit state selector.
4. Loop 4: route only when hard-surface/null-risk is detected.
5. Loop 5: run Pi gate on the winning routed profile.

Loop 2A proposal recovery runs in parallel only when oracle@K fails.

## Professor Answer Intake

When the professor answer arrives, map each recommendation to one of these
loops:

```text
new observation primitive -> Loop 2B
new proposal mechanism -> Loop 2A
state/HMM advice -> Loop 3
router advice -> Loop 4
runtime/Pi advice -> Loop 5
labeling advice -> Loop 1
```

If the professor suggests a method that bypasses the current loop order, still
run the same promotion gates. A mathematically cleaner idea is not promoted
until it improves held-out full-video behavior and runtime constraints.

## Immediate Next Commanded Work

Before adding another model class:

1. Use `artifacts/crop_stack_pair_failure_review_v1/review_packet_top16/` as the
   primary annotation target.
2. Convert the reviewed rows into typed hard-negative training examples.
3. Train a typed clutter/null observation model.
4. Evaluate it against the current HGBDT crop-stack baseline and current routed
   selector.
5. Produce a new professor drop-off only if the typed observation still fails
   and the failure packet shows why.
