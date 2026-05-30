# FPV Drone Vision Tracking

Client-facing repository for a small-drone visual tracking research program.

The goal is to track a small FPV-style drone in monocular video, including hard
cases where the target is only a few pixels wide and appears against sky,
skyline, trees, grass, terrain, roads, or clouds. The repository contains the
detector core, labeling tools, training/evaluation harnesses, and a bounded
Raspberry Pi 5 runtime scaffold.

This is an honest engineering prototype, not a finished production tracker. The
current architecture is promising in some operating modes, but hard-surface
tracking and onboard validation are still active research work.

## Current Status

The system is useful for research, labeling, benchmark generation, and offline
algorithm iteration. It is not yet production-ready for live safety-critical
operation.

Current honest read:

- Strongest current direction: cheap high-recall proposals plus short-window
  track-before-detect selection.
- Best practical production shape: route between lightweight sky/continuous
  tracking and stricter surface/null-risk behavior instead of using one global
  profile.
- Main technical blocker: tree, grass, terrain, cloud, and skyline clutter can
  still win over the true drone in hard clips.
- Raspberry Pi readiness: the Pi path is scaffolded and measurable, but rated
  **4/10** for real deployment until tested on actual Pi 5 camera hardware with
  latency, thermal, and hard-surface tracking evidence.
- Rust rewrite is intentionally deferred. The Python/OpenCV state machine and
  router behavior must be stable before porting hot paths.

## How We Work

The project is run as an evidence loop:

1. Generate high-recall candidate boxes from real footage.
2. Export top tube alternatives, not just the selected box.
3. Label true target, false competitors, null frames, and hard clutter classes.
4. Train or tune only on labeled alternatives with leave-one-clip-out checks.
5. Evaluate full-video strict/loose hits, no-target no-box rate, and runtime
   tails.
6. Promote only changes that improve both tracking quality and false-lock
   behavior without hiding latency cost.

We separate two tracks:

- **Desktop lab**: heavier research path for labeling, model training,
  failure mining, visual review, and rendering.
- **Raspberry Pi runtime**: bounded, minimal deployment path that calls the
  shared detector with constrained settings and telemetry.

Generated videos, raw footage, model dumps, review packets, and benchmark
sweeps stay out of Git under ignored directories such as `artifacts/`,
`deploy_assets/`, and `models/`. Stable conclusions are summarized in docs.

## Core Method

The detector treats the problem as tiny-target tracking under ego-motion and
clutter, not as single-frame object detection.

### 1. Ego-Motion Compensation

For adjacent frames, the camera/background motion is estimated with an affine or
homography-like transform:

```text
x'_bg ~= H_t x_bg
```

After warping the previous frame into the current frame, the residual image is:

```text
R_t(x) = I_t(x) - I_{t-1}(H_t^{-1} x)
```

This removes much of the camera motion, but not all parallax, rolling-shutter
error, tree motion, terrain texture, glare, or skyline boundaries. Those
residuals are exactly where many false positives come from.

### 2. High-Recall Proposal Generation

The proposal layer intentionally over-generates. It uses:

- motion residuals after ego-motion compensation;
- compact dark/blob cues;
- multiscale map evidence;
- native-resolution micro-candidate checks;
- optional large-dark and temporal-stack candidates for hard cases.

The proposal layer is judged by oracle recall: whether the real drone appears
anywhere in the top alternatives. It is not expected to be precise by itself.

### 3. Track-Before-Detect Selection

Tiny drones often cannot be separated from clutter in a single frame. The
selection layer keeps short candidate tubes and scores them over time.

A simplified tube state is:

```text
s_t = (x_t, y_t, w_t, h_t, v_x, v_y, hits, misses, score)
```

The tracker scores a path by combining image evidence and motion plausibility:

```text
S(path) =
  sum_t observation(candidate_t)
  - motion_cost(v_t, a_t)
  - miss_cost
  - clutter/null_cost
```

The practical implementation is a candidate-based beam/Viterbi-style tracker,
not a dense full-image dynamic program. This keeps compute bounded and makes
failures inspectable.

### 4. Target-vs-Background Evidence

The strongest current research direction compares two explanations for the same
candidate:

```text
target-aligned tube quality
minus
background-aligned / clutter-aligned tube quality
```

For a candidate path, we compare compact-object evidence along the target path
against evidence at the stabilized background location:

```text
D_TB = Q(target-aligned crops) - Q(background-aligned crops)
```

The intended observation model is:

```text
obs(candidate) =
  O_target
  - logsumexp(O_static_bg, O_attached_edge, O_skyline_boundary, O_noise)
```

Current evidence says this signal is real but not sufficient as a standalone
selector. The best crop-stack verifier works as an additional feature family in
the ranker; simple pairwise-linear and source/box-size variants did not produce
a global win.

### 5. Routing and State

No single profile works across all scenes. The runtime direction is
state-conditioned:

- clean sky: fast permissive tracking with low clutter pressure;
- skyline/boundary: conservative silhouette handling;
- surface-backed: stricter target-vs-clutter ranking;
- close/large target: larger box bank and native-resolution checks;
- unknown/low-confidence: conservative fallback and diagnostics.

The candidate-local router is more important than a broad frame-level router:
a branch tip in sky and a real target near a ridge need different behavior even
inside the same frame.

## Operating Profiles

The repo uses "profiles" to describe different quality/runtime tradeoffs.

| Profile family | Purpose | Current read |
| --- | --- | --- |
| Desktop lab | Training, sweeps, labeling, review packets, long rendered demos | Best for finding what works; not bounded for embedded use. |
| Baseline / pair-rescue | Cheap proposals plus short-window continuity | Current conservative default family. |
| Crop-stack / surface ranker | Adds target-aligned/background-aligned crop features for hard alternatives | Useful as a feature family, not a standalone selector. |
| HMM / null-risk selector | More conservative no-box behavior for surface/null-heavy clips | Good for suppressing clutter, unsafe as a global default because it can hurt continuous-visible recall. |
| `pi_light_live` | Lightweight Pi-facing runtime profile | Fast and bounded, but lower recall. |
| `pi_balanced_live` | Higher-quality Pi-facing profile | Better e271 sequence recall, but higher latency and not hard-surface-ready. |

The production architecture should route between these behaviors rather than
promoting the strictest or heaviest mode globally.

## Raspberry Pi 5 Readiness

The Pi runtime lives in `raspberry_pi_runtime/`. It does not fork detector
logic; it wraps the shared detector with bounded reporting, telemetry, and
profile settings.

What is green:

- minimal Pi bundle generation;
- stream-only service path;
- selected-box JSONL and per-frame telemetry JSONL;
- runtime summaries with p95/p99/max wall-time;
- production gate for bounded reporting, telemetry coverage, and latency tails;
- camera alias smoke support through OpenCV capture;
- proxy benchmarks on local video files.

What is not green:

- no real Pi 5 camera/decode/thermal p95/p99 benchmark has been run;
- camera ingest is still an OpenCV shim, not validated Picamera2/libcamera or
  GStreamer with fixed exposure/FPS/buffer policy;
- best e271 recall uses delayed sequence windows that may be too latent for
  live control;
- hard surface clips such as tree/grass/terrain remain algorithmically weak;
- Pi profiles intentionally drop heavier branches and are not Mac-baseline
  parity;
- no long real-time soak test with dropped-frame and watchdog behavior exists.

Current Pi readiness rating: **4/10** for real deployment.

Recent proxy evidence:

| Clip | Profile | Window | Gate | Strict | Loose | Wall p99 | Read |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| e271 | `pi_light_live` | 15 | pass | 49.1% | 50.2% | 15.46 ms | realtime fallback only |
| e271 | `pi_balanced_live` | 15 | pass | 75.5% | 94.1% | 23.01 ms | better latency/recall trade |
| e271 | `pi_balanced_live` | 60 | pass | 80.1% | 97.6% | 22.12 ms | best e271 recall, too much delay |
| aaf1 | `pi_light_live` | 60 | pass | 8.0% | 8.0% | 23.27 ms | runtime OK, tracking poor |
| d129 | `pi_light_live` | 60 | pass | 0.0% | 0.0% | 21.73 ms | runtime OK, target not acquired |

Interpretation: e271 proves the lightweight sequence architecture can be useful,
but the hard-surface clips prove this is not deployment-ready.

## Repository Layout

```text
raspberry_pi_runtime/ Pi-facing wrappers, profiles, service files, bundle tool.
scripts/              Detector core plus desktop training/evaluation tools.
web/tube_labeler/     Browser labeling UI.
tests/                Unit and regression tests.
docs/                 Architecture, status, data policy, runtime plans.
config/               Example deployment config.
artifacts/            Ignored local experiment output.
deploy_assets/        Ignored local labeler/review/deploy assets.
models/               Ignored local promoted model files and calibration assets.
```

See `docs/REPO_LAYOUT.md` for the production/Desktop split and promotion rules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Local Labeling Website

Run the labeling website locally:

```bash
python scripts/tube_labeling_server.py \
  --host 127.0.0.1 \
  --port 8768 \
  --csv deploy_assets/tube_hard_negative_review_packet_thr060_top8/tube_alternatives_to_label.csv \
  --video_dir deploy_assets/videos \
  --app_dir web/tube_labeler
```

Open:

```text
http://127.0.0.1:8768/
```

The labeler writes labels back to CSV atomically and keeps timestamped backups.

## Raspberry Pi Runtime Commands

Create a clean Pi bundle:

```bash
python raspberry_pi_runtime/make_pi_bundle.py \
  --out artifacts/pi_runtime_bundle/fpv-drone-vision-tracking-pi.tar.gz \
  --force
```

Run a live-style video pass:

```bash
python raspberry_pi_runtime/run_pi_detector.py \
  /path/to/video.MP4 \
  --output_dir artifacts/pi_run \
  --profile pi_light_live
```

Run a camera smoke pass:

```bash
python raspberry_pi_runtime/run_pi_detector.py \
  camera:0 \
  --output_dir /tmp/fpv-tracker-smoke \
  --profile pi_light_live \
  --max_frames 120 \
  --selected_jsonl /tmp/fpv-tracker-selected.jsonl \
  --telemetry_jsonl /tmp/fpv-tracker-telemetry.jsonl \
  --stream_only
```

Gate a run directory:

```bash
cp /tmp/fpv-tracker-telemetry.jsonl /tmp/fpv-tracker-smoke/telemetry.jsonl
python raspberry_pi_runtime/production_gate.py \
  --run_dir /tmp/fpv-tracker-smoke
```

The production gate is a deployment sanity check. It is not proof of flight or
field readiness.

## Evaluation Standards

A new algorithmic idea is not accepted from a highlight video alone. Required
evidence should include:

- full-video selected-box metric, not only sparse checkpoints;
- strict and loose hit rates;
- visible-frame misses and no-target no-box rate;
- oracle@K separated from selected@K;
- leave-one-clip-out or held-out clip behavior for learned models;
- runtime p50/p90/p95/p99/max by stage;
- visual review sheets for worst misses and worst false locks;
- exact command/config/artifact path.

## Data and Artifact Policy

Do not commit generated experiment output or raw footage.

- `artifacts/` is local generated output.
- `deploy_assets/` is local review/labeler/deployment data.
- `models/` is local promoted model/calibration storage.
- Stable results should be summarized in `docs/STATUS.md` or a small manifest.
- Large shareable outputs should be published as release assets or external
  packets, not committed to the source repo.

See `docs/DATA_AND_ARTIFACTS.md` for details.

## Next Technical Milestones

The most valuable next work is:

1. Improve target-vs-background observation for hard surface clips.
2. Add more typed hard-negative labels for tree, grass, terrain, cloud, skyline,
   and false large-dark patches.
3. Make routing candidate-local and conservative enough for live use.
4. Reduce sequence-window latency while preserving recall.
5. Run real Pi 5 camera benchmarks with telemetry, thermal, and watchdog
   behavior.

Only after those pass should we consider porting stable hot paths to Rust or
C++ for deployment.

The concrete experiment ladder is tracked in
`docs/IMPROVEMENT_LOOP_PLAN.md`.
