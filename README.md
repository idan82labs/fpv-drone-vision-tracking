# FPV Drone Vision Tracking

This repository contains the source code, tests, docs, labeler, and Raspberry Pi
runtime wrappers for the FPV drone tracking project.

The repo is intentionally source-first. Generated videos, contact sheets, raw
footage, trained model dumps, labeling packets, and benchmark sweeps stay local
under ignored directories such as `artifacts/`, `deploy_assets/`, and `models/`.

## Honest Status

This is not a solved autonomous tracker yet.

- Best current architecture: cheap high-recall proposals plus short-window
  continuity/track-before-detect selection.
- Best current deployment signal: lightweight sequence selection can recover
  strong e271 continuity without the heavy learned ranker in the hot loop.
- Main blocker: tree/grass/terrain and skyline clutter still beat the true
  drone in hard clips.
- Production readiness: `raspberry_pi_runtime/PRODUCTION_READINESS.md` rates
  the current Pi path at **4/10**. Runtime scaffolding is improving; physical
  validation and hard-surface tracking are still not production-grade.
- Rust is not the next step. Stabilize Python/OpenCV branch behavior first,
  then port measured hot paths if needed.

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
models/               Ignored promoted model files and calibration assets.
```

See `docs/REPO_LAYOUT.md` for the production/Desktop split and promotion rules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the regression tests:

```bash
python -m unittest discover -s tests -v
```

## Desktop Lab Path

Use the desktop path for labeling, training, evaluation, rendering, and failure
analysis. It is allowed to be heavier than the Pi runtime because the goal is to
discover what works before distilling it.

Run the labeling website locally:

```bash
python scripts/tube_labeling_server.py \
  --host 127.0.0.1 \
  --port 8768 \
  --csv deploy_assets/tube_hard_negative_review_packet_thr060_top8/tube_alternatives_to_label.csv \
  --video_dir deploy_assets/videos \
  --app_dir web/tube_labeler
```

Open `http://127.0.0.1:8768/`.

The server writes labels back to the CSV atomically and creates timestamped
backups beside the CSV.

## Raspberry Pi Runtime Path

The Pi path lives in `raspberry_pi_runtime/`. It does not fork detector logic;
it calls the shared detector with bounded settings.

Create a clean Pi bundle:

```bash
python raspberry_pi_runtime/make_pi_bundle.py \
  --out artifacts/pi_runtime_bundle/fpv-drone-vision-tracking-pi.tar.gz \
  --force
```

Live-style pass:

```bash
python raspberry_pi_runtime/run_pi_detector.py \
  /path/to/video.MP4 \
  --output_dir artifacts/pi_run \
  --profile pi_light_live
```

Camera smoke pass with selected-box telemetry:

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

The gate checks reporting mode, stream-only output, telemetry coverage, and
p95/p99/max latency tails. It is a deployment sanity check, not proof of flight
readiness.

## Data and Artifact Policy

Do not commit generated experiment output or raw footage.

- `artifacts/` is local generated output.
- `deploy_assets/` is local review/labeler/deployment data.
- `models/` is local promoted model/calibration storage.
- Stable results should be summarized in `docs/STATUS.md` or a small manifest.
- Large shareable outputs should be published as release assets or external
  packets, not committed to the source repo.

See `docs/DATA_AND_ARTIFACTS.md` for details.

## Development Notes

- Prefer leave-one-clip-out or held-out evaluation for learned rankers.
- Always include null/no-target frames when judging a selector.
- Treat `vision_assisted` and `vision_assisted_gapfill` rows as weak labels
  until manually reviewed.
- Runtime changes need timing tails, not just average ms/frame.
- Do not copy `artifacts/` or `deploy_assets/` to the Pi.
