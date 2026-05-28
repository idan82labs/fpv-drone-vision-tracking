# FPV Drone Vision Tracking Lab

This repository contains the current FPV drone video tracking work: classical motion proposals, short-window track-before-detect ranking, a browser labeling tool, training utilities, and curated experiment artifacts.

The repository is intentionally curated. The full local workspace had roughly 11 GB of generated outputs; this repo keeps the code, deployment assets, current labels, useful summaries, and selected demo videos.

## Current Status

This is not a solved autonomous tracker yet.

- Best current operating direction: high-recall cheap proposals plus tube-level ranking/verifier.
- Most useful recent progress: temporal-stack proposal recovery finds more visible drone candidates in the proposal pool.
- Main blocker: ranking the real drone tube above cloud, terrain, skyline, and static-hotspot clutter.
- Recent vision-assisted videos show what the target track should look like, but they are not autonomous detector performance.

## Repository Layout

```text
scripts/          Detection, proposal generation, training, calibration, rendering.
web/tube_labeler/ Browser UI for candidate and frame-level labeling.
deploy_assets/   Small seed review packet and compressed review videos for deployment.
artifacts/        Curated labels, summaries, diagnostics, and demo videos.
docs/             Research questions and professor follow-up notes.
config/           Example deployment config.
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Labeling Website Locally

```bash
python scripts/tube_labeling_server.py \
  --host 127.0.0.1 \
  --port 8768 \
  --csv deploy_assets/tube_hard_negative_review_packet_thr060_top8/tube_alternatives_to_label.csv \
  --video_dir deploy_assets/videos \
  --app_dir web/tube_labeler
```

Open `http://127.0.0.1:8768/`.

The server writes labels back into the CSV atomically and creates timestamped backups beside the CSV.

## Current Demo Artifacts

The most useful team-facing clip is:

```text
artifacts/current_demo/e271_vision_assisted_clean_9p0-12p8s_v2.mp4
```

Important caveat: this is vision-assisted target-track data, not autonomous detector output. Use it to inspect the target trajectory and to train/check future proposal recovery, not as a claim of final tracking accuracy.

## Fly.io Deployment

Use the example config, set secrets, then deploy:

```bash
cp config/fly.example.toml fly.toml
fly apps create <app-name>
fly volumes create label_data --region fra --size 1
fly secrets set BASIC_AUTH_PASSWORD='<password>'
fly deploy
```

Do not commit `.fly-basic-auth-password`, `.env`, or a real shared password.

## Development Notes

- Keep raw full-resolution footage out of Git unless it has been intentionally compressed and curated.
- Keep generated sweeps under local `results/`; copy only stable summaries into `artifacts/`.
- Treat `vision_assisted` and `vision_assisted_gapfill` rows as weak labels unless manually reviewed frame by frame.
- Prefer leave-one-clip-out validation for learned rankers; random row splits overstate performance.

