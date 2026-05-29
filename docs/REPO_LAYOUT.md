# Repository Layout and Promotion Rules

This repo has two deliberate tracks: desktop research and Raspberry Pi runtime.
They share the detector core, but they should not share generated artifacts or
unbounded experiment state.

## Production / Raspberry Pi Track

Deployment-facing files live in:

```text
raspberry_pi_runtime/
scripts/tbd_motion_detector.py
scripts/motion_detector_v2.py
config/
```

`raspberry_pi_runtime/` owns the Pi wrapper, service files, bundle creation,
runtime profiles, telemetry output, and production gate. It should stay small
enough to reason about and ship.

The Pi bundle is intentionally minimal:

```bash
python raspberry_pi_runtime/make_pi_bundle.py \
  --out artifacts/pi_runtime_bundle/fpv-drone-vision-tracking-pi.tar.gz \
  --force
```

The bundle must not include `artifacts/`, `deploy_assets/`, raw videos, review
packets, caches, or training outputs.

## Desktop Lab Track

Desktop/offline work lives in:

```text
scripts/
web/tube_labeler/
tests/
docs/
artifacts/       # ignored local output
deploy_assets/   # ignored local review/deploy data
models/          # ignored promoted model files unless released separately
```

The desktop side can run heavier proposal sweeps, labeler flows, learned-ranker
training, renders, and failure mining. Its purpose is to discover behavior with
less constrained hardware, then distill stable pieces into bounded Pi profiles.

## Promotion Criteria

Before moving a desktop idea into the Pi path, require:

1. A named artifact or report in `docs/STATUS.md`.
2. Held-out or leave-one-clip-out behavior, not only all-fit numbers.
3. Null/no-target evaluation, not only visible-drone recall.
4. Per-frame runtime timing with p95/p99/max tails.
5. A production-gate pass for the relevant runtime profile.
6. Documentation of latency tradeoffs, especially sequence-window delay.

## Rust Boundary

Do not rewrite the system in Rust yet. First stabilize branch behavior and the
surface/background router in Python/OpenCV. Rust becomes useful later for stable
hot paths only: NMS, candidate scoring, box/ring statistics, tube update, and
router feature extraction.

## Artifact Policy

Generated outputs stay local and ignored. Keep the source repo useful to a new
engineer by committing code, tests, docs, and small manifests, not videos,
contact sheets, raw footage, or one-off model dumps.
