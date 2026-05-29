# Raspberry Pi Deployment Notes

Status: engineering scaffold, not flight-ready.

## What Goes On The Pi

Copy only the runtime code and minimal dependencies:

- `scripts/tbd_motion_detector.py`
- `scripts/motion_detector_v2.py`
- `raspberry_pi_runtime/`
- `raspberry_pi_runtime/requirements-pi.txt`

Do not deploy lab artifacts, review packets, rendered demos, or training outputs. The current `artifacts/` tree is multi-GB and belongs on the workstation.

Build a clean runtime tarball from the workstation:

```bash
python raspberry_pi_runtime/make_pi_bundle.py \
  --out artifacts/pi_runtime_bundle/fpv-drone-vision-tracking-pi.tar.gz \
  --force
```

The tarball contains `bundle_manifest.json` with file sizes and SHA-256 hashes.
It should include only `scripts/tbd_motion_detector.py`,
`scripts/motion_detector_v2.py`, and `raspberry_pi_runtime/`.

## Setup

```bash
cd /opt/fpv-drone-vision-tracking
python3 -m venv .venv
. .venv/bin/activate
pip install -r raspberry_pi_runtime/requirements-pi.txt
python -m py_compile scripts/tbd_motion_detector.py raspberry_pi_runtime/run_pi_detector.py
```

Smoke test a camera device:

```bash
python raspberry_pi_runtime/run_pi_detector.py camera:0 \
  --output_dir /tmp/fpv-tracker-smoke \
  --profile pi_light_live \
  --max_frames 120 \
  --selected_jsonl /tmp/fpv-tracker-selected.jsonl \
  --telemetry_jsonl /tmp/fpv-tracker-telemetry.jsonl \
  --stream_only
```

Gate the smoke output:

```bash
cp /tmp/fpv-tracker-telemetry.jsonl /tmp/fpv-tracker-smoke/telemetry.jsonl
python raspberry_pi_runtime/production_gate.py \
  --run_dir /tmp/fpv-tracker-smoke \
  --budget_ms 33.3 \
  --max_wall_ms 100
```

Passing this gate is not flight readiness by itself; it only proves that the run
used bounded reporting, produced per-frame telemetry, and stayed inside the
configured latency budget for that test.

`camera:0` maps to `cv2.VideoCapture(0)`. File paths still work. GStreamer/libcamera/Picamera2 capture has not been validated in this repo yet; that is a blocker before drone integration.

## Optional systemd install

```bash
sudo cp raspberry_pi_runtime/fpv-tracker.env.example /etc/fpv-tracker.env
sudo cp raspberry_pi_runtime/fpv-tracker.service /etc/systemd/system/fpv-tracker.service
sudo systemctl daemon-reload
sudo systemctl enable fpv-tracker
sudo systemctl start fpv-tracker
```

The service uses `--stream_only`, writes selected boxes incrementally to `$FPV_SELECTED_JSONL`, and writes one status record per processed frame to `$FPV_TELEMETRY_JSONL`. The per-frame telemetry stream is the safer integration surface for a guidance process because it includes no-target/warmup status and latency. Lab CSV/report files are not the live control interface.

## Current Blockers

- Real Pi 5 camera/decode/thermal p95/p99 timing has not been measured.
- `pi_balanced_live --live_sequence` still has a 60-frame output delay, which is too high for guidance.
- Surface clips remain algorithmically weak; runtime work does not fix that by itself.
- Long-run summary mode is bounded, but the camera backend is still an OpenCV shim and the service has not been soak-tested on real Pi hardware.
