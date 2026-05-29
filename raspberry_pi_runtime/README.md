# Raspberry Pi Runtime

Target: Raspberry Pi 5, 8 GB RAM, live 30 Hz guidance path.

This folder contains the lightweight deployment-facing wrappers and benchmark
tools. It does not fork the detector logic; it calls
`scripts/tbd_motion_detector.py` with bounded candidate/beam settings so the
core algorithm remains testable in one place.

## Profiles

`run_pi_detector.py` exposes three profiles:

- `pi_light_live`: fastest, low recall. Useful as a floor.
- `pi_balanced_live`: bounded candidate count plus limited `large_dark`.
- `pi_quality_live`: heavier quality/audit candidate. It keeps more proposal
  paths, but the multi-clip benchmark shows it is not a safe Pi default.

All profiles:

- process at `downscale=0.5`;
- disable debug frame output;
- avoid sklearn/learned-ranker inference in the live loop;
- bound `beam_width`, `top_k_candidates`, map peaks, and large-dark peaks.

## Current Recommendation

Use `pi_light_live` as the current engineering floor, not as a proven flight
profile. It is the only profile with a realistic chance of fitting 30 Hz under
the conservative Mac-to-Pi proxy, and on the current multi-clip labels it also
beats the heavier profiles on weighted strict recall because it avoids some
clutter overgeneration. It is still not good enough as a final guidance profile.

Use `pi_balanced_live --live_sequence` for delayed quality experiments. It uses
the in-detector incremental continuity selector over top-20 states with a
60-frame output delay. That delay is too high for final guidance, but it is the
best current way to test how much continuity can recover without sklearn or
offline post-processing.

Use `pi_quality_live` only for offline audits or known easy/sky clips where
extra proposal recall matters more than runtime.

For the dense e271 benchmark:

| Path | Strict | Loose | Mac avg | Mac p90 |
| --- | ---: | ---: | ---: | ---: |
| `pi_light_live` | 47.9% | 50.6% | 6.76 ms | 8.50 ms |
| `pi_balanced_live` | 62.4% | 72.3% | 11.32 ms | 13.37 ms |
| `pi_quality_live` | 63.2% | 72.7% | 12.42 ms | 14.69 ms |
| `pi_balanced_live` + live sequence 60 | 80.1% | 97.7% | 12.94 ms | 16.21 ms |
| `pi_quality_live` + verified sequence replay | 79.8% | 97.9% | 18.54 ms export pass | 22.89 ms |

The live sequence result is the important architecture signal: high recall does
not require the heavy learned ranker in the Pi hot loop. A cheap verified-score
continuity selector over top-20 states recovers most of the quality. The current
version is incremental and bounded, but the 60-frame output delay is a real
guidance limitation.

Post-audit sequence-window check on e271:

| Window | Strict | Loose | Output rate | Mac p90 | Pi proxy p90 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 75.5% | 94.1% | 99.7% | 15.26 ms | 39.63 ms |
| 30 | 78.0% | 96.6% | 99.1% | 15.13 ms | 39.30 ms |
| 60 | 80.1% | 97.6% | 98.1% | 14.92 ms | 38.80 ms |

That makes the tradeoff explicit: shorter windows are better for control-loop
latency but lose recall, and none of these are Pi p90-proven under the proxy.

A selector-only sweep on the same export found a small e271 gain at
`max_jump_px=10`, `transition_weight=1.5`, `threshold=-5`: 80.0% strict and
98.3% loose. I am not making that the default because it is less conservative
for no-object/null frames.

For the current 9-clip label benchmark:

| Profile | Weighted strict | Weighted loose | Mean Mac avg | Mean Mac p90 | Conservative Pi avg fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi_light_live` | 62.1% | 65.7% | 11.29 ms | 14.63 ms | 6/9 clips |
| `pi_balanced_live` | 54.7% | 66.1% | 18.30 ms | 23.19 ms | 1/9 clips |
| `pi_quality_live` | 54.9% | 65.8% | 19.82 ms | 25.41 ms | 1/9 clips |
| `pi_balanced_live` + live sequence 60 | 71.8% | 88.3% | 20.33 ms | 25.71 ms | 0/9 clips |

This is not a final accuracy result; it is a deployment-shaping result. The hard
surface clips still miss badly, especially `aaf1`, `d129`, and parts of
`7bd`. Those are proposal/ranking failures. Making the runtime heavier does not
fix them.

## Commands

Live-style pass:

```bash
python raspberry_pi_runtime/run_pi_detector.py \
  /path/to/video.MP4 \
  --output_dir artifacts/pi_run \
  --profile pi_light_live
```

Camera smoke pass with live selected-box telemetry:

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

Create a clean Pi bundle:

```bash
python raspberry_pi_runtime/make_pi_bundle.py \
  --out artifacts/pi_runtime_bundle/fpv-drone-vision-tracking-pi.tar.gz \
  --force
```

Gate a run directory:

```bash
cp /tmp/fpv-tracker-telemetry.jsonl /tmp/fpv-tracker-smoke/telemetry.jsonl
python raspberry_pi_runtime/production_gate.py \
  --run_dir /tmp/fpv-tracker-smoke
```

The gate checks bounded summary mode, stream-only output, per-frame telemetry,
and p95/p99/max wall-time tails. It is a deployment sanity check, not a
substitute for a real Pi camera/thermal soak.

Deferred quality pass:

```bash
python raspberry_pi_runtime/run_pi_detector.py \
  /path/to/video.MP4 \
  --output_dir artifacts/pi_run_deferred \
  --profile pi_quality_live \
  --deferred_sequence
```

Delayed live-sequence quality pass:

```bash
python raspberry_pi_runtime/run_pi_detector.py \
  /path/to/video.MP4 \
  --output_dir artifacts/pi_run_sequence \
  --profile pi_balanced_live \
  --live_sequence \
  --sequence_window 15
```

Benchmark profiles:

```bash
python raspberry_pi_runtime/benchmark_pi_profiles.py \
  --video /path/to/video.MP4 \
  --labels artifacts/surface_dense_label_expansion_v1/labels_plus_dense_surface_v1.csv \
  --out_dir artifacts/pi_benchmark \
  --live_sequence \
  --production_gate
```

`--production_gate` makes each run use `--stream_only`, writes selected and
per-frame telemetry JSONL in the run directory, converts selected JSONL back to
CSV for label evaluation, then appends gate pass/fail and failed-check columns
to `pi_profile_benchmark.csv`. It is the preferred benchmark mode for live Pi
readiness. Deferred sequence replay is still a lab/offline mode because it
needs `top_tubes.csv`.

## Honest Runtime Caveat

The Mac-side benchmark is not a Pi 5 benchmark. `benchmark_pi_profiles.py`
reports a conservative proxy:

```text
pi_estimate_ms = mac_ms * 2.4 + 3.0
```

Treat this as a filter for bad configs, not as deployment proof. A real Pi 5
run must still measure camera ingest, decode, thermals, and p90/p99 timing.

For onboard setup, see `PI_DEPLOYMENT.md`. Do not copy `artifacts/` to the Pi;
the runtime deployment should be scripts plus `raspberry_pi_runtime/` only.
The Pi wrapper defaults to bounded `--report_mode summary`; use
`--report_mode full` only for lab diagnostics.

## Next Embedded Step

Reduce the delayed selector latency and add a Pi-side benchmark run:

- compare 15/30/60-frame live sequence windows on real Pi 5 hardware;
- keep `pi_light_live` as the hard realtime fallback;
- use `pi_balanced_live --live_sequence` as the current quality ceiling;
- do not port to Rust until the surface proposal/ranking failure is improved.
