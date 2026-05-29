# Production Readiness Audit

Current honest rating: **4/10** for real drone deployment.

This is higher than the first Pi scaffold because the runtime now has bounded
summary reporting, stream-only mode, per-frame telemetry, camera alias smoke
support, and service/dependency scaffolding. It is still not close to 9/10
because the remaining blockers require hardware and algorithm evidence, not
just code cleanup.

## What Is Green

- Pi wrapper defaults to bounded `--report_mode summary`.
- Service path uses `--stream_only`, selected JSONL, and per-frame telemetry JSONL.
- Runtime summaries now expose p95/p99/max wall-time tails.
- `make_pi_bundle.py` creates a minimal Pi deployment tarball with a manifest.
- `production_gate.py` validates bounded reporting, stream-only mode, telemetry coverage, and latency tails for a run directory.
- `camera:0` and numeric camera aliases map into OpenCV capture.
- Sequence window is configurable.
- Render path works with detector-selected CSV schema.
- e271 `pi_balanced_live --live_sequence --sequence_window 60` holds the current
  architecture signal: **80.1% strict / 97.6% loose** on dense e271 labels.

## What Is Not Green

- No real Raspberry Pi 5 camera/decode/thermal p95/p99 benchmark has been run.
- Camera ingest is still an OpenCV `VideoCapture` shim, not validated Picamera2,
  libcamera, or GStreamer with fixed FPS/resolution/exposure/buffer policy.
- Best quality path still has high latency: 15/30/60-frame delayed sequence
  windows trade control latency for recall.
- Hard surface clips remain poor, especially `aaf1` and `d129`; runtime cleanup
  does not solve those algorithmic misses.
- The Pi profiles are not full Mac-baseline parity. They intentionally drop
  heavier proposal/ranker branches to fit runtime.
- No long real-time soak with dropped-frame, restart, thermal, CPU, and memory
  telemetry exists yet.

## Latest Production-Gated Proxy Runs

Artifact roots:

- `artifacts/pi_runtime_sweep_v1/production_gate_e271_v1/`
- `artifacts/pi_runtime_sweep_v1/production_gate_e271_window_30_v1/`
- `artifacts/pi_runtime_sweep_v1/production_gate_e271_window_60_v1/`
- `artifacts/pi_runtime_sweep_v1/production_gate_surface_hard_v1/`
- `artifacts/pi_runtime_sweep_v1/production_gate_surface_hard_light_v1/`

All rows below used bounded summary mode, `--stream_only`, selected JSONL, and
per-frame telemetry JSONL. The gate budget was 33.3 ms wall-time p95/p99.

| Clip | Profile | Window | Gate | Strict | Loose | Mac avg | Wall p99 | Read |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| e271 | `pi_light_live` | 15 | pass | 49.1% | 50.2% | 6.90 ms | 15.46 ms | realtime fallback only |
| e271 | `pi_balanced_live` | 15 | pass | 75.5% | 94.1% | 12.16 ms | 23.01 ms | better latency/recall trade |
| e271 | `pi_balanced_live` | 30 | pass | 78.0% | 96.6% | 11.89 ms | 22.35 ms | near target, 0.6 s delay at 50 Hz |
| e271 | `pi_balanced_live` | 60 | pass | 80.1% | 97.6% | 11.91 ms | 22.12 ms | best e271 recall, too much guidance delay |
| aaf1 | `pi_light_live` | 60 | pass | 8.0% | 8.0% | 16.32 ms | 23.27 ms | runtime OK, tracking poor |
| aaf1 | `pi_balanced_live` | 60 | fail | 0.0% | 25.3% | 26.24 ms | 35.80 ms | surface tracking failure plus p99 miss |
| d129 | `pi_light_live` | 60 | pass | 0.0% | 0.0% | 16.32 ms | 21.73 ms | runtime OK, target not acquired |
| d129 | `pi_balanced_live` | 60 | fail | 0.0% | 0.0% | 26.39 ms | 33.99 ms | null/acquisition failure plus p99 miss |

Interpretation: e271 is now an honest positive benchmark for the lightweight
sequence architecture, but it is not representative enough. The current Pi path
is not production-ready for tree/grass/terrain tracking. The next algorithm
work should reduce surface candidate pressure and improve surface ranking before
any Rust port.

## Gates For 9/10

1. Real Pi 5 run at target camera mode with p95/p99/max latency under budget.
2. Per-frame telemetry consumed by a separate watchdog/guidance process.
3. Camera backend validated with timestamps, frame dropping policy, and fixed exposure.
4. Sequence delay reduced to a control-acceptable window with measured recall loss.
5. Multiclip benchmark regenerated with current output-rate semantics and null/no-target frames.
6. Surface clips reach acceptable recall or are explicitly routed to a different mode.
7. Run the new bundle/gate flow on actual Pi hardware and archive the passing gate report with telemetry.
