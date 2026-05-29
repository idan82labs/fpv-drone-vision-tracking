# Embedded Runtime Plan

Target machine: Raspberry Pi 5, 8 GB RAM.

Target behavior: live 30 Hz guidance first. 60 Hz can be tested later, but it
should not drive architecture choices until 30 Hz is stable.

## Position

Do not make the heavy hybrid proposal mode the global default.

The current evidence says:

- More candidates help oracle recall.
- More candidates also increase clutter pressure.
- Hybrid/coast proposals help aaf1-style surface frames but hurt clean sky and
  e271-style frames when used globally.
- The learned surface ranker helps only after a router says the candidate/frame
  is truly surface-backed.

So the runtime should be state-conditioned:

1. Cheap global frame router.
2. Cheap baseline proposal path.
3. Candidate-local router.
4. Specialized proposal/ranker only where the router says it is useful.

## Two-Level Router

### Frame Router

Runs before expensive proposal generation.

Purpose:

- decide whether the frame is mostly clean sky, skyline/boundary, surface-heavy,
  glare/sun, or unknown;
- choose the maximum candidate budget;
- choose whether temporal-stack/surface mode is allowed.

Inputs should be cheap:

- grayscale downsample;
- brightness histogram;
- edge/texture density;
- horizon/boundary estimate;
- sun/glare saturation score;
- motion-registration confidence.

### Candidate-Local Router

Runs after cheap proposals exist.

Purpose:

- decide whether each candidate is `clean_sky`, `sky_target_near_surface`,
  `boundary_mixed`, `surface_backed`, `line_attached`, or `unknown`;
- allow the learned surface ranker only for `surface_backed`;
- prevent surface-mode tricks from damaging sky/skyline cases.

This is the important nuance: live code does not know the target location
before proposing candidates. The router must work both at frame level and at
candidate level.

## Runtime Modes

### Mode 0: Clean Sky

Goal: fast, stable, low clutter.

Enabled:

- baseline motion residual proposals;
- compact dark/map/native candidates;
- strict top-K;
- normal tube beam.

Disabled:

- broad scenario-balanced export;
- coast proposals except for already-locked tracks;
- learned surface ranker.

### Mode 1: Skyline / Boundary

Goal: avoid false silhouettes and horizon clutter.

Enabled:

- baseline proposals;
- large-dark only with conservative limits;
- temporal continuity and soft kinematics;
- null/clutter margin.

Disabled by default:

- learned surface ranker;
- aggressive coast proposals.

### Mode 2: Surface-Backed

Goal: recover target against trees, grass, terrain, road, and ridge texture.

Enabled:

- scenario-balanced candidate export;
- temporal-stack proposals;
- local/background-aware features;
- learned surface ranker;
- stricter clutter/null threshold.

Coast proposals:

- allowed only from strong existing tracks;
- low prior score;
- must be supported by real local evidence before selection.

### Mode 3: Close / Large Target

Goal: recover close visible silhouettes.

Enabled:

- `large_dark` proposal path;
- native-resolution ROI scoring;
- larger box bank.

Disabled:

- broad surface ranker unless candidate-local router says surface-backed.

### Mode 4: Unknown / Low Confidence

Goal: conservative behavior.

Enabled:

- baseline proposal/ranker only;
- top-tube export for debugging.

Disabled:

- high-score coast;
- heavy surface mode.

## Pi 5 Budget

For 30 Hz, one frame has 33.3 ms. The realistic budget should be lower because
guidance/control and I/O also need time.

Initial budget target:

| Block | Target |
| --- | ---: |
| capture + grayscale + resize | 2-4 ms |
| global motion model | 4-7 ms |
| frame router | <1 ms |
| baseline proposals | 4-8 ms |
| candidate-local router | 1-3 ms |
| TBD beam update | 2-5 ms |
| selected-box output | <1 ms |
| optional surface extras | only if total stays under budget |

Hard rule: a mode is not a candidate for onboard default until measured on Pi 5
or an equivalent ARM profile. Mac runtime is useful for iteration but not for
deployment claims.

## Rust Position

Do not rewrite the whole detector in Rust yet.

Correct sequence:

1. Stabilize the Python/OpenCV state machine and prove which branches help.
2. Freeze input/output contracts:
   - frame in;
   - candidate list;
   - tube state;
   - selected box;
   - diagnostics.
3. Profile on Pi 5.
4. Port only stable hot paths first.

Likely Rust targets:

- candidate scoring loops;
- non-maximum suppression;
- small box/ring statistics;
- track/tube state update;
- router feature extraction;
- static learned-model inference if it becomes fixed.

Keep in Python/OpenCV until stable:

- experiment orchestration;
- labeling/export scripts;
- model training;
- new feature prototyping.

Rust is useful for deployment, but rewriting before the state machine is stable
would slow the algorithm work and make mistakes harder to inspect.

## Immediate Engineering Path

1. Make the current router candidate-local, not label-local.
2. Add an offline state-machine selector evaluator:
   - baseline for sky/boundary/unknown;
   - learned ranker only for surface-backed candidates;
   - report strict/loose/oracle by state.
3. Generate more true surface labels, especially tree/grass/road-backed target
   frames, not merely skyline/cloud-backed frames.
4. Once state-conditioned selection consistently beats baseline, integrate it
   into `tbd_motion_detector.py` behind explicit flags.
5. Run a Pi 5 benchmark before any default change.

## Implementation Checkpoint - 2026-05-29

Implemented in `scripts/tbd_motion_detector.py`:

- `--runtime_mode baseline|clean_sky|boundary|surface|auto`
- `--candidate_router off|log|apply`
- cheap frame router with per-frame mode logging;
- candidate-local router states:
  - `clean_sky`
  - `sky_target_near_surface`
  - `boundary_mixed`
  - `surface_backed`
  - `line_attached`
  - `unknown`
- optional router application that penalizes surface-only proposal sources
  outside `surface_backed` candidates;
- optional `--surface_ranker_scope surface_backed` so the tube
  verifier/ranker can be constrained to surface-backed tubes;
- per-frame timing breakdown in `report.json` and `timing_summary.csv`;
- `scripts/benchmark_runtime_modes.py` for repeatable runtime-mode benchmarks.

Current status:

- default behavior remains `--runtime_mode baseline --candidate_router off`;
- `auto` is still a logging/experiment path, not a default control path;
- candidate-local routing is cheap enough for continued testing after the
  integral-image rewrite;
- Python beam hot-path passes are complete for the current prototype: state
  warp/prediction references are precomputed once per frame, diagnostic
  pair/background/alignment features are skipped unless needed, and losing
  candidate-transition states are not materialized;
- normal pair-rescue mode now averages near Mac-side 30 Hz on d129/e271 slices
  and is borderline on aaf1, but p90 timing still misses 30 Hz on the harder
  clips and Pi 5 capture/decode/p99/thermal behavior is untested;
- explicit surface mode is still too slow for live Python deployment.

Benchmark artifacts:

- `artifacts/runtime_mode_benchmark_v1/`
- `artifacts/runtime_mode_benchmark_surface_extras_v1/`

## Default Policy

Current default should remain the conservative baseline/pair-rescue path.

Experimental onboard modes should be explicit:

- `--runtime_mode clean_sky`
- `--runtime_mode boundary`
- `--runtime_mode surface`
- `--runtime_mode auto`

`auto` should initially log decisions without changing behavior. Only after it
beats baseline in evaluation should it control proposal/ranker selection.
