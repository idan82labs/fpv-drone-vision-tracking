# Runtime Mode Benchmark

This harness compares detector runtime modes on the current machine. Use it for branch overhead and candidate-pressure comparisons only; Pi 5 deployment still needs an ARM run.

- Profile: `pair_rescue`
- Max frames per run: `700`
- Summary CSV: `runtime_mode_benchmark.csv`

e271 full-clip timing on this Mac:

- `baseline`: 15.70 ms/frame average, 24.55 ms p90.
- `auto_apply`: 16.97 ms/frame average, 26.27 ms p90.
- `surface`: 17.14 ms/frame average, 26.81 ms p90.

Interpretation: the router/surface branch overhead is modest on this machine
for pair-rescue mode, but this is still not a Raspberry Pi 5 measurement.
The detector fits 30 Hz here and does not safely fit 60 Hz here.
