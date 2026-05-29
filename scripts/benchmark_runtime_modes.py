#!/usr/bin/env python3
"""Run tbd_motion_detector.py across runtime modes and summarize timing.

This is an offline harness for the embedded runtime plan. It measures mode
overhead and branch timing; it does not claim Raspberry Pi 5 performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MODE_ARGS: dict[str, list[str]] = {
    "baseline": ["--runtime_mode", "baseline", "--candidate_router", "off"],
    "auto_log": ["--runtime_mode", "auto", "--candidate_router", "log"],
    "auto_apply": ["--runtime_mode", "auto", "--candidate_router", "apply"],
    "clean_sky": ["--runtime_mode", "clean_sky", "--candidate_router", "apply"],
    "boundary": ["--runtime_mode", "boundary", "--candidate_router", "apply"],
    "surface": ["--runtime_mode", "surface", "--candidate_router", "apply"],
}


PAIR_RESCUE_ARGS = [
    "--downscale",
    "0.5",
    "--kinematic_gate",
    "--beam_width",
    "100",
    "--top_k_candidates",
    "90",
    "--map_peaks",
    "--map_radii",
    "1,2,3,5",
    "--selected_score",
    "18",
    "--birth_penalty",
    "2.2",
    "--miss_penalty",
    "1.35",
    "--pair_weight",
    "1.05",
    "--line_weight",
    "0.75",
    "--support_penalty_weight",
    "3.0",
    "--app_low_residual_penalty",
    "0.3",
    "--tube_verifier",
    "heuristic",
    "--tube_verifier_floor",
    "-1.0",
    "--save_every",
    "0",
]


SURFACE_EXTRAS_ARGS = [
    "--scenario_balance",
    "--temporal_stack_peaks",
    "--temporal_stack_offsets",
    "-5,-3,-1",
    "--temporal_stack_top_k",
    "80",
    "--temporal_stack_halo_bases",
    "16",
    "--hybrid_coast_proposals",
    "--hybrid_coast_top_k",
    "12",
    "--surface_ranker_scope",
    "surface_backed",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", action="append", required=True, help="Video path. Repeat for multiple clips.")
    p.add_argument("--out_dir", default="results/runtime_mode_benchmark")
    p.add_argument("--max_frames", type=int, default=220)
    p.add_argument("--profile", choices=("minimal", "pair_rescue", "surface_extras"), default="pair_rescue")
    p.add_argument("--mode", action="append", choices=sorted(MODE_ARGS), help="Mode to run. Defaults to all modes.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--detector", default="scripts/tbd_motion_detector.py")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra detector args after --extra.")
    return p.parse_args()


def read_summary(report_path: Path) -> dict[str, Any]:
    data = json.loads(report_path.read_text())
    return data.get("summary", {})


def profile_args(profile: str) -> list[str]:
    if profile == "minimal":
        return ["--save_every", "0"]
    if profile == "surface_extras":
        return PAIR_RESCUE_ARGS + SURFACE_EXTRAS_ARGS
    return PAIR_RESCUE_ARGS


def clip_name(path: Path) -> str:
    return path.stem.split(".")[0]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = args.mode or list(MODE_ARGS)
    rows: list[dict[str, Any]] = []

    for video_str in args.video:
        video = Path(video_str)
        for mode in modes:
            run_dir = out_dir / clip_name(video) / mode
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                args.python,
                args.detector,
                str(video),
                "--output_dir",
                str(run_dir),
                "--max_frames",
                str(args.max_frames),
                *profile_args(args.profile),
                *MODE_ARGS[mode],
                *args.extra,
            ]
            subprocess.run(cmd, check=True)
            summary = read_summary(run_dir / "report.json")
            timing = summary.get("avg_timing_ms", {})
            p90_timing = summary.get("p90_timing_ms", {})
            p95_timing = summary.get("p95_timing_ms", {})
            row: dict[str, Any] = {
                "clip": clip_name(video),
                "mode": mode,
                "profile": args.profile,
                "frames": summary.get("n_processed", 0),
                "avg_ms": summary.get("avg_ms_per_frame", 0),
                "p90_ms": summary.get("p90_ms_per_frame", 0),
                "p95_ms": summary.get("p95_ms_per_frame", 0),
                "fits_30hz_mac": summary.get("fits_30hz", False),
                "fits_60hz_mac": summary.get("fits_60hz_on_this_machine", False),
                "avg_candidates": summary.get("avg_candidates_per_frame", 0),
                "p90_candidates": summary.get("p90_candidates_per_frame", 0),
                "selected_rate": summary.get("selected_frame_rate", 0),
                "runtime_mode_counts": json.dumps(summary.get("runtime_mode_counts", {}), sort_keys=True),
                "candidate_router_counts": json.dumps(summary.get("candidate_router_counts", {}), sort_keys=True),
            }
            for key, value in timing.items():
                row[f"timing_{key}_avg_ms"] = value
            for key, value in p90_timing.items():
                row[f"timing_{key}_p90_ms"] = value
            for key, value in p95_timing.items():
                row[f"timing_{key}_p95_ms"] = value
            rows.append(row)

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_dir / "runtime_mode_benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "README.md").write_text(
        "# Runtime Mode Benchmark\n\n"
        "This harness compares detector runtime modes on the current machine. "
        "Use it for branch overhead and candidate-pressure comparisons only; "
        "Pi 5 deployment still needs an ARM run.\n\n"
        f"- Profile: `{args.profile}`\n"
        f"- Max frames per run: `{args.max_frames}`\n"
        f"- Summary CSV: `runtime_mode_benchmark.csv`\n"
    )
    print(out_dir / "runtime_mode_benchmark.csv")


if __name__ == "__main__":
    main()
