#!/usr/bin/env python3
"""Run the Raspberry Pi oriented detector profiles.

The profiles are deliberately conservative: they bound candidate count and beam
width, avoid learned-ranker inference in the live loop, and optionally run a
lightweight verified-score sequence selector for deferred/offline validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


RENDER_SELECTION_FIELDS = [
    "clip",
    "frame",
    "selected",
    "rank",
    "x",
    "y",
    "w",
    "h",
    "learned_score",
    "track_id",
]

PROFILES: dict[str, list[str]] = {
    "pi_light_live": [
        "--save_every", "0",
        "--downscale", "0.5",
        "--max_corners", "550",
        "--quality", "0.01",
        "--min_distance", "8",
        "--beam_width", "32",
        "--top_k_candidates", "32",
        "--map_peaks",
        "--map_radii", "2,3",
        "--map_top_k", "45",
        "--tube_verifier", "heuristic",
        "--selected_score", "6.0",
        "--birth_penalty", "1.8",
        "--miss_penalty", "1.1",
        "--pair_weight", "0.85",
    ],
    "pi_balanced_live": [
        "--save_every", "0",
        "--downscale", "0.5",
        "--max_corners", "650",
        "--quality", "0.009",
        "--min_distance", "8",
        "--beam_width", "44",
        "--top_k_candidates", "44",
        "--map_peaks",
        "--map_radii", "2,3,5",
        "--map_top_k", "65",
        "--large_dark_peaks",
        "--large_dark_top_k", "10",
        "--large_dark_score_floor", "38",
        "--tube_verifier", "heuristic",
        "--selected_score", "6.0",
        "--birth_penalty", "1.8",
        "--miss_penalty", "1.1",
        "--pair_weight", "0.85",
    ],
    "pi_quality_live": [
        "--save_every", "0",
        "--downscale", "0.5",
        "--max_corners", "750",
        "--quality", "0.008",
        "--min_distance", "7",
        "--beam_width", "56",
        "--top_k_candidates", "56",
        "--map_peaks",
        "--map_radii", "2,3,5",
        "--map_top_k", "85",
        "--large_dark_peaks",
        "--large_dark_top_k", "16",
        "--large_dark_score_floor", "36",
        "--tube_verifier", "heuristic",
        "--selected_score", "6.0",
        "--birth_penalty", "1.8",
        "--miss_penalty", "1.1",
        "--pair_weight", "0.9",
    ],
}

SEQUENCE_DEFAULTS = {
    "max_rank": "20",
    "max_jump_px": "10",
    "transition_weight": "1.5",
    "window": "60",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--profile", choices=sorted(PROFILES), default="pi_light_live")
    p.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--max_frames", default="")
    p.add_argument(
        "--report_mode",
        choices=("summary", "full"),
        default="summary",
        help="Use summary for onboard bounded memory; full preserves per-frame diagnostics for lab analysis.",
    )
    p.add_argument("--stream_only", action="store_true", help="Do not retain CSV rows; use JSONL streams only.")
    sequence_group = p.add_mutually_exclusive_group()
    sequence_group.add_argument(
        "--deferred_sequence",
        action="store_true",
        help="Export top states and run verified-score Viterbi after the detector pass.",
    )
    sequence_group.add_argument(
        "--live_sequence",
        action="store_true",
        help="Use the detector's bounded delayed sequence selector.",
    )
    p.add_argument(
        "--selected_jsonl",
        default="",
        help="Optional selected-box telemetry stream. Defaults to no stream; use /run/fpv-tracker/selected.jsonl on Pi.",
    )
    p.add_argument(
        "--telemetry_jsonl",
        default="",
        help="Optional per-frame status telemetry stream. Use /run/fpv-tracker/telemetry.jsonl on Pi.",
    )
    p.add_argument("--sequence_window", default=SEQUENCE_DEFAULTS["window"])
    p.add_argument("--sequence_max_rank", default=SEQUENCE_DEFAULTS["max_rank"])
    p.add_argument("--sequence_max_jump_px", default=SEQUENCE_DEFAULTS["max_jump_px"])
    p.add_argument("--sequence_transition_weight", default=SEQUENCE_DEFAULTS["transition_weight"])
    p.add_argument("--clip", default="", help="Clip id for deferred selected-track CSV; defaults to video stem.")
    p.add_argument("--render_demo", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_selection_csv(selected_path: Path, out_path: Path, clip: str) -> Path:
    rows = read_csv(selected_path)
    if not rows:
        write_csv(out_path, [], RENDER_SELECTION_FIELDS)
        return out_path
    if {"clip", "selected", "rank", "learned_score"}.issubset(rows[0].keys()):
        return selected_path
    converted: list[dict[str, object]] = []
    for row in rows:
        if not row.get("frame") or not row.get("x") or not row.get("y"):
            continue
        score = row.get("verified_score") or row.get("learned_score") or row.get("score") or "0"
        converted.append(
            {
                "clip": clip,
                "frame": row["frame"],
                "selected": 1,
                "rank": row.get("rank") or 1,
                "x": row.get("x", ""),
                "y": row.get("y", ""),
                "w": row.get("w", 1),
                "h": row.get("h", 1),
                "learned_score": score,
                "track_id": row.get("track_id", ""),
            }
        )
    write_csv(out_path, converted, RENDER_SELECTION_FIELDS)
    return out_path


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = repo / "scripts" / "tbd_motion_detector.py"
    selector = Path(__file__).resolve().parent / "verified_sequence_selector.py"
    clip = args.clip or Path(args.video).stem

    detector_args = list(PROFILES[args.profile])
    detector_args += ["--report_mode", args.report_mode]
    if args.stream_only:
        detector_args.append("--stream_only")
    sequence_defaults = {
        "max_rank": str(args.sequence_max_rank),
        "max_jump_px": str(args.sequence_max_jump_px),
        "transition_weight": str(args.sequence_transition_weight),
        "window": str(args.sequence_window),
    }
    if args.live_sequence:
        detector_args += [
            "--delayed_sequence_select",
            "--delayed_sequence_top_n", sequence_defaults["max_rank"],
            "--delayed_sequence_min_hits", "1",
            "--delayed_sequence_max_jump_px", sequence_defaults["max_jump_px"],
            "--delayed_sequence_transition_weight", sequence_defaults["transition_weight"],
            "--delayed_sequence_window", sequence_defaults["window"],
        ]
    if args.deferred_sequence:
        detector_args += ["--export_top_tubes", sequence_defaults["max_rank"]]
    if args.max_frames:
        detector_args += ["--max_frames", args.max_frames]
    if args.selected_jsonl:
        detector_args += ["--selected_jsonl", args.selected_jsonl]
    if args.telemetry_jsonl:
        detector_args += ["--telemetry_jsonl", args.telemetry_jsonl]
    detector_args += args.extra

    run([args.python, str(detector), args.video, "--output_dir", str(out_dir), *detector_args])

    selected_path = out_dir / "selected_tracks.csv"
    if args.deferred_sequence:
        selected_path = out_dir / "sequence_selected_tracks.csv"
        run(
            [
                args.python,
                str(selector),
                "--top_tubes",
                str(out_dir / "top_tubes.csv"),
                "--clip",
                clip,
                "--out_csv",
                str(selected_path),
                "--max_rank",
                sequence_defaults["max_rank"],
                "--max_jump_px",
                sequence_defaults["max_jump_px"],
                "--transition_weight",
                sequence_defaults["transition_weight"],
            ]
        )

    if args.render_demo:
        renderer = repo / "scripts" / "render_tracking_demo_zoom.py"
        render_selected_path = render_selection_csv(selected_path, out_dir / "render_selected_tracks.csv", clip)
        run(
            [
                args.python,
                str(renderer),
                "--video",
                args.video,
                "--selections",
                str(render_selected_path),
                "--clip",
                clip,
                "--out",
                str(out_dir / "pi_tracking_demo.mp4"),
                "--title",
                f"{args.profile} {'deferred' if args.deferred_sequence else 'live'}",
            ]
        )

    (out_dir / "pi_profile_summary.json").write_text(
        json.dumps(
            {
                "video": args.video,
                "clip": clip,
                "profile": args.profile,
                "deferred_sequence": args.deferred_sequence,
                "live_sequence": args.live_sequence,
                "selected_tracks": str(selected_path),
                "profile_args": detector_args,
                "sequence_defaults": sequence_defaults if (args.deferred_sequence or args.live_sequence) else None,
                "selected_jsonl": args.selected_jsonl or None,
                "telemetry_jsonl": args.telemetry_jsonl or None,
                "report_mode": args.report_mode,
                "stream_only": args.stream_only,
            },
            indent=2,
        )
    )
    print(selected_path)


if __name__ == "__main__":
    main()
