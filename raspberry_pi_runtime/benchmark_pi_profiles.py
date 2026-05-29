#!/usr/bin/env python3
"""Benchmark Pi runtime profiles and optionally evaluate labels.

This is still a Mac-side proxy. The ``pi_estimate_*`` columns use a configurable
slowdown multiplier plus I/O reserve so we do not mistake Mac timing for Pi 5
deployment proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SELECTED_EVAL_FIELDS = ["clip", "frame", "selected", "rank", "x", "y", "w", "h", "learned_score", "track_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", action="append", required=True)
    p.add_argument("--labels", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--profile", action="append", default=[])
    p.add_argument("--deferred_sequence", action="store_true")
    p.add_argument("--live_sequence", action="store_true")
    p.add_argument("--sequence_window", default="")
    p.add_argument("--max_frames", default="")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--pi_slowdown", type=float, default=2.4)
    p.add_argument("--io_reserve_ms", type=float, default=3.0)
    p.add_argument(
        "--production_gate",
        action="store_true",
        help="Run each profile in stream-only telemetry mode and append production-gate results.",
    )
    p.add_argument("--gate_budget_ms", type=float, default=33.3)
    p.add_argument("--gate_max_wall_ms", type=float, default=100.0)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text())["summary"]


def first_summary(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return len(read_csv(path))


def selected_jsonl_to_csv(jsonl_path: Path, out_path: Path, clip: str) -> Path:
    """Convert selected-box JSONL telemetry to the CSV schema used by evaluators."""
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                bbox = rec.get("bbox") or []
                if len(bbox) < 4:
                    continue
                rows.append(
                    {
                        "clip": clip,
                        "frame": rec.get("frame", ""),
                        "selected": 1,
                        "rank": rec.get("rank", 1),
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2],
                        "h": bbox[3],
                        "learned_score": rec.get("verified_score", rec.get("score", "")),
                        "track_id": rec.get("track_id", ""),
                    }
                )
    if rows:
        write_csv(out_path, rows)
    else:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SELECTED_EVAL_FIELDS)
            writer.writeheader()
    return out_path


def load_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parent / "run_pi_detector.py"
    gate = Path(__file__).resolve().parent / "production_gate.py"
    evaluator = repo / "scripts" / "evaluate_tracking_run.py"
    profiles = args.profile or ["pi_light_live", "pi_balanced_live", "pi_quality_live"]
    if args.production_gate and args.deferred_sequence:
        raise SystemExit("--production_gate is for live/non-deferred runs; deferred replay requires top_tubes.csv.")
    rows: list[dict[str, Any]] = []
    for video_str in args.video:
        video = Path(video_str)
        clip = video.stem
        for profile in profiles:
            suffix = "_deferred" if args.deferred_sequence else ""
            suffix += "_live_sequence" if args.live_sequence else ""
            run_name = f"{clip}_{profile}{suffix}"
            run_dir = out_dir / run_name
            cmd = [
                args.python,
                str(runner),
                str(video),
                "--output_dir",
                str(run_dir),
                "--profile",
                profile,
                "--clip",
                clip,
                "--repo_root",
                str(repo),
            ]
            if args.max_frames:
                cmd += ["--max_frames", args.max_frames]
            if args.deferred_sequence:
                cmd.append("--deferred_sequence")
            if args.live_sequence:
                cmd.append("--live_sequence")
            if args.sequence_window:
                cmd += ["--sequence_window", args.sequence_window]
            selected_jsonl = run_dir / "selected.jsonl"
            telemetry_jsonl = run_dir / "telemetry.jsonl"
            if args.production_gate:
                cmd += [
                    "--stream_only",
                    "--selected_jsonl",
                    str(selected_jsonl),
                    "--telemetry_jsonl",
                    str(telemetry_jsonl),
                ]
            run(cmd)
            summary = load_report(run_dir)
            selected = run_dir / ("sequence_selected_tracks.csv" if args.deferred_sequence else "selected_tracks.csv")
            if args.production_gate:
                selected = selected_jsonl_to_csv(selected_jsonl, run_dir / "selected_from_jsonl.csv", clip)
                run(
                    [
                        args.python,
                        str(gate),
                        "--run_dir",
                        str(run_dir),
                        "--telemetry_jsonl",
                        str(telemetry_jsonl),
                        "--budget_ms",
                        str(args.gate_budget_ms),
                        "--max_wall_ms",
                        str(args.gate_max_wall_ms),
                        "--allow_fail",
                    ]
                )
            gate_result = load_gate(run_dir / "production_gate.json") if args.production_gate else {}
            eval_summary: dict[str, str] = {}
            if args.labels:
                eval_dir = run_dir / "eval"
                run(
                    [
                        args.python,
                        str(evaluator),
                        "--labels",
                        args.labels,
                        "--clip",
                        clip,
                        "--selected",
                        str(selected),
                        "--report",
                        str(run_dir / "report.json"),
                        "--out_dir",
                        str(eval_dir),
                        "--strict_tol_px",
                        "8",
                        "--loose_tol_px",
                        "16",
                    ]
                )
                eval_summary = first_summary(eval_dir / "summary.csv")
            avg_ms = float(summary.get("avg_ms_per_frame", 0.0) or 0.0)
            p90_ms = float(summary.get("p90_ms_per_frame", 0.0) or 0.0)
            p95_ms = float(summary.get("p95_ms_per_frame", 0.0) or 0.0)
            p99_ms = float(summary.get("p99_ms_per_frame", 0.0) or 0.0)
            max_ms = float(summary.get("max_ms_per_frame", 0.0) or 0.0)
            p95_wall_ms = float(summary.get("p95_wall_ms_per_frame", p95_ms) or 0.0)
            p99_wall_ms = float(summary.get("p99_wall_ms_per_frame", p99_ms) or 0.0)
            n_processed = float(summary.get("n_processed", 0.0) or 0.0)
            detector_selected_rate = summary.get("selected_frame_rate", "")
            if args.deferred_sequence:
                output_selected_rate = round(csv_row_count(selected) / n_processed, 3) if n_processed else ""
            elif args.live_sequence:
                output_selected_rate = summary.get("selected_output_frame_rate", "")
            else:
                output_selected_rate = detector_selected_rate
            row: dict[str, Any] = {
                "clip": clip,
                "profile": profile,
                "deferred_sequence": int(args.deferred_sequence),
                "live_sequence": int(args.live_sequence),
                "mac_avg_ms": round(avg_ms, 3),
                "mac_p90_ms": round(p90_ms, 3),
                "mac_p95_ms": round(p95_ms, 3),
                "mac_p99_ms": round(p99_ms, 3),
                "mac_max_ms": round(max_ms, 3),
                "mac_p95_wall_ms": round(p95_wall_ms, 3),
                "mac_p99_wall_ms": round(p99_wall_ms, 3),
                "pi_estimate_avg_ms": round(avg_ms * args.pi_slowdown + args.io_reserve_ms, 3),
                "pi_estimate_p90_ms": round(p90_ms * args.pi_slowdown + args.io_reserve_ms, 3),
                "pi_estimate_p95_wall_ms": round(p95_wall_ms * args.pi_slowdown + args.io_reserve_ms, 3),
                "pi_estimate_p99_wall_ms": round(p99_wall_ms * args.pi_slowdown + args.io_reserve_ms, 3),
                "pi_estimate_fits_30hz_avg": avg_ms * args.pi_slowdown + args.io_reserve_ms <= 33.3,
                "pi_estimate_fits_30hz_p90": p90_ms * args.pi_slowdown + args.io_reserve_ms <= 33.3,
                "pi_estimate_fits_30hz_p95_wall": p95_wall_ms * args.pi_slowdown + args.io_reserve_ms <= 33.3,
                "pi_estimate_fits_30hz_p99_wall": p99_wall_ms * args.pi_slowdown + args.io_reserve_ms <= 33.3,
                "avg_candidates": summary.get("avg_candidates_per_frame", ""),
                "p90_candidates": summary.get("p90_candidates_per_frame", ""),
                "selected_rate": output_selected_rate,
                "detector_selected_rate": detector_selected_rate,
                "output_selected_rate": output_selected_rate,
                "selected_output_rows": summary.get("selected_output_rows", ""),
            }
            if args.production_gate:
                gate_checks = gate_result.get("checks", [])
                failed_checks = [str(check.get("name", "")) for check in gate_checks if not check.get("passed")]
                gate_summary = gate_result.get("summary", {})
                telemetry_summary = gate_result.get("telemetry_summary", {})
                row.update(
                    {
                        "production_gate_passed": bool(gate_result.get("passed")),
                        "production_gate_failed_checks": ";".join(failed_checks),
                        "gate_p99_wall_ms": gate_summary.get("p99_wall_ms_per_frame", ""),
                        "gate_max_wall_ms": gate_summary.get("max_wall_ms_per_frame", ""),
                        "telemetry_records": telemetry_summary.get("records", ""),
                        "telemetry_selected_rate": telemetry_summary.get("selected_rate", ""),
                    }
                )
            for key in ("visible_frames", "strict_recall", "loose_recall", "invisible_no_box_rate", "selected_source_frame_rate"):
                row[key] = eval_summary.get(key, "")
            rows.append(row)
    write_csv(out_dir / "pi_profile_benchmark.csv", rows)
    print(out_dir / "pi_profile_benchmark.csv")


if __name__ == "__main__":
    main()
