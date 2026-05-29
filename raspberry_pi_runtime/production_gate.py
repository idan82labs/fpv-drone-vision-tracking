#!/usr/bin/env python3
"""Validate a Raspberry Pi runtime run directory.

This gate is deliberately conservative. It checks the deployment-facing
properties that matter before a run can be treated as live-control evidence:
bounded reporting, per-frame telemetry, latency tails, and stream-only output.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--report", default="", help="Defaults to <run_dir>/report.json")
    p.add_argument("--telemetry_jsonl", default="", help="Defaults to <run_dir>/telemetry.jsonl if present")
    p.add_argument("--out", default="", help="Defaults to <run_dir>/production_gate.json")
    p.add_argument("--budget_ms", type=float, default=33.3)
    p.add_argument("--max_wall_ms", type=float, default=100.0)
    p.add_argument("--min_frames", type=int, default=30)
    p.add_argument("--max_report_frames", type=int, default=0)
    p.add_argument("--require_stream_only", action="store_true", default=True)
    p.add_argument("--no_require_stream_only", action="store_false", dest="require_stream_only")
    p.add_argument("--require_telemetry", action="store_true", default=True)
    p.add_argument("--no_require_telemetry", action="store_false", dest="require_telemetry")
    p.add_argument("--allow_fail", action="store_true", help="Write output and exit 0 even when checks fail.")
    return p.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo))


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "summary" not in data:
        raise ValueError(f"{path} has no summary object")
    return data


def load_telemetry(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
    return rows


def telemetry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wall = [float(row.get("wall_ms", 0.0) or 0.0) for row in rows]
    process = [float(row.get("process_ms", 0.0) or 0.0) for row in rows]
    selected = sum(1 for row in rows if row.get("selected"))
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "records": len(rows),
        "selected_records": selected,
        "selected_rate": round(selected / len(rows), 4) if rows else 0.0,
        "wall_avg_ms": round(statistics.fmean(wall), 3) if wall else 0.0,
        "wall_p95_ms": round(percentile(wall, 95), 3),
        "wall_p99_ms": round(percentile(wall, 99), 3),
        "wall_max_ms": round(max(wall), 3) if wall else 0.0,
        "process_p99_ms": round(percentile(process, 99), 3),
        "statuses": statuses,
    }


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    report_path = Path(args.report) if args.report else run_dir / "report.json"
    telemetry_path = Path(args.telemetry_jsonl) if args.telemetry_jsonl else run_dir / "telemetry.jsonl"
    report = load_report(report_path)
    summary = report["summary"]
    n_processed = int(summary.get("n_processed", 0) or 0)
    checks: list[dict[str, Any]] = []

    add_check(checks, "processed_frames", n_processed >= args.min_frames, f"{n_processed} >= {args.min_frames}")
    add_check(
        checks,
        "report_mode_summary",
        report.get("report_mode") == "summary",
        f"report_mode={report.get('report_mode')!r}",
    )
    add_check(
        checks,
        "bounded_report_frames",
        int(summary.get("report_frames_stored", 0) or 0) <= args.max_report_frames,
        f"report_frames_stored={summary.get('report_frames_stored')}",
    )
    if args.require_stream_only:
        add_check(checks, "stream_only", bool(report.get("stream_only")), f"stream_only={report.get('stream_only')}")

    p95_wall = float(summary.get("p95_wall_ms_per_frame", 0.0) or 0.0)
    p99_wall = float(summary.get("p99_wall_ms_per_frame", p95_wall) or 0.0)
    max_wall = float(summary.get("max_wall_ms_per_frame", p99_wall) or 0.0)
    add_check(checks, "p95_wall_budget", p95_wall <= args.budget_ms, f"{p95_wall:.3f} <= {args.budget_ms:.3f} ms")
    add_check(checks, "p99_wall_budget", p99_wall <= args.budget_ms, f"{p99_wall:.3f} <= {args.budget_ms:.3f} ms")
    add_check(checks, "max_wall_budget", max_wall <= args.max_wall_ms, f"{max_wall:.3f} <= {args.max_wall_ms:.3f} ms")

    telemetry: list[dict[str, Any]] = []
    telemetry_stats: dict[str, Any] = {}
    if telemetry_path.exists():
        telemetry = load_telemetry(telemetry_path)
        telemetry_stats = telemetry_summary(telemetry)
        add_check(checks, "telemetry_present", True, str(telemetry_path))
        add_check(
            checks,
            "telemetry_frame_coverage",
            len(telemetry) >= n_processed,
            f"{len(telemetry)} telemetry records >= {n_processed} processed frames",
        )
        add_check(
            checks,
            "telemetry_p99_wall_budget",
            float(telemetry_stats["wall_p99_ms"]) <= args.budget_ms,
            f"{telemetry_stats['wall_p99_ms']:.3f} <= {args.budget_ms:.3f} ms",
        )
    elif args.require_telemetry:
        add_check(checks, "telemetry_present", False, f"{telemetry_path} not found")

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "run_dir": str(run_dir),
        "report": str(report_path),
        "telemetry_jsonl": str(telemetry_path) if telemetry_path.exists() else None,
        "budget_ms": args.budget_ms,
        "max_wall_ms": args.max_wall_ms,
        "summary": {
            "n_processed": n_processed,
            "selected_output_frame_rate": summary.get("selected_output_frame_rate"),
            "p95_wall_ms_per_frame": p95_wall,
            "p99_wall_ms_per_frame": p99_wall,
            "max_wall_ms_per_frame": max_wall,
        },
        "telemetry_summary": telemetry_stats,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    out = Path(args.out) if args.out else Path(args.run_dir) / "production_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(("PASS" if result["passed"] else "FAIL") + f" -> {out}")
    for check in result["checks"]:
        marker = "ok" if check["passed"] else "fail"
        print(f"- {marker}: {check['name']} ({check['detail']})")
    if not result["passed"] and not args.allow_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
