#!/usr/bin/env python3
"""Export selected tube features from TBD report JSON files.

This backfills the same training table that tbd_motion_detector.py now writes
at runtime. It is intentionally selected-tube only; hard-negative mining from
non-selected beam states needs a richer report format.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("result_dir")
    p.add_argument("--output", default=None)
    return p.parse_args()


def flatten_selected(clip: str, rec: dict) -> dict | None:
    selected = rec.get("selected")
    if not selected:
        return None
    bbox = selected.get("bbox") or ["", "", "", ""]
    row = {
        "clip": clip,
        "frame": rec.get("frame", ""),
        "track_id": selected.get("track_id", ""),
        "x": bbox[0],
        "y": bbox[1],
        "w": bbox[2],
        "h": bbox[3],
        "score": selected.get("score", ""),
        "verified_score": selected.get("verified_score", ""),
        "tube_verifier_score": selected.get("tube_verifier_score", ""),
        "misses": selected.get("misses", ""),
        "vx": selected.get("vx", ""),
        "vy": selected.get("vy", ""),
    }
    cand = selected.get("candidate") or {}
    for key, value in cand.items():
        if isinstance(value, (int, float, str)):
            row[f"cand_{key}"] = value
    for key, value in (selected.get("tube_features") or {}).items():
        if isinstance(value, (int, float)):
            row[f"tube_{key}"] = value
    return row


def main() -> None:
    args = parse_args()
    root = Path(args.result_dir)
    rows: list[dict] = []
    for report_path in sorted(root.glob("*/report.json")):
        data = json.loads(report_path.read_text())
        clip = report_path.parent.name
        for rec in data.get("frames", []):
            row = flatten_selected(clip, rec)
            if row is not None:
                rows.append(row)
    out = Path(args.output) if args.output else root / "selected_tube_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
