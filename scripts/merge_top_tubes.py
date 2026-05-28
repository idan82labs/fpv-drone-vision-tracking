#!/usr/bin/env python3
"""Merge exported top_tubes.csv files into one candidate pool.

This is for proposal-source experiments: keep the same tube rows, add numeric
variant flags, dedupe near-identical boxes per frame, then re-rank by the
existing verified score so train_xy_tube_ranker can audit the union.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True, help="NAME=/path/to/top_tubes.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--nms_px", type=float, default=2.5)
    p.add_argument("--max_per_frame", type=int, default=320)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def center(row: dict[str, str]) -> tuple[float, float]:
    return (
        safe_float(row.get("x")) + 0.5 * safe_float(row.get("w"), 1.0),
        safe_float(row.get("y")) + 0.5 * safe_float(row.get("h"), 1.0),
    )


def main() -> None:
    args = parse_args()
    variant_names: list[str] = []
    rows: list[dict[str, str]] = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"--input must be NAME=PATH, got {spec!r}")
        name, path_text = spec.split("=", 1)
        name = name.strip()
        variant_names.append(name)
        for row in read_csv(Path(path_text)):
            rec = dict(row)
            rec["proposal_variant"] = name
            for variant in variant_names:
                rec[f"variant_{variant}"] = "1" if variant == name else "0"
            rows.append(rec)

    # Backfill flags for rows read before later variant names were known.
    for rec in rows:
        for variant in variant_names:
            rec.setdefault(f"variant_{variant}", "1" if rec["proposal_variant"] == variant else "0")

    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        frame = int(safe_float(row.get("frame")))
        by_frame.setdefault(frame, []).append(row)

    merged: list[dict[str, str]] = []
    for frame, frame_rows in sorted(by_frame.items()):
        kept: list[dict[str, str]] = []
        for row in sorted(frame_rows, key=lambda r: safe_float(r.get("verified_score"), -999.0), reverse=True):
            cx, cy = center(row)
            duplicate = False
            for other in kept:
                ox, oy = center(other)
                if math.hypot(cx - ox, cy - oy) <= args.nms_px:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(row)
            if len(kept) >= args.max_per_frame:
                break
        for rank, row in enumerate(kept, start=1):
            rec = dict(row)
            rec["rank"] = str(rank)
            merged.append(rec)

    fields: list[str] = []
    for row in merged:
        for key in row:
            if key not in fields:
                fields.append(key)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    print(out)


if __name__ == "__main__":
    main()
