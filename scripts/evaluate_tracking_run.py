#!/usr/bin/env python3
"""Evaluate detector selected tracks against frame-level labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--selected", required=True)
    p.add_argument("--report", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def fnum(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def visible(row: dict[str, str]) -> bool:
    raw = str(row.get("visible", "")).strip().lower()
    if raw in {"0", "false", "no", "empty", "none", "not_visible", "not visible"}:
        return False
    return True


def label_bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    x = fnum(row.get("det_x", row.get("x")))
    y = fnum(row.get("det_y", row.get("y")))
    w = fnum(row.get("det_w", row.get("w")), 1.0)
    h = fnum(row.get("det_h", row.get("h")), 1.0)
    if x is None or y is None or w is None or h is None:
        return None
    return x, y, w, h


def row_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row["x"]),
        float(row["y"]),
        float(row.get("w", 1.0) or 1.0),
        float(row.get("h", 1.0) or 1.0),
    )


def row_is_selected(row: dict[str, str]) -> bool:
    raw = str(row.get("selected", "1")).strip().lower()
    if raw in {"0", "false", "no", "none", "null", ""}:
        return False
    return all(str(row.get(key, "")).strip() != "" for key in ("x", "y"))


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return float(math.hypot(ax - bx, ay - by))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {
        int(float(r["frame"])): r
        for r in read_csv(Path(args.selected))
        if r.get("frame") and row_is_selected(r)
    }
    labels = read_csv(Path(args.labels))
    if args.clip:
        labels = [r for r in labels if r.get("clip", args.clip) == args.clip]

    rows: list[dict[str, Any]] = []
    visible_rows = 0
    strict_hits = 0
    loose_hits = 0
    invisible_rows = 0
    invisible_no_box = 0
    missing_visible: list[dict[str, Any]] = []
    for lab in labels:
        frame_val = fnum(lab.get("frame"))
        if frame_val is None:
            continue
        frame = int(frame_val)
        is_visible = visible(lab)
        sel = selected.get(frame)
        selected_box = row_bbox(sel) if sel is not None else None
        bbox = label_bbox(lab) if is_visible else None
        dist = None
        strict = False
        loose = False
        if bbox is not None and selected_box is not None:
            dist = center_dist(selected_box, bbox)
            strict = dist <= args.strict_tol_px
            loose = dist <= args.loose_tol_px
        if is_visible:
            visible_rows += 1
            strict_hits += int(strict)
            loose_hits += int(loose)
            if not strict:
                missing_visible.append(
                    {
                        "frame": frame,
                        "time_s": lab.get("time_s", ""),
                        "det_x": bbox[0] if bbox else "",
                        "det_y": bbox[1] if bbox else "",
                        "det_w": bbox[2] if bbox else "",
                        "det_h": bbox[3] if bbox else "",
                        "selected": int(sel is not None),
                        "selected_x": "" if selected_box is None else selected_box[0],
                        "selected_y": "" if selected_box is None else selected_box[1],
                        "selected_w": "" if selected_box is None else selected_box[2],
                        "selected_h": "" if selected_box is None else selected_box[3],
                        "dist_px": "" if dist is None else round(dist, 3),
                        "confidence": lab.get("confidence", ""),
                        "notes": lab.get("notes", ""),
                    }
                )
        else:
            invisible_rows += 1
            invisible_no_box += int(sel is None)
        rows.append(
            {
                "frame": frame,
                "visible": int(is_visible),
                "selected": int(sel is not None),
                "strict_hit": int(strict),
                "loose_hit": int(loose),
                "dist_px": "" if dist is None else round(dist, 3),
                "det_x": "" if bbox is None else bbox[0],
                "det_y": "" if bbox is None else bbox[1],
                "det_w": "" if bbox is None else bbox[2],
                "det_h": "" if bbox is None else bbox[3],
                "selected_x": "" if selected_box is None else selected_box[0],
                "selected_y": "" if selected_box is None else selected_box[1],
                "selected_w": "" if selected_box is None else selected_box[2],
                "selected_h": "" if selected_box is None else selected_box[3],
            }
        )

    report_summary: dict[str, Any] = {}
    if args.report:
        report_path = Path(args.report)
        if report_path.exists():
            report_summary = json.loads(report_path.read_text()).get("summary", {})

    source_frames = ""
    source_fps = ""
    if args.report:
        report_data = json.loads(Path(args.report).read_text())
        source_frames = report_data.get("source_frames", "")
        source_fps = report_data.get("source_fps", "")

    summary = {
        "label_frames": len(rows),
        "visible_frames": visible_rows,
        "strict_hits": strict_hits,
        "strict_recall": round(strict_hits / max(1, visible_rows), 4),
        "loose_hits": loose_hits,
        "loose_recall": round(loose_hits / max(1, visible_rows), 4),
        "visible_misses_strict": len(missing_visible),
        "invisible_frames": invisible_rows,
        "invisible_no_box": invisible_no_box,
        "invisible_no_box_rate": round(invisible_no_box / max(1, invisible_rows), 4),
        "selected_frames_total": len(selected),
        "selected_source_frame_rate": round(len(selected) / max(1, int(source_frames or len(selected))), 4),
        "detector_selected_rate": report_summary.get("selected_frame_rate", ""),
        "processed_frames": report_summary.get("n_processed", ""),
        "source_frames": source_frames,
        "source_fps": source_fps,
        "avg_ms_per_frame": report_summary.get("avg_ms_per_frame", ""),
        "p90_ms_per_frame": report_summary.get("p90_ms_per_frame", ""),
    }
    write_csv(out_dir / "frame_eval.csv", rows)
    write_csv(out_dir / "visible_strict_misses.csv", missing_visible)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(out_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
