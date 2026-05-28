#!/usr/bin/env python3
"""
Evaluate detector outputs against human-reviewed checkpoint labels.

The labels are intentionally interpreted conservatively:
- gold: selected box should land near the reviewed gold box.
- empty: no selected box should be emitted.
- clutter: the reviewed selected box was clutter; repeating that same box is bad,
  selecting elsewhere is reported as unknown unless the review says no object.
- miss: no target bbox is available, so selections are reported as unknown.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--result", action="append", required=True, help="name=/path/to/results_root")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--center_tol_px", type=float, default=14.0)
    p.add_argument("--iou_tol", type=float, default=0.10)
    return p.parse_args()


def bbox_from_text(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    try:
        vals = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(vals, list | tuple) or len(vals) != 4:
        return None
    return tuple(int(round(float(v))) for v in vals)  # type: ignore[return-value]


def center(b: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = b
    return x + 0.5 * w, y + 0.5 * h


def center_dist(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_labels(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def result_roots(items: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--result must be name=/path, got {item!r}")
        name, path = item.split("=", 1)
        roots[name] = Path(path)
    return roots


def load_frame(root: Path, clip: str, frame_no: int) -> dict[str, Any] | None:
    report_path = root / clip / "report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    for rec in report.get("frames", []):
        if int(rec.get("frame", -1)) == frame_no:
            return rec
    return None


def classify(
    label: str,
    notes: str,
    gold_or_clutter: tuple[int, int, int, int] | None,
    selected: tuple[int, int, int, int] | None,
    center_tol_px: float,
    iou_tol: float,
) -> str:
    label = label.strip().lower()
    notes_l = notes.lower()
    no_object_note = (
        "no object" in notes_l
        or "no isolated object" in notes_l
        or "no clear object" in notes_l
        or "no selected box and no clear object" in notes_l
    )
    if label == "gold":
        if selected is None:
            return "gold_miss"
        if gold_or_clutter is not None and (
            center_dist(gold_or_clutter, selected) <= center_tol_px
            or iou(gold_or_clutter, selected) >= iou_tol
        ):
            return "gold_hit"
        return "gold_wrong_box"
    if label == "empty":
        return "empty_tn" if selected is None else "empty_fp"
    if label == "clutter":
        if selected is None:
            return "clutter_suppressed"
        if gold_or_clutter is not None and (
            center_dist(gold_or_clutter, selected) <= center_tol_px
            or iou(gold_or_clutter, selected) >= iou_tol
        ):
            return "clutter_repeated"
        return "clutter_selected_elsewhere_fp" if no_object_note else "clutter_selected_elsewhere_unknown"
    if label == "miss":
        return "miss_no_select" if selected is None else "miss_selected_unknown"
    return "unknown_label"


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels))
    roots = result_roots(args.result)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for name, root in roots.items():
        counts: dict[str, int] = {}
        for row in labels:
            clip = row.get("clip", "")
            frame_no = int(row.get("frame", "0") or 0)
            label = row.get("label", "").strip().lower()
            reviewed_bbox = bbox_from_text(row.get("selected_bbox"))
            frame = load_frame(root, clip, frame_no)
            selected_json = frame.get("selected") if frame else None
            selected_bbox = tuple(selected_json["bbox"]) if selected_json else None
            outcome = classify(
                label,
                row.get("notes", ""),
                reviewed_bbox,
                selected_bbox,  # type: ignore[arg-type]
                args.center_tol_px,
                args.iou_tol,
            )
            counts[outcome] = counts.get(outcome, 0) + 1
            detail_rows.append(
                {
                    "variant": name,
                    "clip": clip,
                    "frame": frame_no,
                    "label": label,
                    "outcome": outcome,
                    "reviewed_bbox": reviewed_bbox,
                    "selected_bbox": selected_bbox,
                    "verified_score": selected_json.get("verified_score") if selected_json else "",
                    "score": selected_json.get("score") if selected_json else "",
                    "tube_verifier_score": selected_json.get("tube_verifier_score") if selected_json else "",
                    "notes": row.get("notes", ""),
                }
            )

        gold_total = sum(1 for r in labels if r.get("label", "").strip().lower() == "gold")
        clutter_total = sum(1 for r in labels if r.get("label", "").strip().lower() == "clutter")
        empty_total = sum(1 for r in labels if r.get("label", "").strip().lower() == "empty")
        known_fp = (
            counts.get("gold_wrong_box", 0)
            + counts.get("clutter_repeated", 0)
            + counts.get("clutter_selected_elsewhere_fp", 0)
            + counts.get("empty_fp", 0)
        )
        summary_rows.append(
            {
                "variant": name,
                "gold_total": gold_total,
                "gold_hit": counts.get("gold_hit", 0),
                "gold_miss": counts.get("gold_miss", 0),
                "gold_wrong_box": counts.get("gold_wrong_box", 0),
                "gold_recall": round(counts.get("gold_hit", 0) / max(1, gold_total), 3),
                "clutter_total": clutter_total,
                "clutter_suppressed": counts.get("clutter_suppressed", 0),
                "clutter_repeated": counts.get("clutter_repeated", 0),
                "clutter_selected_elsewhere_fp": counts.get("clutter_selected_elsewhere_fp", 0),
                "clutter_selected_elsewhere_unknown": counts.get("clutter_selected_elsewhere_unknown", 0),
                "empty_total": empty_total,
                "empty_tn": counts.get("empty_tn", 0),
                "empty_fp": counts.get("empty_fp", 0),
                "known_fp": known_fp,
                "miss_selected_unknown": counts.get("miss_selected_unknown", 0),
                "miss_no_select": counts.get("miss_no_select", 0),
            }
        )

    with (out_dir / "checkpoint_details.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    with (out_dir / "checkpoint_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(out_dir / "checkpoint_summary.csv")


if __name__ == "__main__":
    main()
