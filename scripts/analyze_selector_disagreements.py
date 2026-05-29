#!/usr/bin/env python3
"""Analyze where two selector outputs disagree against frame labels.

The current tracker work has two useful but incomplete selector families:

* permissive Viterbi keeps continuity but often emits boxes through null frames;
* conservative HMM/null mode suppresses clutter but misses continuous-visible
  segments.

This script turns that tradeoff into concrete training/evaluation rows: frames
where one selector is right and the other is wrong, null frames where a selector
hallucinates, and visible frames where both selectors miss. Those rows are the
best next candidates for router/ranker training or manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import evaluate_tracking_run as tracking_eval
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import evaluate_tracking_run as tracking_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--selector_a_dir", required=True)
    p.add_argument("--selector_b_dir", required=True)
    p.add_argument("--selector_a_name", default="a")
    p.add_argument("--selector_b_name", default="b")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
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


def labels_by_clip(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        clip = row.get("clip", "")
        frame = tracking_eval.fnum(row.get("frame"))
        if clip and frame is not None:
            grouped[clip].append(row)
    return dict(sorted(grouped.items()))


def selected_by_frame(selector_dir: Path, clip: str) -> dict[int, dict[str, str]]:
    path = selector_dir / clip / "selected_tracks.csv"
    if not path.exists():
        return {}
    out: dict[int, dict[str, str]] = {}
    for row in read_csv(path):
        if not tracking_eval.row_is_selected(row):
            continue
        frame = tracking_eval.fnum(row.get("frame"))
        if frame is None:
            continue
        out[int(frame)] = row
    return out


def selected_box(row: dict[str, str] | None) -> tuple[float, float, float, float] | None:
    if row is None:
        return None
    try:
        return tracking_eval.row_bbox(row)
    except Exception:
        return None


def center_distance(
    label_box: tuple[float, float, float, float] | None,
    selector_box: tuple[float, float, float, float] | None,
) -> float | None:
    if label_box is None or selector_box is None:
        return None
    out = tracking_eval.center_dist(label_box, selector_box)
    return out if math.isfinite(out) else None


def selector_status(
    label: dict[str, str],
    selected: dict[str, str] | None,
    strict_tol: float,
    loose_tol: float,
) -> dict[str, Any]:
    is_visible = tracking_eval.visible(label)
    box = selected_box(selected)
    label_box = tracking_eval.label_bbox(label) if is_visible else None
    dist = center_distance(label_box, box)
    strict = dist is not None and dist <= strict_tol
    loose = dist is not None and dist <= loose_tol
    has_box = box is not None
    if is_visible:
        if strict:
            status = "strict_hit"
        elif loose:
            status = "loose_only"
        elif has_box:
            status = "wrong_box"
        else:
            status = "miss"
    else:
        status = "false_box" if has_box else "no_box"
    return {
        "status": status,
        "selected": int(has_box),
        "strict_hit": int(strict),
        "loose_hit": int(loose),
        "dist_px": "" if dist is None else round(dist, 3),
        "x": "" if box is None else box[0],
        "y": "" if box is None else box[1],
        "w": "" if box is None else box[2],
        "h": "" if box is None else box[3],
    }


def category(visible: bool, a_status: str, b_status: str) -> str:
    a_hit = a_status in {"strict_hit", "loose_only"}
    b_hit = b_status in {"strict_hit", "loose_only"}
    if visible:
        if a_hit and b_hit:
            return "both_visible_hit"
        if a_hit and not b_hit:
            return "a_visible_hit_b_miss"
        if b_hit and not a_hit:
            return "b_visible_hit_a_miss"
        if a_status == "wrong_box" and b_status == "wrong_box":
            return "both_visible_wrong_box"
        if a_status == "miss" and b_status == "miss":
            return "both_visible_no_box_miss"
        return "both_visible_miss_or_wrong"
    if a_status == "no_box" and b_status == "no_box":
        return "both_null_suppressed"
    if a_status == "false_box" and b_status == "no_box":
        return "a_false_b_suppressed"
    if b_status == "false_box" and a_status == "no_box":
        return "b_false_a_suppressed"
    return "both_null_false_box"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels_by_clip(Path(args.labels))
    selector_a_dir = Path(args.selector_a_dir)
    selector_b_dir = Path(args.selector_b_dir)

    rows: list[dict[str, Any]] = []
    summary_by_clip: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()

    for clip, clip_labels in labels.items():
        a_selected = selected_by_frame(selector_a_dir, clip)
        b_selected = selected_by_frame(selector_b_dir, clip)
        clip_counts: Counter[str] = Counter()
        for label in clip_labels:
            frame = int(float(label["frame"]))
            is_visible = tracking_eval.visible(label)
            a = selector_status(label, a_selected.get(frame), args.strict_tol_px, args.loose_tol_px)
            b = selector_status(label, b_selected.get(frame), args.strict_tol_px, args.loose_tol_px)
            cat = category(is_visible, a["status"], b["status"])
            clip_counts[cat] += 1
            global_counts[cat] += 1
            if cat.startswith("both_") and cat not in {"both_visible_no_box_miss", "both_visible_wrong_box", "both_null_false_box"}:
                # Keep the main CSV focused on actionable disagreements/failures.
                continue
            rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "time_s": label.get("time_s", ""),
                    "visible": int(is_visible),
                    "category": cat,
                    f"{args.selector_a_name}_status": a["status"],
                    f"{args.selector_a_name}_selected": a["selected"],
                    f"{args.selector_a_name}_strict_hit": a["strict_hit"],
                    f"{args.selector_a_name}_loose_hit": a["loose_hit"],
                    f"{args.selector_a_name}_dist_px": a["dist_px"],
                    f"{args.selector_a_name}_x": a["x"],
                    f"{args.selector_a_name}_y": a["y"],
                    f"{args.selector_a_name}_w": a["w"],
                    f"{args.selector_a_name}_h": a["h"],
                    f"{args.selector_b_name}_status": b["status"],
                    f"{args.selector_b_name}_selected": b["selected"],
                    f"{args.selector_b_name}_strict_hit": b["strict_hit"],
                    f"{args.selector_b_name}_loose_hit": b["loose_hit"],
                    f"{args.selector_b_name}_dist_px": b["dist_px"],
                    f"{args.selector_b_name}_x": b["x"],
                    f"{args.selector_b_name}_y": b["y"],
                    f"{args.selector_b_name}_w": b["w"],
                    f"{args.selector_b_name}_h": b["h"],
                    "det_x": label.get("det_x", label.get("x", "")),
                    "det_y": label.get("det_y", label.get("y", "")),
                    "det_w": label.get("det_w", label.get("w", "")),
                    "det_h": label.get("det_h", label.get("h", "")),
                    "confidence": label.get("confidence", ""),
                    "notes": label.get("notes", ""),
                }
            )
        summary_by_clip.append({"clip": clip, **dict(sorted(clip_counts.items()))})

    summary_rows = [{"clip": "__ALL__", **dict(sorted(global_counts.items()))}, *summary_by_clip]
    write_csv(out_dir / "selector_disagreements.csv", rows)
    write_csv(out_dir / "selector_disagreement_summary.csv", summary_rows)
    metadata = {
        "labels": str(Path(args.labels)),
        "selector_a_dir": str(selector_a_dir),
        "selector_b_dir": str(selector_b_dir),
        "selector_a_name": args.selector_a_name,
        "selector_b_name": args.selector_b_name,
        "strict_tol_px": args.strict_tol_px,
        "loose_tol_px": args.loose_tol_px,
        "actionable_rows": len(rows),
    }
    (out_dir / "selector_disagreement_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"summary": summary_rows[0], "actionable_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
