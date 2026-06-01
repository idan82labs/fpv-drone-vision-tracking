#!/usr/bin/env python3
"""Seed high-confidence taxonomy labels in a tube review packet.

This does not replace human review. It creates a derived CSV that can be used
for a quick training/replay gate while preserving uncertain rows for manual
labeling.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet_csv", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--target_tol_px", type=float, default=8.0)
    p.add_argument("--near_tol_px", type=float, default=16.0)
    p.add_argument("--negative_min_dist_px", type=float, default=24.0)
    p.add_argument("--prob_conf", type=float, default=0.62)
    return p.parse_args()


def fnum(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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


def bbox_from_text(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    try:
        vals = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
        return None
    try:
        return tuple(float(v) for v in vals)  # type: ignore[return-value]
    except Exception:
        return None


def bbox_from_label(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    if int(fnum(row.get("visible"), 0)) != 1:
        return None
    vals = [fnum(row.get(k), math.nan) for k in ("det_x", "det_y", "det_w", "det_h")]
    if any(not math.isfinite(v) for v in vals):
        return None
    return vals[0], vals[1], max(1.0, vals[2]), max(1.0, vals[3])


def center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + 0.5 * w, y + 0.5 * h


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def load_labels(path: Path) -> dict[tuple[str, int], tuple[bool, tuple[float, float, float, float] | None]]:
    out: dict[tuple[str, int], tuple[bool, tuple[float, float, float, float] | None]] = {}
    for row in read_csv(path):
        clip = str(row.get("clip", ""))
        frame = int(fnum(row.get("frame"), -1))
        if not clip or frame < 0:
            continue
        bbox = bbox_from_label(row)
        out[(clip, frame)] = (bbox is not None, bbox)
    return out


def model_clutter_label(row: dict[str, str], prob_conf: float) -> tuple[str, str]:
    probs = {
        "static_hotspot": fnum(row.get("crop_s_prob")),
        "attached_tree_branch_terrain": fnum(row.get("crop_e_prob")),
        "skyline_boundary_parallax": fnum(row.get("crop_h_prob")),
        "noise": fnum(row.get("crop_g_prob")),
    }
    best_label, best_prob = max(probs.items(), key=lambda item: item[1])
    line = max(fnum(row.get("cand_line_context")), fnum(row.get("tube_mean_line_context")))
    support = fnum(row.get("cand_attached_support"))
    texture = fnum(row.get("cand_texture"))
    bg_static = fnum(row.get("clba_bg_static_likelihood"))
    boundary_rate = fnum(row.get("tube_router_boundary_rate"))
    line_rate = fnum(row.get("tube_router_line_attached_rate"))
    pred = str(row.get("crop_pred_class", "")).strip()
    source = str(row.get("candidate_source", "")).strip()

    if best_prob >= prob_conf:
        return best_label, f"model_prob_{best_prob:.3f}"
    if boundary_rate >= 0.35 or (pred == "H" and best_prob >= 0.45):
        return "skyline_boundary_parallax", "boundary_rate_or_pred"
    if line_rate >= 0.35 or line >= 0.55 or support >= 8.0 or (pred == "E" and best_prob >= 0.45):
        return "attached_tree_branch_terrain", "line_support_or_pred"
    if bg_static >= 1.35 or (pred == "S" and fnum(row.get("crop_s_prob")) >= 0.45):
        return "static_hotspot", "static_likelihood_or_pred"
    if source == "large_dark" and texture >= 45.0:
        return "terrain_texture", "large_dark_textured"
    if pred == "G" and fnum(row.get("crop_g_prob")) >= 0.45:
        return "noise", "generic_pred"
    return "uncertain", "low_confidence"


def seed_row(
    row: dict[str, str],
    labels: dict[tuple[str, int], tuple[bool, tuple[float, float, float, float] | None]],
    target_tol_px: float,
    near_tol_px: float,
    negative_min_dist_px: float,
    prob_conf: float,
) -> tuple[str, str]:
    clip = str(row.get("clip", ""))
    frame = int(fnum(row.get("frame"), -1))
    visible, true_bbox = labels.get((clip, frame), (False, None))
    cand_bbox = bbox_from_text(row.get("bbox"))
    if visible and true_bbox is not None and cand_bbox is not None:
        dist = center_dist(cand_bbox, true_bbox)
        if dist <= target_tol_px:
            return "target", f"geometry_target_dist_{dist:.2f}"
        if dist <= near_tol_px:
            return "near_target_wrong_center", f"geometry_near_dist_{dist:.2f}"
        if dist < negative_min_dist_px:
            return "uncertain", f"geometry_gap_dist_{dist:.2f}"
    label, reason = model_clutter_label(row, prob_conf)
    if visible and true_bbox is not None and cand_bbox is not None:
        reason = f"{reason}_dist_{center_dist(cand_bbox, true_bbox):.2f}"
    return label, reason


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels))
    out_rows: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for row in read_csv(Path(args.packet_csv)):
        out = dict(row)
        existing = str(out.get("taxonomy_label", "") or out.get("human_label", "")).strip()
        if existing:
            label, reason = existing, "existing_manual_label"
        else:
            label, reason = seed_row(
                out,
                labels,
                args.target_tol_px,
                args.near_tol_px,
                args.negative_min_dist_px,
                args.prob_conf,
            )
        out["taxonomy_label"] = label
        out["taxonomy_seed_reason"] = reason
        counts[label] += 1
        out_rows.append(out)
    write_csv(Path(args.out_csv), out_rows)
    summary_path = Path(args.out_csv).with_suffix(".summary.csv")
    write_csv(summary_path, [{"taxonomy_label": k, "count": v} for k, v in sorted(counts.items())])
    print(args.out_csv)
    print(dict(counts))


if __name__ == "__main__":
    main()
