#!/usr/bin/env python3
"""Seed conservative review labels for selector-disagreement packets.

This converts a rendered disagreement review packet into a first-pass labeled
CSV. The labels are intentionally conservative and are meant to bootstrap router
training/review, not replace human review. Rows where both selector families are
wrong are marked as null-override/reselect cases rather than being forced into
the binary HMM-vs-Viterbi mode-supervisor target.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", required=True)
    p.add_argument("--out", required=True)
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


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def false_lock_kind(row: dict[str, str]) -> str:
    clip = row.get("clip", "")
    category = row.get("category", "")
    y = max(fnum(row.get("selector_a_y")), fnum(row.get("selector_b_y")))
    if category.startswith("both_visible") or category in {"a_visible_hit_b_miss", "b_visible_hit_a_miss"}:
        return "visible_mode_error"
    if clip.startswith("b96"):
        return "sky_haze_speck"
    if clip.startswith("1c"):
        return "cloud_sky_speck"
    if clip.startswith("7bd"):
        return "horizon_field_texture"
    if clip.startswith("59e"):
        return "terrain_horizon_texture"
    if clip.startswith("aaf1"):
        return "cloud_sky_speck"
    if clip.startswith("d129"):
        return "cloud_sky_speck" if y < 95.0 else "tree_terrain_edge"
    if clip.startswith("e6"):
        return "terrain_tree_edge" if y >= 90.0 else "skyline_cloud_edge"
    return "hard_clutter"


def router_label(row: dict[str, str]) -> str:
    category = row.get("category", "")
    visible = row.get("visible", "") == "1"
    if visible:
        if category == "a_visible_hit_b_miss":
            return "protect_continuous_visible"
        if category == "b_visible_hit_a_miss":
            return "hmm_can_track_visible"
        if category == "both_visible_wrong_box":
            return "visible_reselect_needed"
        if category == "both_visible_no_box_miss":
            return "visible_acquisition_miss"
        return "visible_review_needed"
    if category == "a_false_b_suppressed":
        return "hard_null_use_hmm"
    if category == "both_null_false_box":
        return "hard_null_needs_override"
    return "null_review_needed"


def binary_mode_target(label: str) -> str:
    if label == "hard_null_use_hmm":
        return "hmm"
    if label == "hmm_can_track_visible":
        return "hmm"
    if label == "protect_continuous_visible":
        return "viterbi"
    return ""


def confidence(row: dict[str, str], label: str) -> str:
    source_conf = row.get("confidence", "")
    if label in {"visible_reselect_needed", "visible_acquisition_miss", "hard_null_needs_override"}:
        return "medium_high"
    if source_conf in {"high", "high_not_visible"}:
        return "high"
    if source_conf:
        return source_conf
    return "medium_high"


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.packet))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        label = router_label(row)
        visible = row.get("visible", "") == "1"
        reviewed = dict(row)
        reviewed["target_visible"] = "1" if visible else "0"
        if visible:
            reviewed["target_x"] = row.get("det_x", "")
            reviewed["target_y"] = row.get("det_y", "")
            reviewed["target_w"] = row.get("det_w", "")
            reviewed["target_h"] = row.get("det_h", "")
        else:
            reviewed["target_x"] = ""
            reviewed["target_y"] = ""
            reviewed["target_w"] = ""
            reviewed["target_h"] = ""
        reviewed["false_lock_kind"] = false_lock_kind(row)
        reviewed["router_label"] = label
        reviewed["binary_mode_target"] = binary_mode_target(label)
        reviewed["review_confidence"] = confidence(row, label)
        reviewed["review_notes"] = (
            "Codex contact-sheet seed label; use for training only when review_confidence is high/medium_high "
            "and binary_mode_target is non-empty for binary router training."
        )
        out_rows.append(reviewed)
    write_csv(Path(args.out), out_rows)
    print(args.out)


if __name__ == "__main__":
    main()
