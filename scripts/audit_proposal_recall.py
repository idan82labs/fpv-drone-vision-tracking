#!/usr/bin/env python3
"""
Audit whether reviewed gold boxes appear in exported top-tube alternatives.

Run a detector pass with --export_top_tubes first, then use this script to split
misses into proposal failures vs ranking/threshold failures.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--top_tube_root", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--label", default="gold")
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


def load_top_rows(root: Path, clip: str, frame_no: int) -> tuple[bool, list[dict[str, str]]]:
    path = root / clip / "top_tubes.csv"
    if not path.exists():
        return False, []
    with path.open() as f:
        return True, [r for r in csv.DictReader(f) if int(r.get("frame", -1)) == frame_no]


def main() -> None:
    args = parse_args()
    label_path = Path(args.labels)
    root = Path(args.top_tube_root)
    with label_path.open() as f:
        labels = list(csv.DictReader(f))

    rows: list[dict[str, Any]] = []
    for lab in labels:
        if lab.get("label", "").strip().lower() != args.label:
            continue
        clip = lab.get("clip", "")
        frame_no = int(lab.get("frame", "0") or 0)
        target = bbox_from_text(lab.get("selected_bbox"))
        if target is None:
            continue

        has_export, top_rows = load_top_rows(root, clip, frame_no)
        if not has_export:
            rows.append(
                {
                    "clip": clip,
                    "frame": frame_no,
                    "target_bbox": target,
                    "status": "no_top_tube_export",
                    "found_in_top_tubes": "",
                    "best_rank": "",
                    "center_dist_px": "",
                    "iou": "",
                    "best_bbox": "",
                    "verified_score": "",
                    "raw_score": "",
                    "tube_verifier_score": "",
                    "eligible": "",
                    "passes_floor": "",
                    "notes": lab.get("notes", ""),
                }
            )
            continue

        matches: list[tuple[int, float, float, dict[str, str], tuple[int, int, int, int]]] = []
        for r in top_rows:
            b = (
                int(float(r["x"])),
                int(float(r["y"])),
                int(float(r["w"])),
                int(float(r["h"])),
            )
            d = center_dist(target, b)
            ov = iou(target, b)
            if d <= args.center_tol_px or ov >= args.iou_tol:
                matches.append((int(r["rank"]), d, ov, r, b))

        matches.sort(key=lambda x: x[0])
        best = matches[0] if matches else None
        rows.append(
            {
                "clip": clip,
                "frame": frame_no,
                "target_bbox": target,
                "status": "audited",
                "found_in_top_tubes": bool(best),
                "best_rank": best[0] if best else "",
                "center_dist_px": round(best[1], 3) if best else "",
                "iou": round(best[2], 4) if best else "",
                "best_bbox": best[4] if best else "",
                "verified_score": best[3].get("verified_score", "") if best else "",
                "raw_score": best[3].get("score", "") if best else "",
                "tube_verifier_score": best[3].get("tube_verifier_score", "") if best else "",
                "eligible": best[3].get("eligible", "") if best else "",
                "passes_floor": best[3].get("passes_floor", "") if best else "",
                "notes": lab.get("notes", ""),
            }
        )

    with Path(args.out_csv).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    audited = [r for r in rows if r["status"] == "audited"]
    found = sum(1 for r in audited if r["found_in_top_tubes"])
    print(f"{Path(args.out_csv)} | found {found}/{len(audited)} audited gold boxes")


if __name__ == "__main__":
    main()
