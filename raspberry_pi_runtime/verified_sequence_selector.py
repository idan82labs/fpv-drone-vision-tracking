#!/usr/bin/env python3
"""Lightweight Viterbi selector for Pi-style top-tube exports.

This uses only the detector's own ``verified_score`` and a small continuity
prior. It intentionally avoids sklearn/joblib so it can run on a Raspberry Pi
without dragging the learned-ranker stack into the hot path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "clip",
    "frame",
    "selected",
    "rank",
    "x",
    "y",
    "w",
    "h",
    "learned_score",
    "verified_score",
    "source",
    "track_id",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--max_rank", type=int, default=20)
    p.add_argument("--max_jump_px", type=float, default=10.0)
    p.add_argument("--transition_weight", type=float, default=1.5)
    p.add_argument("--score_column", default="verified_score")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--allow_ineligible", action="store_true", help="Allow rows failing detector eligible=1.")
    p.add_argument("--require_floor", action="store_true", help="Require rows to pass detector passes_floor=1.")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = list(fieldnames or [])
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


def inum(value: Any, default: int = 0) -> int:
    return int(fnum(value, float(default)))


def true_flag(row: dict[str, Any], key: str, default: bool = True) -> bool:
    if key not in row or row.get(key) == "":
        return default
    return bool(inum(row.get(key), 0))


def center(row: dict[str, Any]) -> tuple[float, float]:
    return fnum(row.get("x")) + 0.5 * fnum(row.get("w"), 1.0), fnum(row.get("y")) + 0.5 * fnum(row.get("h"), 1.0)


def center_dist(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def load_rows(
    path: Path,
    max_rank: int,
    score_column: str,
    allow_ineligible: bool = False,
    require_floor: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in read_csv(path):
        rank = inum(row.get("rank"), 999999)
        if rank > max_rank:
            continue
        if not allow_ineligible and not true_flag(row, "eligible", True):
            continue
        if require_floor and not true_flag(row, "passes_floor", True):
            continue
        frame = inum(row.get("frame"), -1)
        if frame < 0:
            continue
        rec = dict(row)
        rec["rank"] = rank
        rec["frame"] = frame
        rec["selector_score"] = fnum(row.get(score_column), fnum(row.get("score")))
        by_frame.setdefault(frame, []).append(rec)
    for rows in by_frame.values():
        rows.sort(key=lambda r: (-fnum(r.get("selector_score")), inum(r.get("rank"), 999999)))
    return by_frame


def viterbi_select(
    by_frame: dict[int, list[dict[str, Any]]],
    max_jump_px: float,
    transition_weight: float,
) -> dict[int, dict[str, Any]]:
    frames = sorted(by_frame)
    if not frames:
        return {}
    layers: list[list[dict[str, Any]]] = []
    backptrs: list[list[int | None]] = []
    prev_scores: list[float] = []
    for fi, frame in enumerate(frames):
        rows = by_frame[frame]
        layers.append(rows)
        current: list[float] = []
        current_bp: list[int | None] = []
        if fi == 0:
            for row in rows:
                current.append(fnum(row.get("selector_score")))
                current_bp.append(None)
        else:
            prev_rows = layers[fi - 1]
            gap = max(1, frame - frames[fi - 1])
            allowed = max_jump_px * gap
            for row in rows:
                best_score = -1e18
                best_idx: int | None = None
                for pi, prev in enumerate(prev_rows):
                    jump = center_dist(row, prev)
                    if jump > allowed:
                        continue
                    cost = transition_weight * (jump / max(1e-6, allowed)) ** 2
                    score = prev_scores[pi] + fnum(row.get("selector_score")) - cost
                    if score > best_score:
                        best_score = score
                        best_idx = pi
                if best_idx is None:
                    best_score = fnum(row.get("selector_score")) - 1.0
                current.append(best_score)
                current_bp.append(best_idx)
        prev_scores = current
        backptrs.append(current_bp)
    best_last = max(range(len(prev_scores)), key=lambda i: prev_scores[i])
    selected_pairs: list[tuple[int, int]] = []
    fi = len(frames) - 1
    idx: int | None = best_last
    while fi >= 0 and idx is not None:
        selected_pairs.append((fi, idx))
        idx = backptrs[fi][idx]
        fi -= 1
    selected_pairs.reverse()
    return {frames[fi]: layers[fi][idx] for fi, idx in selected_pairs}


def output_rows(
    clip: str,
    by_frame: dict[int, list[dict[str, Any]]],
    selected: dict[int, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in sorted(by_frame):
        row = selected.get(frame)
        score = fnum(row.get("selector_score")) if row is not None else 0.0
        keep = row is not None and score >= threshold
        if not keep:
            continue
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "selected": 1,
                "rank": row.get("rank", ""),
                "x": row.get("x", ""),
                "y": row.get("y", ""),
                "w": row.get("w", ""),
                "h": row.get("h", ""),
                "learned_score": round(score, 6),
                "verified_score": row.get("verified_score", ""),
                "source": row.get("cand_source", ""),
                "track_id": row.get("track_id", ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    by_frame = load_rows(
        Path(args.top_tubes),
        args.max_rank,
        args.score_column,
        allow_ineligible=args.allow_ineligible,
        require_floor=args.require_floor,
    )
    selected = viterbi_select(by_frame, args.max_jump_px, args.transition_weight)
    rows = output_rows(args.clip, by_frame, selected, args.threshold)
    out_path = Path(args.out_csv)
    write_csv(out_path, rows, OUTPUT_FIELDS)
    summary = {
        "top_tubes": args.top_tubes,
        "out_csv": str(out_path),
        "clip": args.clip,
        "candidate_frames": len(by_frame),
        "selected_frames": sum(int(r["selected"]) for r in rows),
        "max_rank": args.max_rank,
        "max_jump_px": args.max_jump_px,
        "transition_weight": args.transition_weight,
        "score_column": args.score_column,
        "threshold": args.threshold,
    }
    (out_path.parent / "verified_sequence_summary.json").write_text(json.dumps(summary, indent=2))
    print(out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
