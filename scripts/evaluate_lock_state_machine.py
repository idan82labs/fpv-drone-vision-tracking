#!/usr/bin/env python3
"""Evaluate a simple acquire/lock/null selector from per-frame candidate scores.

This is an offline selector harness. It does not create proposals or train a
model; it asks whether the existing per-frame best candidate scores can be
turned into better full-video behavior with state instead of one global
threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Candidate:
    score: float
    track_score: float
    rank: int
    bbox: tuple[float, float, float, float]


@dataclass
class Label:
    visible: bool
    bbox: tuple[float, float, float, float] | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, help="CSV with frame, visible, det_x/det_y/det_w/det_h.")
    p.add_argument("--candidates", required=True, help="Candidate CSV with frame, score, x/y/w/h. Multiple rows per frame are allowed.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="", help="Optional clip id filter when labels/candidates contain multiple videos.")
    p.add_argument("--score_column", default="score", help="Score used while acquiring a target.")
    p.add_argument(
        "--track_score_column",
        default="",
        help="Optional separate score used after lock. Defaults to --score_column.",
    )
    p.add_argument("--max_rank", type=int, default=9999)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--acquire_thresholds", default="0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    p.add_argument("--track_thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    p.add_argument("--acquire_hits", default="1,2,3")
    p.add_argument("--max_misses", default="0,1,2,3")
    p.add_argument("--max_jump_px", default="12,18,24,32,48")
    p.add_argument("--output_tentative", action="store_true")
    p.add_argument("--coast_output", action="store_true", help="Emit predicted last box during misses. Usually bad for null suppression.")
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


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + 0.5 * w, y + 0.5 * h


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def load_labels(path: Path, clip: str = "") -> dict[int, Label]:
    out: dict[int, Label] = {}
    for row in read_csv(path):
        if clip and row.get("clip", "") != clip:
            continue
        frame_val = fnum(row.get("frame"), -1)
        frame = int(frame_val if frame_val is not None else -1)
        if frame < 0:
            continue
        visible = bool(int(fnum(row.get("visible"), 0) or 0))
        bbox = None
        if visible:
            x = fnum(row.get("det_x", row.get("x")))
            y = fnum(row.get("det_y", row.get("y")))
            w = fnum(row.get("det_w", row.get("w")), 1.0)
            h = fnum(row.get("det_h", row.get("h")), 1.0)
            if x is not None and y is not None and w is not None and h is not None:
                bbox = (x, y, w, h)
        out[frame] = Label(visible=visible, bbox=bbox)
    return out


def load_candidates(
    path: Path,
    score_column: str,
    max_rank: int,
    track_score_column: str = "",
    clip: str = "",
) -> dict[int, Candidate]:
    out: dict[int, Candidate] = {}
    for row in read_csv(path):
        if clip and row.get("clip", "") != clip:
            continue
        frame_val = fnum(row.get("frame"), -1)
        frame = int(frame_val if frame_val is not None else -1)
        rank = int(fnum(row.get("rank"), 0) or 0)
        if rank > max_rank:
            continue
        score = fnum(row.get(score_column))
        if score is None and score_column != "score":
            score = fnum(row.get("score"))
        track_score = None
        if track_score_column:
            track_score = fnum(row.get(track_score_column))
        if track_score is None:
            track_score = score
        x = fnum(row.get("x"))
        y = fnum(row.get("y"))
        w = fnum(row.get("w"), 1.0)
        h = fnum(row.get("h"), 1.0)
        if frame < 0 or score is None or track_score is None or x is None or y is None or w is None or h is None:
            continue
        cand = Candidate(score=score, track_score=track_score, rank=rank, bbox=(x, y, w, h))
        if frame not in out or cand.score > out[frame].score:
            out[frame] = cand
    return out


def evaluate_rows(
    labels: dict[int, Label],
    candidates: dict[int, Candidate],
    acquire_threshold: float,
    track_threshold: float,
    acquire_hits: int,
    max_misses: int,
    max_jump_px: float,
    strict_tol_px: float,
    loose_tol_px: float,
    output_tentative: bool,
    coast_output: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = "search"
    streak = 0
    misses = 0
    lock_frame: int | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    rows: list[dict[str, Any]] = []

    first_visible = next((f for f in sorted(labels) if labels[f].visible), None)
    first_strict = None

    for frame in sorted(labels):
        lab = labels[frame]
        cand = candidates.get(frame)
        selected_bbox = None
        selected = False
        reason = "none"
        acquire_score = None if cand is None else cand.score
        track_score = None if cand is None else cand.track_score

        jump_ok = True
        if cand is not None and last_bbox is not None:
            jump_ok = center_dist(cand.bbox, last_bbox) <= max_jump_px

        if state == "search":
            if cand is not None and acquire_score is not None and acquire_score >= acquire_threshold:
                streak = 1
                last_bbox = cand.bbox
                state = "tentative" if acquire_hits > 1 else "locked"
                reason = "acquire_start"
                if state == "locked":
                    lock_frame = frame
                    selected = True
                    selected_bbox = cand.bbox
                elif output_tentative:
                    selected = True
                    selected_bbox = cand.bbox
            else:
                streak = 0
        elif state == "tentative":
            if cand is not None and acquire_score is not None and acquire_score >= acquire_threshold and jump_ok:
                streak += 1
                last_bbox = cand.bbox
                reason = "acquire_continue"
                if streak >= acquire_hits:
                    state = "locked"
                    lock_frame = frame
                    selected = True
                    selected_bbox = cand.bbox
                elif output_tentative:
                    selected = True
                    selected_bbox = cand.bbox
            elif cand is not None and acquire_score is not None and acquire_score >= acquire_threshold:
                streak = 1
                last_bbox = cand.bbox
                reason = "acquire_restart_jump"
                if output_tentative:
                    selected = True
                    selected_bbox = cand.bbox
            else:
                state = "search"
                streak = 0
                last_bbox = None
        else:
            if cand is not None and track_score is not None and track_score >= track_threshold and jump_ok:
                misses = 0
                last_bbox = cand.bbox
                selected = True
                selected_bbox = cand.bbox
                reason = "track"
            else:
                misses += 1
                reason = "miss"
                if coast_output and last_bbox is not None and misses <= max_misses:
                    selected = True
                    selected_bbox = last_bbox
                    reason = "coast"
                if misses > max_misses:
                    state = "search"
                    streak = 0
                    misses = 0
                    last_bbox = None

        dist = ""
        strict_hit = False
        loose_hit = False
        if selected_bbox is not None and lab.visible and lab.bbox is not None:
            d = center_dist(selected_bbox, lab.bbox)
            dist = round(d, 3)
            strict_hit = d <= strict_tol_px
            loose_hit = d <= loose_tol_px
            if strict_hit and first_strict is None:
                first_strict = frame
        correct = (strict_hit if lab.visible else not selected)
        rows.append(
            {
                "frame": frame,
                "visible": int(lab.visible),
                "state": state,
                "candidate_score": "" if acquire_score is None else round(acquire_score, 6),
                "track_score": "" if track_score is None else round(track_score, 6),
                "candidate_rank": "" if cand is None else cand.rank,
                "selected": int(selected),
                "reason": reason,
                "dist_px": dist,
                "strict_hit": strict_hit,
                "loose_hit": loose_hit,
                "correct_all_frame": correct,
            }
        )

    visible_rows = [r for r in rows if r["visible"]]
    invisible_rows = [r for r in rows if not r["visible"]]
    selected_rows = [r for r in rows if r["selected"]]
    visible_strict = sum(bool(r["strict_hit"]) for r in visible_rows)
    visible_loose = sum(bool(r["loose_hit"]) for r in visible_rows)
    invisible_no_box = sum(not bool(r["selected"]) for r in invisible_rows)
    correct = sum(bool(r["correct_all_frame"]) for r in rows)
    summary = {
        "acquire_threshold": acquire_threshold,
        "track_threshold": track_threshold,
        "acquire_hits": acquire_hits,
        "max_misses": max_misses,
        "max_jump_px": max_jump_px,
        "output_tentative": int(output_tentative),
        "coast_output": int(coast_output),
        "frames_all": len(rows),
        "visible_frames": len(visible_rows),
        "invisible_frames": len(invisible_rows),
        "all_frame_correct": correct,
        "all_frame_accuracy": round(correct / max(1, len(rows)), 4),
        "visible_strict": visible_strict,
        "visible_strict_recall": round(visible_strict / max(1, len(visible_rows)), 4),
        "visible_loose": visible_loose,
        "visible_loose_recall": round(visible_loose / max(1, len(visible_rows)), 4),
        "invisible_no_box": invisible_no_box,
        "invisible_no_box_rate": round(invisible_no_box / max(1, len(invisible_rows)), 4),
        "selected_frames": len(selected_rows),
        "first_lock_frame": "" if lock_frame is None else lock_frame,
        "first_strict_frame": "" if first_strict is None else first_strict,
        "strict_latency_frames": (
            "" if first_visible is None or first_strict is None else max(0, first_strict - first_visible)
        ),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels), args.clip)
    candidates = load_candidates(
        Path(args.candidates),
        args.score_column,
        args.max_rank,
        args.track_score_column,
        args.clip,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] | None = None
    best_score = (-1.0, -1.0, -1.0, 0)
    same_score_scale = not args.track_score_column or args.track_score_column == args.score_column
    for aq in parse_float_list(args.acquire_thresholds):
        for tr in parse_float_list(args.track_thresholds):
            if same_score_scale and tr > aq:
                continue
            for hits in parse_int_list(args.acquire_hits):
                for misses in parse_int_list(args.max_misses):
                    for jump in parse_float_list(args.max_jump_px):
                        summary, rows = evaluate_rows(
                            labels,
                            candidates,
                            aq,
                            tr,
                            hits,
                            misses,
                            jump,
                            args.strict_tol_px,
                            args.loose_tol_px,
                            args.output_tentative,
                            args.coast_output,
                        )
                        summaries.append(summary)
                        score = (
                            summary["all_frame_accuracy"],
                            summary["visible_strict_recall"],
                            summary["invisible_no_box_rate"],
                            -summary["selected_frames"],
                        )
                        if score > best_score:
                            best_score = score
                            best_rows = rows

    summaries.sort(
        key=lambda r: (
            r["all_frame_accuracy"],
            r["visible_strict_recall"],
            r["invisible_no_box_rate"],
            -r["selected_frames"],
        ),
        reverse=True,
    )
    write_csv(out_dir / "state_machine_sweep.csv", summaries)
    write_csv(out_dir / "best_frame_predictions.csv", best_rows or [])
    (out_dir / "best_config.json").write_text(json.dumps(summaries[0] if summaries else {}, indent=2))
    print(out_dir / "state_machine_sweep.csv")
    if summaries:
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
