#!/usr/bin/env python3
"""Apply a learned surface ranker with a continuity/Viterbi selector.

This is an offline/deferred selector for exported ``top_tubes.csv`` rows. It is
meant to test the runtime direction before putting a delayed sliding-window
version into ``tbd_motion_detector.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib

try:
    import evaluate_xy_sequence_ranker as seq
    import train_surface_xy_ranker as surface
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import evaluate_xy_sequence_ranker as seq
    from scripts import train_surface_xy_ranker as surface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True)
    p.add_argument("--model", required=True, help="Surface ranker .joblib bundle.")
    p.add_argument("--clip", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--scored_csv", default="")
    p.add_argument("--max_rank", type=int, default=40)
    p.add_argument("--max_jump_px", type=float, default=12.0)
    p.add_argument("--transition_weight", type=float, default=0.35)
    p.add_argument("--size_jump_weight", type=float, default=0.0)
    p.add_argument(
        "--sequence_window",
        type=int,
        default=0,
        help="Use rolling-window Viterbi over this many frames. 0 keeps the legacy full-video path.",
    )
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument(
        "--acquire_threshold",
        type=float,
        default=None,
        help=(
            "Optional track-acquisition threshold. When set, selections only "
            "start after a candidate reaches this score."
        ),
    )
    p.add_argument(
        "--keep_threshold",
        type=float,
        default=None,
        help="Score threshold for keeping an acquired track. Defaults to --threshold.",
    )
    p.add_argument(
        "--hysteresis_max_jump_px",
        type=float,
        default=None,
        help="Maximum accepted jump while tracking. Defaults to --max_jump_px.",
    )
    p.add_argument(
        "--lost_patience",
        type=int,
        default=0,
        help="Consecutive failed keep frames allowed before returning to acquisition.",
    )
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


def load_ranked_rows(path: Path, max_rank: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(path):
        rank = surface.int_or_default(row.get("rank"), 999999)
        if rank > max_rank:
            continue
        rows.append(row)
    rows.sort(
        key=lambda r: (
            surface.int_or_default(r.get("frame"), 0),
            surface.int_or_default(r.get("rank"), 999999),
        )
    )
    return rows


def score_rows(rows: list[dict[str, str]], model_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    numeric = list(bundle["numeric_features"])
    sources = list(bundle["source_features"])
    scores = surface.predict_score(model, surface.vectorize(rows, numeric, sources))
    scored: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        out = dict(row)
        out["learned_score"] = float(score)
        scored.append(out)
    meta = {
        "model": str(model_path),
        "numeric_features": numeric,
        "source_features": sources,
        "max_rank_from_model": bundle.get("max_rank", ""),
        "final_exclude_clip": bundle.get("final_exclude_clip", ""),
    }
    return scored, meta


def group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame = surface.int_or_default(row.get("frame"), 0)
        by_frame.setdefault(frame, []).append(row)
    return by_frame


def bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        surface.float_or_default(row.get("x"), 0.0),
        surface.float_or_default(row.get("y"), 0.0),
        surface.float_or_default(row.get("w"), 1.0),
        surface.float_or_default(row.get("h"), 1.0),
    )


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def score(row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    return float(row.get("learned_score", 0.0) or 0.0)


def apply_hysteresis_gate(
    selected: dict[int, dict[str, Any]],
    acquire_threshold: float,
    keep_threshold: float,
    max_jump_px: float,
    lost_patience: int = 0,
) -> dict[int, dict[str, Any]]:
    """Gate a selected candidate stream with acquire/keep state.

    This is intentionally simpler than Viterbi: acquisition is score-first, then
    the tracker only keeps boxes that remain plausible by score and frame-to-frame
    jump. It targets the current failure mode where background branches are
    emitted before a real target appears.
    """

    out: dict[int, dict[str, Any]] = {}
    active = False
    lost = 0
    last_frame: int | None = None
    last_row: dict[str, Any] | None = None
    for frame in sorted(selected):
        row = selected[frame]
        row_score = score(row)
        emit = False
        if not active:
            if row_score >= acquire_threshold:
                active = True
                emit = True
                lost = 0
        else:
            gap = max(1, frame - last_frame) if last_frame is not None else 1
            allowed_jump = max_jump_px * gap
            jump_ok = last_row is None or center_dist(bbox(last_row), bbox(row)) <= allowed_jump
            if row_score >= keep_threshold and jump_ok:
                emit = True
                lost = 0
            else:
                lost += 1
                if lost > max(0, lost_patience):
                    active = False
                    last_row = None
                    last_frame = None
        if emit:
            out[frame] = row
            last_row = row
            last_frame = frame
    return out


def output_rows(
    clip: str,
    scored_rows: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    frames = sorted({surface.int_or_default(row.get("frame"), 0) for row in scored_rows})
    rows: list[dict[str, Any]] = []
    for frame in frames:
        row = selected.get(frame)
        score = float(row.get("learned_score", 0.0) or 0.0) if row is not None else 0.0
        is_selected = row is not None and score >= threshold
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "selected": int(is_selected),
                "rank": surface.int_or_default(row.get("rank"), 0) if is_selected else "",
                "learned_score": round(score, 6) if row is not None else "",
                "threshold": threshold,
                "x": row.get("x", "") if is_selected else "",
                "y": row.get("y", "") if is_selected else "",
                "w": row.get("w", "") if is_selected else "",
                "h": row.get("h", "") if is_selected else "",
                "verified_score": row.get("verified_score", "") if is_selected else "",
                "source": row.get("cand_source", "") if is_selected else "",
                "track_id": row.get("track_id", "") if is_selected else "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = load_ranked_rows(Path(args.top_tubes), args.max_rank)
    if not rows:
        raise SystemExit("no top-tube rows loaded")
    scored, meta = score_rows(rows, Path(args.model))
    by_frame = group_by_frame(scored)
    if args.sequence_window > 0:
        selected = seq.rolling_viterbi_select(
            by_frame,
            max_jump_px=args.max_jump_px,
            transition_weight=args.transition_weight,
            size_jump_weight=args.size_jump_weight,
            sequence_window=args.sequence_window,
        )
    else:
        selected = seq.viterbi_select(
            by_frame,
            max_jump_px=args.max_jump_px,
            transition_weight=args.transition_weight,
            size_jump_weight=args.size_jump_weight,
        )
    if args.acquire_threshold is not None:
        selected = apply_hysteresis_gate(
            selected,
            acquire_threshold=args.acquire_threshold,
            keep_threshold=args.keep_threshold if args.keep_threshold is not None else args.threshold,
            max_jump_px=args.hysteresis_max_jump_px if args.hysteresis_max_jump_px is not None else args.max_jump_px,
            lost_patience=args.lost_patience,
        )
    out_rows = output_rows(args.clip, scored, selected, args.threshold)
    out_path = Path(args.out_csv)
    write_csv(out_path, out_rows)
    if args.scored_csv:
        write_csv(Path(args.scored_csv), scored)
    summary = {
        "top_tubes": args.top_tubes,
        "output": str(out_path),
        "clip": args.clip,
        "rows": len(rows),
        "frames": len(out_rows),
        "selected_frames": sum(int(r["selected"]) for r in out_rows),
        "max_rank": args.max_rank,
        "max_jump_px": args.max_jump_px,
        "transition_weight": args.transition_weight,
        "size_jump_weight": args.size_jump_weight,
        "sequence_window": args.sequence_window,
        "threshold": args.threshold,
        "acquire_threshold": args.acquire_threshold,
        "keep_threshold": args.keep_threshold,
        "hysteresis_max_jump_px": args.hysteresis_max_jump_px,
        "lost_patience": args.lost_patience,
        **meta,
    }
    (out_path.parent / "sequence_selector_summary.json").write_text(json.dumps(summary, indent=2))
    print(out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
