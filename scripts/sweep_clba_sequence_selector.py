#!/usr/bin/env python3
"""Sweep CLBA-adjusted OOF scores through a continuity selector.

This is the offline bridge between the CLBA score-modifier result and the
delayed/rolling sequence selector.  It consumes per-candidate out-of-fold scores
that already include the original top-tube feature columns, optionally adjusts
those scores with CLBA target/background terms, then evaluates framewise,
rolling-window, and full-window Viterbi selection against frame-level labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

try:
    import evaluate_xy_sequence_ranker as seq
    import sweep_clba_score_adjustment as clba_adjust
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import evaluate_xy_sequence_ranker as seq
    from scripts import sweep_clba_score_adjustment as clba_adjust


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--candidates", required=True, help="OOF per-candidate CSV with x/y/w/h and score columns.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--score_column", default="score")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--sequence_beam", type=int, default=20, help="Top adjusted-score candidates per frame used by Viterbi. 0 keeps all.")
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--sequence_windows", default="1,5,9,0", help="1=framewise, 0=full-video Viterbi, >1=rolling Viterbi.")
    p.add_argument("--max_jump_px", default="10,12,18,32")
    p.add_argument("--transition_weights", default="0.1,0.35,0.5")
    p.add_argument("--size_jump_weights", default="0")
    p.add_argument("--thresholds", default="0,0.2,0.35,0.5,0.65,0.75,0.85,0.9")
    p.add_argument(
        "--acquire_thresholds",
        default="",
        help="Optional hysteresis acquire thresholds. When set, evaluated in addition to plain thresholds.",
    )
    p.add_argument("--acquire_hits", default="1")
    p.add_argument("--keep_thresholds", default="", help="Optional hysteresis keep thresholds.")
    p.add_argument("--lost_patience", default="0,1")
    p.add_argument("--gain_weights", default="0,0.15,0.3")
    p.add_argument("--path_weights", default="0,0.1")
    p.add_argument("--target_q_weights", default="0,0.05")
    p.add_argument("--bg_weights", default="0,0.1,0.2")
    p.add_argument("--attached_weights", default="0,0.1,0.2")
    p.add_argument("--density_weights", default="0,0.03")
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
    return [int(float(x)) for x in raw.split(",") if x.strip()]


def label_visible(row: dict[str, str]) -> bool:
    raw = str(row.get("visible", "")).strip().lower()
    if raw in {"0", "false", "no", "empty", "none", "not_visible", "not visible"}:
        return False
    return True


def label_bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    if not label_visible(row):
        return None
    x = fnum(row.get("det_x", row.get("x")), math.nan)
    y = fnum(row.get("det_y", row.get("y")), math.nan)
    w = fnum(row.get("det_w", row.get("w")), math.nan)
    h = fnum(row.get("det_h", row.get("h")), math.nan)
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    return x, y, w, h


def load_labels(path: Path, clip: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        if clip and row.get("clip", clip) != clip:
            continue
        frame = int(fnum(row.get("frame"), -1))
        if frame < 0:
            continue
        bbox = label_bbox(row)
        rows.append(
            {
                "clip": row.get("clip", clip),
                "frame": frame,
                "visible": int(bbox is not None),
                "bbox": bbox,
                "confidence": row.get("confidence", ""),
                "notes": row.get("notes", ""),
            }
        )
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def load_candidates(path: Path, clip: str, max_rank: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        if clip and row.get("clip", clip) != clip:
            continue
        frame = int(fnum(row.get("frame"), -1))
        rank = int(fnum(row.get("rank"), 999999))
        if frame < 0 or rank > max_rank:
            continue
        out: dict[str, Any] = dict(row)
        out["frame"] = frame
        out["rank"] = rank
        rows.append(out)
    rows.sort(key=lambda r: (int(r["frame"]), int(r["rank"])))
    return rows


def bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        fnum(row.get("x")),
        fnum(row.get("y")),
        max(1.0, fnum(row.get("w"), 1.0)),
        max(1.0, fnum(row.get("h"), 1.0)),
    )


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return float(math.hypot(ax - bx, ay - by))


def score_rows(
    rows: list[dict[str, Any]],
    weights: clba_adjust.Weights,
    score_column: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        scored = dict(row)
        base = fnum(scored.get(score_column), fnum(scored.get("score")))
        scored["base_learned_score"] = base
        scored["learned_score"] = clba_adjust.adjusted_score(scored, weights, score_column)
        out.append(scored)
    return out


def group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame"]), []).append(row)
    return by_frame


def framewise_best(by_frame: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for frame, rows in by_frame.items():
        out[frame] = max(rows, key=lambda row: fnum(row.get("learned_score"), -1e9))
    return out


def prune_by_frame(
    by_frame: dict[int, list[dict[str, Any]]],
    beam: int,
) -> dict[int, list[dict[str, Any]]]:
    if beam <= 0:
        return by_frame
    return {
        frame: sorted(rows, key=lambda row: fnum(row.get("learned_score"), -1e9), reverse=True)[:beam]
        for frame, rows in by_frame.items()
    }


def select_rows(
    by_frame: dict[int, list[dict[str, Any]]],
    sequence_window: int,
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float,
    sequence_beam: int,
) -> dict[int, dict[str, Any]]:
    by_frame = prune_by_frame(by_frame, sequence_beam)
    if sequence_window == 1:
        return framewise_best(by_frame)
    if sequence_window <= 0:
        return seq.viterbi_select(by_frame, max_jump_px, transition_weight, size_jump_weight)
    return seq.rolling_viterbi_select(
        by_frame,
        max_jump_px=max_jump_px,
        transition_weight=transition_weight,
        size_jump_weight=size_jump_weight,
        sequence_window=sequence_window,
    )


def apply_hysteresis_gate(
    selected: dict[int, dict[str, Any]],
    acquire_threshold: float,
    acquire_hits: int,
    keep_threshold: float,
    max_jump_px: float,
    lost_patience: int,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    active = False
    pending_hits = 0
    pending_frame: int | None = None
    pending_row: dict[str, Any] | None = None
    lost = 0
    last_frame: int | None = None
    last_row: dict[str, Any] | None = None
    for frame in sorted(selected):
        row = selected[frame]
        row_score = fnum(row.get("learned_score"), 0.0)
        emit = False
        if not active:
            if row_score >= acquire_threshold:
                gap = max(1, frame - pending_frame) if pending_frame is not None else 1
                jump_ok = (
                    pending_row is None
                    or center_dist(bbox(pending_row), bbox(row)) <= max_jump_px * gap
                )
                pending_hits = pending_hits + 1 if jump_ok else 1
                pending_frame = frame
                pending_row = row
                if pending_hits >= max(1, acquire_hits):
                    active = True
                    emit = True
                    lost = 0
            else:
                pending_hits = 0
                pending_frame = None
                pending_row = None
        else:
            gap = max(1, frame - last_frame) if last_frame is not None else 1
            jump_ok = last_row is None or center_dist(bbox(last_row), bbox(row)) <= max_jump_px * gap
            if row_score >= keep_threshold and jump_ok:
                emit = True
                lost = 0
            else:
                lost += 1
                if lost > max(0, lost_patience):
                    active = False
                    pending_hits = 0
                    pending_frame = None
                    pending_row = None
                    last_frame = None
                    last_row = None
        if emit:
            out[frame] = row
            last_frame = frame
            last_row = row
    return out


def evaluate_selection(
    labels: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    threshold: float,
    strict_tol_px: float,
    loose_tol_px: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for lab in labels:
        frame = int(lab["frame"])
        cand = selected.get(frame)
        if cand is not None and fnum(cand.get("learned_score"), -1e9) < threshold:
            cand = None
        dist = None
        strict = False
        loose = False
        if lab["visible"] and cand is not None and lab["bbox"] is not None:
            dist = center_dist(bbox(cand), lab["bbox"])
            strict = dist <= strict_tol_px
            loose = dist <= loose_tol_px
        correct = bool(strict) if lab["visible"] else cand is None
        rows.append(
            {
                "frame": frame,
                "visible": int(lab["visible"]),
                "selected": int(cand is not None),
                "rank": "" if cand is None else cand.get("rank", ""),
                "learned_score": "" if cand is None else round(fnum(cand.get("learned_score")), 6),
                "base_learned_score": "" if cand is None else round(fnum(cand.get("base_learned_score")), 6),
                "dist_px": "" if dist is None else round(dist, 3),
                "strict_hit": int(strict),
                "loose_hit": int(loose),
                "correct_all_frame": int(correct),
                "x": "" if cand is None else cand.get("x", ""),
                "y": "" if cand is None else cand.get("y", ""),
                "w": "" if cand is None else cand.get("w", ""),
                "h": "" if cand is None else cand.get("h", ""),
            }
        )
    visible = [r for r in rows if r["visible"]]
    invisible = [r for r in rows if not r["visible"]]
    selected_rows = [r for r in rows if r["selected"]]
    strict_hits = sum(int(r["strict_hit"]) for r in visible)
    loose_hits = sum(int(r["loose_hit"]) for r in visible)
    invisible_no_box = sum(1 for r in invisible if not r["selected"])
    correct = sum(int(r["correct_all_frame"]) for r in rows)
    return (
        {
            "frames_all": len(rows),
            "visible_frames": len(visible),
            "invisible_frames": len(invisible),
            "all_frame_correct": correct,
            "all_frame_accuracy": round(correct / max(1, len(rows)), 4),
            "visible_strict": strict_hits,
            "visible_strict_recall": round(strict_hits / max(1, len(visible)), 4),
            "visible_loose": loose_hits,
            "visible_loose_recall": round(loose_hits / max(1, len(visible)), 4),
            "invisible_no_box": invisible_no_box,
            "invisible_no_box_rate": round(invisible_no_box / max(1, len(invisible)), 4),
            "selected_frames": len(selected_rows),
        },
        rows,
    )


def weight_grid(args: argparse.Namespace) -> list[clba_adjust.Weights]:
    return [
        clba_adjust.Weights(*vals)
        for vals in product(
            parse_float_list(args.gain_weights),
            parse_float_list(args.path_weights),
            parse_float_list(args.target_q_weights),
            parse_float_list(args.bg_weights),
            parse_float_list(args.attached_weights),
            parse_float_list(args.density_weights),
        )
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels), args.clip)
    candidates = load_candidates(Path(args.candidates), args.clip, args.max_rank)
    if not labels:
        raise SystemExit("no labels loaded")
    if not candidates:
        raise SystemExit("no candidates loaded")

    summaries: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    for weights in weight_grid(args):
        scored = score_rows(candidates, weights, args.score_column)
        by_frame = group_by_frame(scored)
        for window in parse_int_list(args.sequence_windows):
            for max_jump in parse_float_list(args.max_jump_px):
                for transition in parse_float_list(args.transition_weights):
                    for size_weight in parse_float_list(args.size_jump_weights):
                        selected = select_rows(by_frame, window, max_jump, transition, size_weight, args.sequence_beam)
                        for threshold in parse_float_list(args.thresholds):
                            summary, rows = evaluate_selection(
                                labels,
                                selected,
                                threshold,
                                args.strict_tol_px,
                                args.loose_tol_px,
                            )
                            summary.update(
                                {
                                    "sequence_window": window,
                                    "max_jump_px": max_jump,
                                    "transition_weight": transition,
                                    "size_jump_weight": size_weight,
                                    "threshold": threshold,
                                    "sequence_beam": args.sequence_beam,
                                    **asdict(weights),
                                }
                            )
                            summaries.append(summary)
                            sort_key = (
                                summary["all_frame_accuracy"],
                                summary["visible_strict_recall"],
                                summary["invisible_no_box_rate"],
                                -summary["selected_frames"],
                            )
                            best_key = (
                                -1.0,
                                -1.0,
                                -1.0,
                                0,
                            )
                            if best_summary is not None:
                                best_key = (
                                    best_summary["all_frame_accuracy"],
                                    best_summary["visible_strict_recall"],
                                    best_summary["invisible_no_box_rate"],
                                    -best_summary["selected_frames"],
                                )
                            if best_summary is None or sort_key > best_key:
                                best_summary = summary
                                best_rows = rows
                        if args.acquire_thresholds:
                            keep_thresholds = args.keep_thresholds or args.thresholds
                            for acquire in parse_float_list(args.acquire_thresholds):
                                for hits in parse_int_list(args.acquire_hits):
                                    for keep in parse_float_list(keep_thresholds):
                                        for patience in parse_int_list(args.lost_patience):
                                            gated = apply_hysteresis_gate(selected, acquire, hits, keep, max_jump, patience)
                                            summary, rows = evaluate_selection(
                                                labels,
                                                gated,
                                                threshold=0.0,
                                                strict_tol_px=args.strict_tol_px,
                                                loose_tol_px=args.loose_tol_px,
                                            )
                                            summary.update(
                                                {
                                                    "sequence_window": window,
                                                    "max_jump_px": max_jump,
                                                    "transition_weight": transition,
                                                    "size_jump_weight": size_weight,
                                                    "threshold": 0.0,
                                                    "sequence_beam": args.sequence_beam,
                                                    "acquire_threshold": acquire,
                                                    "acquire_hits": hits,
                                                    "keep_threshold": keep,
                                                    "lost_patience": patience,
                                                    **asdict(weights),
                                                }
                                            )
                                            summaries.append(summary)
                                            sort_key = (
                                                summary["all_frame_accuracy"],
                                                summary["visible_strict_recall"],
                                                summary["invisible_no_box_rate"],
                                                -summary["selected_frames"],
                                            )
                                            best_key = (
                                                -1.0,
                                                -1.0,
                                                -1.0,
                                                0,
                                            )
                                            if best_summary is not None:
                                                best_key = (
                                                    best_summary["all_frame_accuracy"],
                                                    best_summary["visible_strict_recall"],
                                                    best_summary["invisible_no_box_rate"],
                                                    -best_summary["selected_frames"],
                                                )
                                            if best_summary is None or sort_key > best_key:
                                                best_summary = summary
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
    write_csv(out_dir / "clba_sequence_sweep.csv", summaries)
    write_csv(out_dir / "best_sequence_predictions.csv", best_rows)
    (out_dir / "best_config.json").write_text(json.dumps(summaries[0] if summaries else {}, indent=2))
    (out_dir / "README.md").write_text(
        "# CLBA Sequence Selector Sweep\n\n"
        "This artifact evaluates out-of-fold candidate scores with optional CLBA\n"
        "target/background score modifiers under framewise, rolling-window, and\n"
        "full-window Viterbi selection. It is offline only.\n"
    )
    print(out_dir / "clba_sequence_sweep.csv")
    if summaries:
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
