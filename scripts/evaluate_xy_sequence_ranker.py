#!/usr/bin/env python3
"""Evaluate learned XY candidate scores with a continuity/Viterbi selector.

This is an offline harness for dense manual/vision XY labels. It answers a
specific question: when the correct candidate is present in top_tubes, can a
short-window continuity prior stop framewise score jumps to terrain clutter?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

try:
    import train_xy_tube_ranker as xy_ranker
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import train_xy_tube_ranker as xy_ranker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", choices=("logistic", "hist_gbdt", "extra_trees"), default="logistic")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--center_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--negative_min_dist_px", type=float, default=24.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max_jump_px", default="10,12,16,20,24,32")
    p.add_argument("--transition_weight", default="0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    p.add_argument("--random_state", type=int, default=11)
    return p.parse_args()


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


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("x", 0.0) or 0.0),
        float(row.get("y", 0.0) or 0.0),
        float(row.get("w", 1.0) or 1.0),
        float(row.get("h", 1.0) or 1.0),
    )


def label_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row.get("det_x", row.get("x", 0.0)) or 0.0),
        float(row.get("det_y", row.get("y", 0.0)) or 0.0),
        float(row.get("det_w", row.get("w", 1.0)) or 1.0),
        float(row.get("det_h", row.get("h", 1.0)) or 1.0),
    )


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + 0.5 * w, y + 0.5 * h


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def candidate_dist(row: dict[str, Any], lab: dict[str, str]) -> float:
    return center_dist(bbox(row), label_bbox(lab))


def contiguous_folds(frames: list[int], n_folds: int) -> dict[int, int]:
    chunks = np.array_split(np.asarray(sorted(frames), dtype=int), max(2, n_folds))
    out: dict[int, int] = {}
    for fold, chunk in enumerate(chunks):
        for frame in chunk:
            out[int(frame)] = int(fold)
    return out


def make_oof_scores(
    labels: list[dict[str, str]],
    top_by_frame: dict[int, list[dict[str, str]]],
    model_name: str,
    center_tol: float,
    negative_min_dist: float,
    folds: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    feature_rows: list[dict[str, str]] = []
    ignored_near = 0
    for lab in labels:
        frame = int(lab["frame"])
        for row in top_by_frame.get(frame, []):
            feature_rows.append(row)
            dist = candidate_dist(row, lab)
            if dist <= center_tol:
                y = 1
            elif dist >= negative_min_dist:
                y = 0
            else:
                ignored_near += 1
                continue
            examples.append({"frame": frame, "row": row, "y": y, "dist_px": dist})
    if not examples:
        raise SystemExit("no training examples")

    numeric, sources = xy_ranker.infer_features([e["row"] for e in examples])
    x = xy_ranker.vectorize([e["row"] for e in examples], numeric, sources)
    y = np.asarray([int(e["y"]) for e in examples], dtype=np.int32)
    ex_frames = np.asarray([int(e["frame"]) for e in examples], dtype=np.int32)
    frame_to_fold = contiguous_folds([int(r["frame"]) for r in labels], folds)
    models = xy_ranker.make_models(random_state)
    model_spec = models[model_name]

    scored_rows: list[dict[str, Any]] = []
    for fold in sorted(set(frame_to_fold.values())):
        train_idx = np.asarray([i for i, frame in enumerate(ex_frames) if frame_to_fold.get(int(frame)) != fold], dtype=int)
        test_frames = [f for f, ff in frame_to_fold.items() if ff == fold]
        if len(train_idx) == 0 or len(set(y[train_idx])) < 2:
            continue
        model = Pipeline(model_spec.steps)
        model.fit(x[train_idx], y[train_idx])
        for frame in sorted(test_frames):
            rows = top_by_frame.get(frame, [])
            if not rows:
                continue
            scores = xy_ranker.predict_score(model, xy_ranker.vectorize(rows, numeric, sources))
            for row, score in zip(rows, scores):
                out = dict(row)
                out["learned_score"] = float(score)
                out["fold"] = fold
                scored_rows.append(out)

    meta = {
        "examples": len(examples),
        "positive_examples": int(np.sum(y == 1)),
        "negative_examples": int(np.sum(y == 0)),
        "ignored_near_examples": ignored_near,
        "numeric_features": numeric,
        "source_features": sources,
    }
    return scored_rows, meta


def summarize_selection(
    labels: list[dict[str, str]],
    selected: dict[int, dict[str, Any]],
    center_tol: float,
    loose_tol: float,
    name: str,
) -> dict[str, Any]:
    rows = []
    for lab in labels:
        frame = int(lab["frame"])
        cand = selected.get(frame)
        if cand is None:
            rows.append({"strict": False, "loose": False})
            continue
        d = candidate_dist(cand, lab)
        rows.append({"strict": d <= center_tol, "loose": d <= loose_tol})
    strict = sum(bool(r["strict"]) for r in rows)
    loose = sum(bool(r["loose"]) for r in rows)
    return {
        "selector": name,
        "frames": len(labels),
        "strict_hit": strict,
        "strict_recall": round(strict / max(1, len(labels)), 4),
        "loose_hit": loose,
        "loose_recall": round(loose / max(1, len(labels)), 4),
    }


def framewise_best(scored_rows: list[dict[str, Any]], score_name: str = "learned_score") -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in scored_rows:
        frame = int(float(row.get("frame", 0) or 0))
        score = float(row.get(score_name, -1e9) or -1e9)
        if frame not in out or score > float(out[frame].get(score_name, -1e9) or -1e9):
            out[frame] = row
    return out


def viterbi_select(
    by_frame: dict[int, list[dict[str, Any]]],
    max_jump_px: float,
    transition_weight: float,
) -> dict[int, dict[str, Any]]:
    frames = sorted(by_frame)
    if not frames:
        return {}
    prev_scores: list[float] = []
    backptrs: list[list[int | None]] = []
    layers: list[list[dict[str, Any]]] = []

    for fi, frame in enumerate(frames):
        rows = by_frame[frame]
        layers.append(rows)
        current: list[float] = []
        current_bp: list[int | None] = []
        if fi == 0:
            for row in rows:
                current.append(float(row.get("learned_score", 0.0) or 0.0))
                current_bp.append(None)
        else:
            prev_rows = layers[fi - 1]
            gap = max(1, frame - frames[fi - 1])
            allowed = max_jump_px * gap
            for row in rows:
                best_score = -1e18
                best_idx: int | None = None
                rb = bbox(row)
                for pi, prev in enumerate(prev_rows):
                    jump = center_dist(rb, bbox(prev))
                    if jump > allowed:
                        continue
                    cost = transition_weight * (jump / max(1e-6, allowed)) ** 2
                    cand_score = prev_scores[pi] + float(row.get("learned_score", 0.0) or 0.0) - cost
                    if cand_score > best_score:
                        best_score = cand_score
                        best_idx = pi
                if best_idx is None:
                    best_score = float(row.get("learned_score", 0.0) or 0.0) - 1.0
                current.append(best_score)
                current_bp.append(best_idx)
        prev_scores = current
        backptrs.append(current_bp)

    best_last = int(np.argmax(np.asarray(prev_scores)))
    selected_idx = [best_last]
    for fi in range(len(frames) - 1, 0, -1):
        prev = backptrs[fi][selected_idx[-1]]
        if prev is None:
            prev = int(np.argmax(np.asarray([float(r.get("learned_score", 0.0) or 0.0) for r in layers[fi - 1]])))
        selected_idx.append(prev)
    selected_idx.reverse()
    return {frame: layers[fi][idx] for fi, (frame, idx) in enumerate(zip(frames, selected_idx))}


def prediction_rows(labels: list[dict[str, str]], selected: dict[int, dict[str, Any]], center_tol: float, loose_tol: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lab in labels:
        frame = int(lab["frame"])
        cand = selected.get(frame)
        if cand is None:
            out.append({"frame": frame, "selected": 0})
            continue
        d = candidate_dist(cand, lab)
        out.append(
            {
                "frame": frame,
                "selected": 1,
                "rank": cand.get("rank", ""),
                "learned_score": round(float(cand.get("learned_score", 0.0) or 0.0), 6),
                "dist_px": round(d, 3),
                "strict_hit": d <= center_tol,
                "loose_hit": d <= loose_tol,
                "x": cand.get("x", ""),
                "y": cand.get("y", ""),
                "w": cand.get("w", ""),
                "h": cand.get("h", ""),
                "source": cand.get("cand_source", ""),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r for r in xy_ranker.read_csv(Path(args.labels)) if r.get("clip") == args.clip and r.get("visible", "1") != "0"]
    labels.sort(key=lambda r: int(r["frame"]))
    top_by_frame = xy_ranker.load_top_tubes(Path(args.results_dir), args.clip, args.max_rank)
    scored_rows, meta = make_oof_scores(
        labels,
        top_by_frame,
        args.model,
        args.center_tol_px,
        args.negative_min_dist_px,
        args.folds,
        args.random_state,
    )
    if not scored_rows:
        raise SystemExit("no scored rows")

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in scored_rows:
        by_frame.setdefault(int(float(row.get("frame", 0) or 0)), []).append(row)

    summaries: list[dict[str, Any]] = []
    baseline_selected = framewise_best(scored_rows)
    base = summarize_selection(labels, baseline_selected, args.center_tol_px, args.loose_tol_px, "framewise_learned")
    summaries.append(base)

    best_rows = prediction_rows(labels, baseline_selected, args.center_tol_px, args.loose_tol_px)
    best_summary = base
    for max_jump in parse_float_list(args.max_jump_px):
        for weight in parse_float_list(args.transition_weight):
            selected = viterbi_select(by_frame, max_jump, weight)
            summary = summarize_selection(
                labels,
                selected,
                args.center_tol_px,
                args.loose_tol_px,
                f"viterbi_jump{max_jump:g}_w{weight:g}",
            )
            summary["max_jump_px"] = max_jump
            summary["transition_weight"] = weight
            summaries.append(summary)
            if (summary["strict_recall"], summary["loose_recall"]) > (
                best_summary["strict_recall"],
                best_summary["loose_recall"],
            ):
                best_summary = summary
                best_rows = prediction_rows(labels, selected, args.center_tol_px, args.loose_tol_px)

    summaries.sort(key=lambda r: (r["strict_recall"], r["loose_recall"]), reverse=True)
    write_csv(out_dir / "sequence_summary.csv", summaries)
    write_csv(out_dir / "best_sequence_predictions.csv", best_rows)
    write_csv(out_dir / "oof_scored_candidates.csv", scored_rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "labels": args.labels,
                "results_dir": args.results_dir,
                "clip": args.clip,
                "model": args.model,
                "frames": len(labels),
                "best_summary": summaries[0] if summaries else {},
                **meta,
            },
            indent=2,
        )
    )
    print(out_dir / "sequence_summary.csv")
    if summaries:
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
