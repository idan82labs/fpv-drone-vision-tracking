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
    p.add_argument("--size_jump_weight", default="0", help="Optional comma-separated box-size allowance added to max_jump_px.")
    p.add_argument("--cv_accel_weight", default="", help="Optional comma-separated acceleration weights for second-order CV Viterbi.")
    p.add_argument("--sequence_beam", type=int, default=50)
    p.add_argument("--state_beam", type=int, default=512)
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
    size_jump_weight: float = 0.0,
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
            for row in rows:
                best_score = -1e18
                best_idx: int | None = None
                rb = bbox(row)
                for pi, prev in enumerate(prev_rows):
                    pb = bbox(prev)
                    size_allowance = size_jump_weight * max(rb[2], rb[3], pb[2], pb[3])
                    allowed = max_jump_px * gap + size_allowance
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


def rolling_viterbi_select(
    by_frame: dict[int, list[dict[str, Any]]],
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float = 0.0,
    sequence_window: int = 60,
) -> dict[int, dict[str, Any]]:
    """Run Viterbi in a sliding window and emit only each window's newest frame.

    The full-video selector forces one path across the whole clip. That is useful
    for short reviewed snippets, but it fails when a false background lock exists
    before target birth: a later real target far away cannot connect to the old
    branch/terrain path. A rolling window keeps the continuity prior local, so
    the selector can restart after enough local evidence accumulates.
    """

    frames = sorted(by_frame)
    if not frames:
        return {}
    window = max(1, int(sequence_window))
    selected: dict[int, dict[str, Any]] = {}
    for idx, frame in enumerate(frames):
        start = max(0, idx - window + 1)
        local_frames = frames[start : idx + 1]
        local_by_frame = {f: by_frame[f] for f in local_frames}
        local_selected = viterbi_select(local_by_frame, max_jump_px, transition_weight, size_jump_weight)
        if frame in local_selected:
            selected[frame] = local_selected[frame]
    return selected


def row_score(row: dict[str, Any], score_name: str = "learned_score") -> float:
    return float(row.get(score_name, 0.0) or 0.0)


def prune_frame_rows(
    by_frame: dict[int, list[dict[str, Any]]],
    beam: int,
    score_name: str = "learned_score",
) -> dict[int, list[dict[str, Any]]]:
    if beam <= 0:
        return by_frame
    return {
        frame: sorted(rows, key=lambda row: row_score(row, score_name), reverse=True)[:beam]
        for frame, rows in by_frame.items()
    }


def viterbi_select_constant_velocity(
    by_frame: dict[int, list[dict[str, Any]]],
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float,
    accel_weight: float,
    state_beam: int,
) -> dict[int, dict[str, Any]]:
    """Second-order Viterbi selector with a constant-velocity prior.

    The first-order selector only penalizes large per-frame displacement. That
    still allows a smooth switch to a nearby terrain branch. This selector keeps
    pair states `(t-1, t)` and penalizes acceleration away from the previous
    velocity, while still using a max-jump guard.
    """

    frames = sorted(by_frame)
    if len(frames) <= 2:
        return viterbi_select(by_frame, max_jump_px, transition_weight, size_jump_weight)

    layers = [by_frame[frame] for frame in frames]
    if any(not layer for layer in layers):
        return viterbi_select(by_frame, max_jump_px, transition_weight, size_jump_weight)

    def frame_gap(fi: int) -> int:
        if fi <= 0:
            return 1
        return max(1, frames[fi] - frames[fi - 1])

    def transition_cost(prev: dict[str, Any], cur: dict[str, Any], base_allowed: float) -> tuple[float, float] | None:
        pb = bbox(prev)
        cb = bbox(cur)
        allowed = base_allowed + size_jump_weight * max(pb[2], pb[3], cb[2], cb[3])
        jump = center_dist(bbox(prev), bbox(cur))
        if jump > allowed:
            return None
        return transition_weight * (jump / max(1e-6, allowed)) ** 2, allowed

    # State at frame index 1 is (idx_at_0, idx_at_1).
    states: dict[tuple[int, int], float] = {}
    first_allowed = max_jump_px * frame_gap(1)
    for i, prev in enumerate(layers[0]):
        for j, cur in enumerate(layers[1]):
            transition = transition_cost(prev, cur, first_allowed)
            if transition is None:
                continue
            cost, _allowed = transition
            states[(i, j)] = row_score(prev) + row_score(cur) - cost
    if not states:
        return viterbi_select(by_frame, max_jump_px, transition_weight, size_jump_weight)
    if state_beam > 0 and len(states) > state_beam:
        states = dict(sorted(states.items(), key=lambda item: item[1], reverse=True)[:state_beam])

    backptrs: list[dict[tuple[int, int], tuple[int, int]]] = [{} for _ in frames]
    for fi in range(2, len(frames)):
        cur_rows = layers[fi]
        prev_rows = layers[fi - 1]
        prev_prev_rows = layers[fi - 2]
        allowed = max_jump_px * frame_gap(fi)
        cur_gap = frame_gap(fi)
        prev_gap = frame_gap(fi - 1)
        new_states: dict[tuple[int, int], float] = {}
        new_backptrs: dict[tuple[int, int], tuple[int, int]] = {}
        for (h, i), prev_score in states.items():
            ppx, ppy = center(bbox(prev_prev_rows[h]))
            px, py = center(bbox(prev_rows[i]))
            pred = (px + (px - ppx) * (cur_gap / prev_gap), py + (py - ppy) * (cur_gap / prev_gap))
            for j, cur in enumerate(cur_rows):
                transition = transition_cost(prev_rows[i], cur, allowed)
                if transition is None:
                    continue
                jump_cost, allowed_with_size = transition
                cx, cy = center(bbox(cur))
                accel = math.hypot(cx - pred[0], cy - pred[1])
                accel_cost = accel_weight * (accel / max(1e-6, allowed_with_size)) ** 2
                score = prev_score + row_score(cur) - jump_cost - accel_cost
                state = (i, j)
                if state not in new_states or score > new_states[state]:
                    new_states[state] = score
                    new_backptrs[state] = (h, i)
        if not new_states:
            return viterbi_select(by_frame, max_jump_px, transition_weight, size_jump_weight)
        if state_beam > 0 and len(new_states) > state_beam:
            keep = {
                state
                for state, _ in sorted(new_states.items(), key=lambda item: item[1], reverse=True)[:state_beam]
            }
            new_states = {state: score for state, score in new_states.items() if state in keep}
            new_backptrs = {state: prev for state, prev in new_backptrs.items() if state in keep}
        states = new_states
        backptrs[fi] = new_backptrs

    state = max(states, key=lambda key: states[key])
    selected_idx: list[int | None] = [None] * len(frames)
    selected_idx[-2], selected_idx[-1] = state
    for fi in range(len(frames) - 1, 1, -1):
        prev_state = backptrs[fi].get(state)
        if prev_state is None:
            break
        selected_idx[fi - 2] = prev_state[0]
        state = prev_state

    if any(idx is None for idx in selected_idx):
        return viterbi_select(by_frame, max_jump_px, transition_weight, size_jump_weight)
    return {frame: layers[fi][int(idx)] for fi, (frame, idx) in enumerate(zip(frames, selected_idx))}


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
            for size_weight in parse_float_list(args.size_jump_weight):
                selected = viterbi_select(by_frame, max_jump, weight, size_weight)
                summary = summarize_selection(
                    labels,
                    selected,
                    args.center_tol_px,
                    args.loose_tol_px,
                    f"viterbi_jump{max_jump:g}_w{weight:g}_sz{size_weight:g}",
                )
                summary["max_jump_px"] = max_jump
                summary["transition_weight"] = weight
                summary["size_jump_weight"] = size_weight
                summaries.append(summary)
                if (summary["strict_recall"], summary["loose_recall"]) > (
                    best_summary["strict_recall"],
                    best_summary["loose_recall"],
                ):
                    best_summary = summary
                    best_rows = prediction_rows(labels, selected, args.center_tol_px, args.loose_tol_px)
                cv_by_frame = prune_frame_rows(by_frame, args.sequence_beam)
                for accel_weight in parse_float_list(args.cv_accel_weight):
                    selected = viterbi_select_constant_velocity(
                        cv_by_frame,
                        max_jump,
                        weight,
                        size_weight,
                        accel_weight,
                        args.state_beam,
                    )
                    summary = summarize_selection(
                        labels,
                        selected,
                        args.center_tol_px,
                        args.loose_tol_px,
                        f"cv_viterbi_jump{max_jump:g}_w{weight:g}_sz{size_weight:g}_a{accel_weight:g}",
                    )
                    summary["max_jump_px"] = max_jump
                    summary["transition_weight"] = weight
                    summary["size_jump_weight"] = size_weight
                    summary["accel_weight"] = accel_weight
                    summary["sequence_beam"] = args.sequence_beam
                    summary["state_beam"] = args.state_beam
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
