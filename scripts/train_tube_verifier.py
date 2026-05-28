#!/usr/bin/env python3
"""Train/evaluate a tiny linear verifier on exported TBD top-tube alternatives.

This is deliberately small and dependency-free. It turns sparse checkpoint
labels into hard positive/negative tube examples, trains a ridge-regularized
logistic model, and evaluates leave-one-clip-out selection on the reviewed
frames.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXCLUDE_COLUMNS = {
    "frame",
    "rank",
    "track_id",
    "x",
    "y",
    "verified_score",
    "tube_verifier_score",
    "eligible",
    "passes_floor",
    "selected",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--center_tol_px", type=float, default=14.0)
    p.add_argument("--iou_tol", type=float, default=0.10)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--epochs", type=int, default=900)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--l2", type=float, default=0.18)
    p.add_argument("--threshold_metric", choices=("balanced", "precision"), default="balanced")
    p.add_argument("--train_mode", choices=("binary", "pairwise"), default="binary")
    p.add_argument("--hard_negatives_per_positive", type=int, default=12)
    p.add_argument("--tube_labels", default=None, help="Optional tube_alternatives_to_label.csv with human_label values")
    return p.parse_args()


def bbox_from_text(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    try:
        vals = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
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


def is_match(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
    center_tol_px: float,
    iou_tol: float,
) -> bool:
    if a is None or b is None:
        return False
    return center_dist(a, b) <= center_tol_px or iou(a, b) >= iou_tol


def classify(
    label: str,
    notes: str,
    reviewed_bbox: tuple[int, int, int, int] | None,
    selected: tuple[int, int, int, int] | None,
    center_tol_px: float,
    iou_tol: float,
) -> str:
    label = label.strip().lower()
    notes_l = notes.lower()
    no_object_note = (
        "no object" in notes_l
        or "no isolated object" in notes_l
        or "no clear object" in notes_l
        or "no selected box and no clear object" in notes_l
    )
    if label == "gold":
        if selected is None:
            return "gold_miss"
        if is_match(reviewed_bbox, selected, center_tol_px, iou_tol):
            return "gold_hit"
        return "gold_wrong_box"
    if label == "empty":
        return "empty_tn" if selected is None else "empty_fp"
    if label == "clutter":
        if selected is None:
            return "clutter_suppressed"
        if is_match(reviewed_bbox, selected, center_tol_px, iou_tol):
            return "clutter_repeated"
        return "clutter_selected_elsewhere_fp" if no_object_note else "clutter_selected_elsewhere_unknown"
    if label == "miss":
        return "miss_no_select" if selected is None else "miss_selected_unknown"
    return "unknown_label"


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def load_labels(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_top_tubes(results_dir: Path, clip: str) -> dict[int, list[dict[str, str]]]:
    path = results_dir / clip / "top_tubes.csv"
    if not path.exists():
        return {}
    by_frame: dict[int, list[dict[str, str]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            frame = int(float(row.get("frame", "0") or 0))
            by_frame.setdefault(frame, []).append(row)
    return by_frame


def row_bbox(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        int(round(float(row.get("x", "0") or 0))),
        int(round(float(row.get("y", "0") or 0))),
        int(round(float(row.get("w", "1") or 1))),
        int(round(float(row.get("h", "1") or 1))),
    )


def infer_feature_names(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    sources: set[str] = set()
    for rec in rows:
        row = rec["row"]
        src = row.get("cand_source", "")
        if src:
            sources.add(src)
        for key, value in row.items():
            if key in EXCLUDE_COLUMNS or key == "cand_source":
                continue
            if safe_float(value) is not None and key not in numeric:
                numeric.append(key)
    source_names = [f"src_{s}" for s in sorted(sources)]
    return numeric, source_names


def feature_vector(row: dict[str, str], numeric_names: list[str], source_names: list[str]) -> list[float]:
    vals = [safe_float(row.get(name)) or 0.0 for name in numeric_names]
    src = row.get("cand_source", "")
    vals.extend(1.0 if name == f"src_{src}" else 0.0 for name in source_names)
    return vals


def make_examples(
    labels: list[dict[str, str]],
    results_dir: Path,
    center_tol_px: float,
    iou_tol: float,
    max_rank: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    frame_cases: list[dict[str, Any]] = []
    cache: dict[str, dict[int, list[dict[str, str]]]] = {}
    for label_row in labels:
        clip = label_row.get("clip", "")
        frame = int(label_row.get("frame", "0") or 0)
        label = label_row.get("label", "").strip().lower()
        reviewed_bbox = bbox_from_text(label_row.get("selected_bbox"))
        if clip not in cache:
            cache[clip] = load_top_tubes(results_dir, clip)
        rows = [
            r for r in cache[clip].get(frame, [])
            if int(float(r.get("rank", "999") or 999)) <= max_rank
            and int(float(r.get("eligible", "1") or 1)) == 1
        ]
        frame_cases.append(
            {
                "clip": clip,
                "frame": frame,
                "label": label,
                "notes": label_row.get("notes", ""),
                "reviewed_bbox": reviewed_bbox,
                "rows": rows,
            }
        )
        if label == "miss":
            continue
        for row in rows:
            bbox = row_bbox(row)
            y: int | None = None
            kind = ""
            if label == "gold":
                if is_match(reviewed_bbox, bbox, center_tol_px, iou_tol):
                    y = 1
                    kind = "gold_match"
                else:
                    y = 0
                    kind = "gold_competitor"
            elif label in {"clutter", "empty"}:
                y = 0
                kind = label
            if y is None:
                continue
            examples.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "label": label,
                    "y": y,
                    "kind": kind,
                    "bbox": bbox,
                    "row": row,
                    "notes": label_row.get("notes", ""),
                }
            )
    return examples, frame_cases


def make_human_examples(
    tube_labels_path: Path,
    labels: list[dict[str, str]],
    results_dir: Path,
    max_rank: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_lookup = {
        (row.get("clip", ""), int(row.get("frame", "0") or 0)): row
        for row in labels
    }
    cache: dict[str, dict[int, list[dict[str, str]]]] = {}
    examples: list[dict[str, Any]] = []
    with tube_labels_path.open() as f:
        for review_row in csv.DictReader(f):
            human_label = review_row.get("human_label", "").strip().lower()
            if not human_label or human_label == "uncertain":
                continue
            clip = review_row.get("clip", "")
            frame = int(review_row.get("frame", "0") or 0)
            rank = int(float(review_row.get("rank", "999") or 999))
            if rank > max_rank:
                continue
            if clip not in cache:
                cache[clip] = load_top_tubes(results_dir, clip)
            top_row = None
            for row in cache[clip].get(frame, []):
                if int(float(row.get("rank", "999") or 999)) == rank:
                    top_row = row
                    break
            if top_row is None:
                continue
            y = 1 if human_label == "target" else 0
            label_row = label_lookup.get((clip, frame), {})
            examples.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "label": label_row.get("label", review_row.get("checkpoint_label", "")),
                    "y": y,
                    "kind": human_label,
                    "bbox": row_bbox(top_row),
                    "row": top_row,
                    "notes": review_row.get("human_notes") or review_row.get("notes", ""),
                }
            )

    frame_cases: list[dict[str, Any]] = []
    for label_row in labels:
        clip = label_row.get("clip", "")
        frame = int(label_row.get("frame", "0") or 0)
        if clip not in cache:
            cache[clip] = load_top_tubes(results_dir, clip)
        rows = [
            r for r in cache[clip].get(frame, [])
            if int(float(r.get("rank", "999") or 999)) <= max_rank
            and int(float(r.get("eligible", "1") or 1)) == 1
        ]
        frame_cases.append(
            {
                "clip": clip,
                "frame": frame,
                "label": label_row.get("label", "").strip().lower(),
                "notes": label_row.get("notes", ""),
                "reviewed_bbox": bbox_from_text(label_row.get("selected_bbox")),
                "rows": rows,
            }
        )
    return examples, frame_cases


def standardize(x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x[idx].mean(axis=0)
    std = x[idx].std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean, std = standardize(x, train_idx)
    xt = (x[train_idx] - mean) / std
    yt = y[train_idx].astype(np.float64)
    n_features = xt.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    n_pos = max(1.0, float(np.sum(yt == 1)))
    n_neg = max(1.0, float(np.sum(yt == 0)))
    weights = np.where(yt == 1, 0.5 / n_pos, 0.5 / n_neg)
    weights = weights / weights.mean()
    for _ in range(epochs):
        z = np.clip(xt @ w + b, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - yt) * weights
        grad_w = (xt.T @ err) / len(yt) + l2 * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, mean, std


def train_pairwise_ranker(
    x: np.ndarray,
    examples: list[dict[str, Any]],
    train_idx: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    hard_negatives_per_positive: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean, std = standardize(x, train_idx)
    groups: dict[tuple[str, int], dict[str, list[int]]] = {}
    train_set = set(int(i) for i in train_idx)
    for i, ex in enumerate(examples):
        if i not in train_set or ex["label"] != "gold":
            continue
        key = (ex["clip"], int(ex["frame"]))
        groups.setdefault(key, {"pos": [], "neg": []})
        groups[key]["pos" if int(ex["y"]) == 1 else "neg"].append(i)

    diffs: list[np.ndarray] = []
    for group in groups.values():
        if not group["pos"] or not group["neg"]:
            continue
        negs = sorted(
            group["neg"],
            key=lambda i: safe_float(examples[i]["row"].get("verified_score")) or -999.0,
            reverse=True,
        )[:hard_negatives_per_positive]
        for pi in group["pos"]:
            xp = (x[pi] - mean) / std
            for ni in negs:
                xn = (x[ni] - mean) / std
                diffs.append(xp - xn)

    if not diffs:
        return train_logistic(x, np.array([int(e["y"]) for e in examples], dtype=np.int32), train_idx, epochs, lr, l2)

    xd = np.vstack(diffs).astype(np.float64)
    w = np.zeros(xd.shape[1], dtype=np.float64)
    for _ in range(epochs):
        z = np.clip(xd @ w, -40.0, 40.0)
        p_bad = 1.0 / (1.0 + np.exp(z))  # derivative of log(1+exp(-z))
        grad_w = -(xd.T @ p_bad) / len(xd) + l2 * w
        w -= lr * grad_w
    return w, 0.0, mean, std


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    examples: list[dict[str, Any]],
    train_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if args.train_mode == "pairwise":
        return train_pairwise_ranker(
            x,
            examples,
            train_idx,
            args.epochs,
            args.lr,
            args.l2,
            args.hard_negatives_per_positive,
        )
    return train_logistic(x, y, train_idx, args.epochs, args.lr, args.l2)


def scores_for_rows(
    rows: list[dict[str, str]],
    numeric_names: list[str],
    source_names: list[str],
    w: np.ndarray,
    b: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[tuple[float, dict[str, str]]]:
    if not rows:
        return []
    x = np.array([feature_vector(r, numeric_names, source_names) for r in rows], dtype=np.float64)
    xz = (x - mean) / std
    scores = xz @ w + b
    return [(float(s), r) for s, r in zip(scores, rows)]


def summarize_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        outcome = row["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
    gold_total = sum(1 for r in rows if r["label"] == "gold")
    clutter_total = sum(1 for r in rows if r["label"] == "clutter")
    empty_total = sum(1 for r in rows if r["label"] == "empty")
    known_fp = (
        counts.get("gold_wrong_box", 0)
        + counts.get("clutter_repeated", 0)
        + counts.get("clutter_selected_elsewhere_fp", 0)
        + counts.get("empty_fp", 0)
    )
    return {
        "gold_total": gold_total,
        "gold_hit": counts.get("gold_hit", 0),
        "gold_miss": counts.get("gold_miss", 0),
        "gold_wrong_box": counts.get("gold_wrong_box", 0),
        "gold_recall": round(counts.get("gold_hit", 0) / max(1, gold_total), 3),
        "clutter_total": clutter_total,
        "clutter_suppressed": counts.get("clutter_suppressed", 0),
        "clutter_repeated": counts.get("clutter_repeated", 0),
        "clutter_selected_elsewhere_fp": counts.get("clutter_selected_elsewhere_fp", 0),
        "clutter_selected_elsewhere_unknown": counts.get("clutter_selected_elsewhere_unknown", 0),
        "empty_total": empty_total,
        "empty_tn": counts.get("empty_tn", 0),
        "empty_fp": counts.get("empty_fp", 0),
        "known_fp": known_fp,
        "miss_selected_unknown": counts.get("miss_selected_unknown", 0),
        "miss_no_select": counts.get("miss_no_select", 0),
    }


def choose_threshold(
    cases: list[dict[str, Any]],
    numeric_names: list[str],
    source_names: list[str],
    w: np.ndarray,
    b: float,
    mean: np.ndarray,
    std: np.ndarray,
    metric: str,
) -> float:
    candidates = [-999.0, 999.0]
    for case in cases:
        for score, _ in scores_for_rows(case["rows"], numeric_names, source_names, w, b, mean, std):
            candidates.append(score)
    best_thr = 999.0
    best_value = -1e9
    for thr in sorted(set(candidates)):
        preds = evaluate_cases(cases, numeric_names, source_names, w, b, mean, std, thr)
        summary = summarize_outcomes(preds)
        if metric == "precision":
            value = 3.0 * summary["gold_hit"] - 5.0 * summary["known_fp"] - 0.5 * summary["gold_miss"]
        else:
            value = 2.0 * summary["gold_hit"] - 3.0 * summary["known_fp"] - 0.75 * summary["gold_miss"]
        if value > best_value:
            best_value = value
            best_thr = thr
    return float(best_thr)


def evaluate_cases(
    cases: list[dict[str, Any]],
    numeric_names: list[str],
    source_names: list[str],
    w: np.ndarray,
    b: float,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in cases:
        scored = scores_for_rows(case["rows"], numeric_names, source_names, w, b, mean, std)
        selected_bbox = None
        best_score = ""
        if scored:
            score, row = max(scored, key=lambda x: x[0])
            if score >= threshold:
                selected_bbox = row_bbox(row)
                best_score = round(score, 6)
        outcome = classify(
            case["label"],
            case["notes"],
            case["reviewed_bbox"],
            selected_bbox,
            14.0,
            0.10,
        )
        out.append(
            {
                "clip": case["clip"],
                "frame": case["frame"],
                "label": case["label"],
                "outcome": outcome,
                "reviewed_bbox": case["reviewed_bbox"],
                "selected_bbox": selected_bbox,
                "learned_score": best_score,
                "threshold": round(threshold, 6),
                "notes": case["notes"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels))
    if args.tube_labels:
        examples, cases = make_human_examples(Path(args.tube_labels), labels, Path(args.results_dir), args.max_rank)
        if not examples:
            raise SystemExit("no human-labeled tube examples found; fill human_label or omit --tube_labels")
    else:
        examples, cases = make_examples(
            labels,
            Path(args.results_dir),
            args.center_tol_px,
            args.iou_tol,
            args.max_rank,
        )
    if not examples:
        raise SystemExit("no training examples; did top_tubes.csv exports exist?")
    numeric_names, source_names = infer_feature_names(examples)
    x = np.array([feature_vector(e["row"], numeric_names, source_names) for e in examples], dtype=np.float64)
    y = np.array([int(e["y"]) for e in examples], dtype=np.int32)
    clips = np.array([e["clip"] for e in examples])
    unique_clips = sorted(set(clips))

    dataset_rows = [
        {
            "clip": e["clip"],
            "frame": e["frame"],
            "y": e["y"],
            "kind": e["kind"],
            "bbox": e["bbox"],
            "rank": e["row"].get("rank", ""),
            "score": e["row"].get("score", ""),
            "verified_score": e["row"].get("verified_score", ""),
            "tube_verifier_score": e["row"].get("tube_verifier_score", ""),
            "notes": e["notes"],
        }
        for e in examples
    ]
    write_csv(out_dir / "tube_training_examples.csv", dataset_rows)

    loco_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for holdout in unique_clips:
        train_idx = np.where(clips != holdout)[0]
        if len(set(y[train_idx])) < 2:
            continue
        w, b, mean, std = train_model(x, y, examples, train_idx, args)
        train_cases = [c for c in cases if c["clip"] != holdout and c["label"] != "miss"]
        test_cases = [c for c in cases if c["clip"] == holdout]
        threshold = choose_threshold(train_cases, numeric_names, source_names, w, b, mean, std, args.threshold_metric)
        preds = evaluate_cases(test_cases, numeric_names, source_names, w, b, mean, std, threshold)
        for pred in preds:
            pred["fold"] = holdout
        prediction_rows.extend(preds)
        summary = summarize_outcomes(preds)
        summary["fold"] = holdout
        summary["threshold"] = round(threshold, 6)
        summary["train_examples"] = int(len(train_idx))
        summary["train_pos"] = int(np.sum(y[train_idx] == 1))
        summary["train_neg"] = int(np.sum(y[train_idx] == 0))
        loco_rows.append(summary)

    overall = summarize_outcomes(prediction_rows)
    overall["fold"] = "leave_one_clip_out_total"
    overall["threshold"] = ""
    overall["train_examples"] = len(examples)
    overall["train_pos"] = int(np.sum(y == 1))
    overall["train_neg"] = int(np.sum(y == 0))
    loco_rows.append(overall)
    write_csv(out_dir / "loco_predictions.csv", prediction_rows)
    write_csv(out_dir / "loco_summary.csv", loco_rows)

    all_idx = np.arange(len(examples))
    w, b, mean, std = train_model(x, y, examples, all_idx, args)
    threshold = choose_threshold([c for c in cases if c["label"] != "miss"], numeric_names, source_names, w, b, mean, std, args.threshold_metric)
    all_preds = evaluate_cases(cases, numeric_names, source_names, w, b, mean, std, threshold)
    all_summary = summarize_outcomes(all_preds)
    all_summary["fold"] = "in_sample_all_clips"
    all_summary["threshold"] = round(threshold, 6)
    all_summary["train_examples"] = len(examples)
    all_summary["train_pos"] = int(np.sum(y == 1))
    all_summary["train_neg"] = int(np.sum(y == 0))
    write_csv(out_dir / "in_sample_predictions.csv", all_preds)
    write_csv(out_dir / "in_sample_summary.csv", [all_summary])

    feature_names = numeric_names + source_names
    weights = sorted(zip(feature_names, w), key=lambda kv: abs(kv[1]), reverse=True)
    write_csv(
        out_dir / "feature_weights.csv",
        [{"feature": name, "weight": round(float(weight), 8)} for name, weight in weights],
    )
    model = {
        "feature_names": feature_names,
        "weights": [float(v) for v in w],
        "bias": float(b),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "threshold": float(threshold),
        "args": vars(args),
    }
    (out_dir / "tube_verifier_model.json").write_text(json.dumps(model, indent=2))
    (out_dir / "summary.md").write_text(
        f"""# Tube Verifier Training Summary

Training examples: {len(examples)}  
Positive examples: {int(np.sum(y == 1))}  
Negative examples: {int(np.sum(y == 0))}

## Leave-One-Clip-Out

See `loco_summary.csv` and `loco_predictions.csv`.

## In-Sample

See `in_sample_summary.csv`. This is only a fit sanity check, not a generalization claim.
"""
    )
    print(out_dir / "loco_summary.csv")
    print(out_dir / "in_sample_summary.csv")


if __name__ == "__main__":
    main()
