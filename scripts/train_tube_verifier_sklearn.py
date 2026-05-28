#!/usr/bin/env python3
"""Train a small tabular tube verifier from human top-candidate labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


EXCLUDE_COLUMNS = {
    "frame",
    "rank",
    "track_id",
    "x",
    "y",
    "eligible",
    "passes_floor",
    "selected",
}

POSITIVE_LABEL = "target"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tube_labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=16)
    p.add_argument("--threshold_metric", choices=("balanced", "precision", "recall"), default="balanced")
    p.add_argument("--random_state", type=int, default=7)
    return p.parse_args()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_top_tubes(results_dir: Path, clip: str) -> dict[tuple[int, int], dict[str, str]]:
    path = results_dir / clip / "top_tubes.csv"
    out: dict[tuple[int, int], dict[str, str]] = {}
    if not path.exists():
        return out
    for row in read_csv(path):
        frame = int(float(row.get("frame", "0") or 0))
        rank = int(float(row.get("rank", "999") or 999))
        out[(frame, rank)] = row
    return out


def load_examples(tube_labels: Path, results_dir: Path, max_rank: int) -> list[dict[str, Any]]:
    review_rows = read_csv(tube_labels)
    cache: dict[str, dict[tuple[int, int], dict[str, str]]] = {}
    examples: list[dict[str, Any]] = []
    for review in review_rows:
        human_label = review.get("human_label", "").strip().lower()
        if not human_label:
            continue
        clip = review.get("clip", "")
        frame = int(float(review.get("frame", "0") or 0))
        rank = int(float(review.get("rank", "999") or 999))
        if rank > max_rank:
            continue
        if clip not in cache:
            cache[clip] = load_top_tubes(results_dir, clip)
        top = cache[clip].get((frame, rank))
        if not top:
            continue
        examples.append(
            {
                "clip": clip,
                "frame": frame,
                "rank": rank,
                "human_label": human_label,
                "y": 1 if human_label == POSITIVE_LABEL else 0,
                "review": review,
                "row": top,
            }
        )
    return examples


def infer_features(examples: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    sources: set[str] = set()
    for ex in examples:
        row = ex["row"]
        source = row.get("cand_source", "")
        if source:
            sources.add(source)
        for key, value in row.items():
            if key in EXCLUDE_COLUMNS or key == "cand_source":
                continue
            if safe_float(value) is not None and key not in numeric:
                numeric.append(key)
    return numeric, [f"src_{s}" for s in sorted(sources)]


def vectorize(examples: list[dict[str, Any]], numeric: list[str], sources: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for ex in examples:
        row = ex["row"]
        vals = [safe_float(row.get(name)) if safe_float(row.get(name)) is not None else np.nan for name in numeric]
        source = row.get("cand_source", "")
        vals.extend(1.0 if name == f"src_{source}" else 0.0 for name in sources)
        rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def model_specs(random_state: int) -> dict[str, Any]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.35,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gbdt": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=80,
                        learning_rate=0.045,
                        max_leaf_nodes=5,
                        l2_regularization=1.5,
                        min_samples_leaf=10,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=250,
                        max_depth=4,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def predict_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def frame_groups(examples: list[dict[str, Any]], indices: np.ndarray) -> dict[tuple[str, int], list[int]]:
    allowed = set(int(i) for i in indices)
    groups: dict[tuple[str, int], list[int]] = {}
    for i, ex in enumerate(examples):
        if i in allowed:
            groups.setdefault((ex["clip"], int(ex["frame"])), []).append(i)
    return groups


def evaluate_frame_selection(
    examples: list[dict[str, Any]],
    groups: dict[tuple[str, int], list[int]],
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "frames": 0,
        "target_frames": 0,
        "no_target_frames": 0,
        "target_hit": 0,
        "target_miss": 0,
        "target_wrong": 0,
        "no_target_tn": 0,
        "no_target_fp": 0,
    }
    for (clip, frame), idxs in sorted(groups.items()):
        counts["frames"] += 1
        has_target = any(examples[i]["y"] == 1 for i in idxs)
        if has_target:
            counts["target_frames"] += 1
        else:
            counts["no_target_frames"] += 1
        best_i = max(idxs, key=lambda i: float(scores[i]))
        best_score = float(scores[best_i])
        selected = best_score >= threshold
        selected_label = examples[best_i]["human_label"] if selected else ""
        if has_target and selected_label == POSITIVE_LABEL:
            outcome = "target_hit"
            counts["target_hit"] += 1
        elif has_target and selected:
            outcome = "target_wrong"
            counts["target_wrong"] += 1
        elif has_target:
            outcome = "target_miss"
            counts["target_miss"] += 1
        elif selected:
            outcome = "no_target_fp"
            counts["no_target_fp"] += 1
        else:
            outcome = "no_target_tn"
            counts["no_target_tn"] += 1
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "has_target": int(has_target),
                "selected": int(selected),
                "selected_rank": examples[best_i]["rank"] if selected else "",
                "selected_label": selected_label,
                "selected_score": round(best_score, 6),
                "threshold": round(float(threshold), 6),
                "outcome": outcome,
            }
        )
    counts["target_recall"] = round(counts["target_hit"] / max(1, counts["target_frames"]), 3)
    selected_total = counts["target_hit"] + counts["target_wrong"] + counts["no_target_fp"]
    counts["selected_precision"] = round(counts["target_hit"] / max(1, selected_total), 3)
    counts["no_target_suppression"] = round(counts["no_target_tn"] / max(1, counts["no_target_frames"]), 3)
    counts["accuracy"] = round((counts["target_hit"] + counts["no_target_tn"]) / max(1, counts["frames"]), 3)
    return counts, rows


def objective(summary: dict[str, Any], metric: str) -> float:
    if metric == "precision":
        return 4.0 * summary["target_hit"] + 1.0 * summary["no_target_tn"] - 5.0 * summary["no_target_fp"] - 3.0 * summary["target_wrong"] - 1.2 * summary["target_miss"]
    if metric == "recall":
        return 5.0 * summary["target_hit"] + 0.6 * summary["no_target_tn"] - 2.5 * summary["no_target_fp"] - 2.5 * summary["target_wrong"] - 0.5 * summary["target_miss"]
    return 4.0 * summary["target_hit"] + 1.5 * summary["no_target_tn"] - 4.0 * summary["no_target_fp"] - 3.0 * summary["target_wrong"] - 1.0 * summary["target_miss"]


def choose_threshold(examples: list[dict[str, Any]], groups: dict[tuple[str, int], list[int]], scores: np.ndarray, metric: str) -> float:
    candidates = [-1e9, 1e9]
    for idxs in groups.values():
        candidates.extend(float(scores[i]) for i in idxs)
    best_thr = 1e9
    best_value = -1e18
    for threshold in sorted(set(candidates)):
        summary, _ = evaluate_frame_selection(examples, groups, scores, threshold)
        value = objective(summary, metric)
        if value > best_value:
            best_value = value
            best_thr = float(threshold)
    return best_thr


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
    examples = load_examples(Path(args.tube_labels), Path(args.results_dir), args.max_rank)
    if not examples:
        raise SystemExit("no examples")
    numeric, sources = infer_features(examples)
    feature_names = numeric + sources
    x = vectorize(examples, numeric, sources)
    y = np.asarray([ex["y"] for ex in examples], dtype=np.int32)
    clips = np.asarray([ex["clip"] for ex in examples])
    all_idx = np.arange(len(examples))
    all_groups = frame_groups(examples, all_idx)

    data_rows = [
        {
            "clip": ex["clip"],
            "frame": ex["frame"],
            "rank": ex["rank"],
            "human_label": ex["human_label"],
            "y": ex["y"],
            "bbox": ex["review"].get("bbox", ""),
            "verified_score": ex["review"].get("verified_score", ""),
            "notes": ex["review"].get("notes", ""),
        }
        for ex in examples
    ]
    write_csv(out_dir / "training_examples.csv", data_rows)

    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    specs = model_specs(args.random_state)
    for model_name, base_model in specs.items():
        for holdout in sorted(set(clips)):
            train_idx = np.where(clips != holdout)[0]
            test_idx = np.where(clips == holdout)[0]
            if len(train_idx) == 0 or len(set(y[train_idx])) < 2:
                continue
            model = clone(base_model)
            model.fit(x[train_idx], y[train_idx])
            scores = np.zeros(len(examples), dtype=np.float64)
            scores[train_idx] = predict_score(model, x[train_idx])
            scores[test_idx] = predict_score(model, x[test_idx])
            train_groups = frame_groups(examples, train_idx)
            test_groups = frame_groups(examples, test_idx)
            threshold = choose_threshold(examples, train_groups, scores, args.threshold_metric)
            summary, rows = evaluate_frame_selection(examples, test_groups, scores, threshold)
            summary.update(
                {
                    "model": model_name,
                    "fold": holdout,
                    "threshold": round(float(threshold), 6),
                    "train_examples": int(len(train_idx)),
                    "train_pos": int(y[train_idx].sum()),
                    "train_neg": int(len(train_idx) - y[train_idx].sum()),
                }
            )
            summary_rows.append(summary)
            for row in rows:
                row["model"] = model_name
                row["fold"] = holdout
            pred_rows.extend(rows)

        model = clone(base_model)
        model.fit(x, y)
        scores = predict_score(model, x)
        threshold = choose_threshold(examples, all_groups, scores, args.threshold_metric)
        summary, rows = evaluate_frame_selection(examples, all_groups, scores, threshold)
        summary.update(
            {
                "model": model_name,
                "fold": "in_sample_all",
                "threshold": round(float(threshold), 6),
                "train_examples": int(len(all_idx)),
                "train_pos": int(y.sum()),
                "train_neg": int(len(y) - y.sum()),
            }
        )
        summary_rows.append(summary)
        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "feature_names": feature_names,
                "numeric_features": numeric,
                "source_features": sources,
                "max_rank": args.max_rank,
                "threshold_metric": args.threshold_metric,
            },
            out_dir / f"{model_name}_model.joblib",
        )
        for row in rows:
            row["model"] = model_name
            row["fold"] = "in_sample_all"
        pred_rows.extend(rows)

    write_csv(out_dir / "frame_selection_summary.csv", summary_rows)
    write_csv(out_dir / "frame_selection_predictions.csv", pred_rows)

    totals: list[dict[str, Any]] = []
    for model_name in specs:
        rows = [r for r in summary_rows if r["model"] == model_name and r["fold"] != "in_sample_all"]
        total = {"model": model_name, "fold": "leave_one_clip_out_total"}
        for key in [
            "frames",
            "target_frames",
            "no_target_frames",
            "target_hit",
            "target_miss",
            "target_wrong",
            "no_target_tn",
            "no_target_fp",
        ]:
            total[key] = sum(int(r[key]) for r in rows)
        total["target_recall"] = round(total["target_hit"] / max(1, total["target_frames"]), 3)
        selected_total = total["target_hit"] + total["target_wrong"] + total["no_target_fp"]
        total["selected_precision"] = round(total["target_hit"] / max(1, selected_total), 3)
        total["no_target_suppression"] = round(total["no_target_tn"] / max(1, total["no_target_frames"]), 3)
        total["accuracy"] = round((total["target_hit"] + total["no_target_tn"]) / max(1, total["frames"]), 3)
        totals.append(total)
    write_csv(out_dir / "loco_totals.csv", totals)
    (out_dir / "summary.md").write_text(
        "# Human-Labeled Tube Verifier\n\n"
        f"Examples: {len(examples)}  \n"
        f"Positive target examples: {int(y.sum())}  \n"
        f"Negative examples: {int(len(y) - y.sum())}  \n"
        f"Frames: {len(all_groups)}  \n"
        f"Frames with target candidate: {sum(any(examples[i]['y'] == 1 for i in idxs) for idxs in all_groups.values())}  \n\n"
        "See `loco_totals.csv`, `frame_selection_summary.csv`, and `frame_selection_predictions.csv`.\n"
    )
    print(out_dir / "loco_totals.csv")


if __name__ == "__main__":
    main()
