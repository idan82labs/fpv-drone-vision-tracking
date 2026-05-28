#!/usr/bin/env python3
"""Train/evaluate a tube ranker from manual XY frame labels.

This is for the dense aaf1-style labels where every row gives the true target
center in detector coordinates. It tests whether exported top_tubes alternatives
contain enough tabular evidence to rank the target tube over clutter.
"""

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


EXCLUDE_COLUMNS = {
    "frame",
    "track_id",
    "x",
    "y",
    "selected",
    "eligible",
    "passes_floor",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--center_tol_px", type=float, default=3.0)
    p.add_argument("--loose_tol_px", type=float, default=6.0)
    p.add_argument("--negative_min_dist_px", type=float, default=8.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=11)
    p.add_argument("--confidence_filter", choices=("all", "high"), default="all")
    p.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "hist_gbdt", "extra_trees"),
        default=("logistic", "hist_gbdt", "extra_trees"),
        help="Model families to evaluate. Use logistic for a fast first pass.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def row_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row.get("x", "0") or 0),
        float(row.get("y", "0") or 0),
        float(row.get("w", "1") or 1),
        float(row.get("h", "1") or 1),
    )


def center_dist_to_label(row: dict[str, str], label: dict[str, str]) -> float:
    x, y, w, h = row_bbox(row)
    tx = float(label["det_x"]) + 0.5 * float(label["det_w"])
    ty = float(label["det_y"]) + 0.5 * float(label["det_h"])
    return float(math.hypot(x + 0.5 * w - tx, y + 0.5 * h - ty))


def load_top_tubes(results_dir: Path, clip: str, max_rank: int) -> dict[int, list[dict[str, str]]]:
    path = results_dir / clip / "top_tubes.csv"
    if not path.exists():
        raise SystemExit(f"missing top_tubes.csv: {path}")
    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(path):
        rank = int(float(row.get("rank", "999") or 999))
        if rank > max_rank:
            continue
        frame = int(float(row.get("frame", "0") or 0))
        by_frame.setdefault(frame, []).append(row)
    return by_frame


def infer_features(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    sources: set[str] = set()
    for row in rows:
        source = row.get("cand_source", "")
        if source:
            sources.add(source)
        for key, value in row.items():
            if key in EXCLUDE_COLUMNS or key == "cand_source":
                continue
            if safe_float(value) is not None and key not in numeric:
                numeric.append(key)
    source_features = [f"src_{s}" for s in sorted(sources)]
    return numeric, source_features


def vectorize(rows: list[dict[str, str]], numeric: list[str], sources: list[str]) -> np.ndarray:
    data: list[list[float]] = []
    for row in rows:
        vals: list[float] = []
        for name in numeric:
            val = safe_float(row.get(name))
            vals.append(np.nan if val is None else val)
        src = row.get("cand_source", "")
        vals.extend(1.0 if name == f"src_{src}" else 0.0 for name in sources)
        data.append(vals)
    return np.asarray(data, dtype=np.float64)


def make_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.28,
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
                        learning_rate=0.04,
                        max_leaf_nodes=5,
                        min_samples_leaf=12,
                        l2_regularization=1.5,
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
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def predict_score(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def contiguous_folds(frames: list[int], n_folds: int) -> dict[int, int]:
    chunks = np.array_split(np.asarray(sorted(frames), dtype=int), max(2, n_folds))
    out: dict[int, int] = {}
    for fold, chunk in enumerate(chunks):
        for frame in chunk:
            out[int(frame)] = int(fold)
    return out


def summarize_predictions(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    all_rows = rows
    high_rows = [r for r in rows if r["confidence"] == "high"]

    def _one(sub: list[dict[str, Any]], name: str) -> dict[str, Any]:
        total = len(sub)
        oracle = sum(1 for r in sub if r["oracle_hit"])
        strict = sum(1 for r in sub if r["strict_hit"])
        loose = sum(1 for r in sub if r["loose_hit"])
        wrong = sum(1 for r in sub if not r["strict_hit"])
        return {
            f"{name}_frames": total,
            f"{name}_oracle_recall": round(oracle / max(1, total), 3),
            f"{name}_strict_hit": strict,
            f"{name}_strict_recall": round(strict / max(1, total), 3),
            f"{name}_loose_hit": loose,
            f"{name}_loose_recall": round(loose / max(1, total), 3),
            f"{name}_wrong": wrong,
        }

    out = {"model": prefix}
    out.update(_one(all_rows, "all"))
    out.update(_one(high_rows, "high"))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
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


def evaluate_baseline(
    labels: list[dict[str, str]],
    top_by_frame: dict[int, list[dict[str, str]]],
    center_tol: float,
    loose_tol: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lab in labels:
        frame = int(lab["frame"])
        rows = top_by_frame.get(frame, [])
        if not rows:
            continue
        scored = sorted(rows, key=lambda r: safe_float(r.get("verified_score")) or -999.0, reverse=True)
        best = scored[0]
        d = center_dist_to_label(best, lab)
        oracle = any(center_dist_to_label(r, lab) <= center_tol for r in rows)
        out.append(
            {
                "model": "baseline_verified_score",
                "frame": frame,
                "fold": "",
                "confidence": lab.get("confidence", ""),
                "rank": int(float(best.get("rank", "999") or 999)),
                "score": safe_float(best.get("verified_score")),
                "strict_hit": d <= center_tol,
                "loose_hit": d <= loose_tol,
                "oracle_hit": oracle,
                "dist_px": round(d, 3),
                "x": best.get("x", ""),
                "y": best.get("y", ""),
                "w": best.get("w", ""),
                "h": best.get("h", ""),
                "source": best.get("cand_source", ""),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r for r in read_csv(Path(args.labels)) if r.get("clip") == args.clip]
    labels = [r for r in labels if r.get("visible", "1") != "0"]
    if args.confidence_filter == "high":
        labels = [r for r in labels if r.get("confidence") == "high"]
    labels.sort(key=lambda r: int(r["frame"]))
    top_by_frame = load_top_tubes(Path(args.results_dir), args.clip, args.max_rank)

    frame_labels = {int(r["frame"]): r for r in labels}
    all_tube_rows: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    ignored_near = 0
    for frame, lab in frame_labels.items():
        rows = top_by_frame.get(frame, [])
        for row in rows:
            all_tube_rows.append(row)
            dist = center_dist_to_label(row, lab)
            if dist <= args.center_tol_px:
                y = 1
            elif dist >= args.negative_min_dist_px:
                y = 0
            else:
                ignored_near += 1
                continue
            examples.append({"frame": frame, "label": lab, "row": row, "y": y, "dist_px": dist})

    if not examples:
        raise SystemExit("no examples")
    numeric, sources = infer_features([e["row"] for e in examples])
    x = vectorize([e["row"] for e in examples], numeric, sources)
    y = np.asarray([int(e["y"]) for e in examples], dtype=np.int32)
    frames = np.asarray([int(e["frame"]) for e in examples], dtype=np.int32)
    frame_to_fold = contiguous_folds(list(frame_labels), args.folds)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_rows = evaluate_baseline(labels, top_by_frame, args.center_tol_px, args.loose_tol_px)
    prediction_rows.extend(baseline_rows)
    summary_rows.append(summarize_predictions(baseline_rows, "baseline_verified_score"))

    all_models = make_models(args.random_state)
    models = {name: all_models[name] for name in args.models}
    for model_name, model_spec in models.items():
        model_pred_rows: list[dict[str, Any]] = []
        for fold in sorted(set(frame_to_fold.values())):
            train_idx = np.asarray([i for i, f in enumerate(frames) if frame_to_fold.get(int(f)) != fold], dtype=int)
            test_frames = [f for f, ff in frame_to_fold.items() if ff == fold]
            if len(set(y[train_idx])) < 2:
                continue
            model = Pipeline(model_spec.steps)
            model.fit(x[train_idx], y[train_idx])
            for frame in sorted(test_frames):
                lab = frame_labels[frame]
                rows = top_by_frame.get(frame, [])
                if not rows:
                    continue
                xt = vectorize(rows, numeric, sources)
                scores = predict_score(model, xt)
                best_i = int(np.argmax(scores))
                best = rows[best_i]
                dist = center_dist_to_label(best, lab)
                oracle = any(center_dist_to_label(r, lab) <= args.center_tol_px for r in rows)
                rec = {
                    "model": model_name,
                    "frame": frame,
                    "fold": fold,
                    "confidence": lab.get("confidence", ""),
                    "rank": int(float(best.get("rank", "999") or 999)),
                    "score": round(float(scores[best_i]), 6),
                    "strict_hit": dist <= args.center_tol_px,
                    "loose_hit": dist <= args.loose_tol_px,
                    "oracle_hit": oracle,
                    "dist_px": round(dist, 3),
                    "x": best.get("x", ""),
                    "y": best.get("y", ""),
                    "w": best.get("w", ""),
                    "h": best.get("h", ""),
                    "source": best.get("cand_source", ""),
                }
                model_pred_rows.append(rec)
                prediction_rows.append(rec)
        summary_rows.append(summarize_predictions(model_pred_rows, model_name))

    # Fit the best-looking simple model on all labeled frames for downstream
    # inspection. Do not treat this as a generalization estimate.
    best_name = max(summary_rows[1:], key=lambda r: (r["all_strict_recall"], r["high_strict_recall"]))["model"]
    final_model = make_models(args.random_state)[best_name]
    final_model.fit(x, y)
    model_path = out_dir / f"{best_name}_manual_xy_ranker.joblib"
    joblib.dump(
        {
            "model": final_model,
            "numeric_features": numeric,
            "source_features": sources,
            "max_rank": args.max_rank,
            "center_tol_px": args.center_tol_px,
            "loose_tol_px": args.loose_tol_px,
            "negative_min_dist_px": args.negative_min_dist_px,
            "clip": args.clip,
            "best_model_cv": best_name,
        },
        model_path,
    )

    write_csv(out_dir / "cv_predictions.csv", prediction_rows)
    write_csv(out_dir / "cv_summary.csv", summary_rows)
    write_csv(
        out_dir / "training_examples.csv",
        [
            {
                "frame": ex["frame"],
                "y": ex["y"],
                "dist_px": round(ex["dist_px"], 3),
                "rank": ex["row"].get("rank", ""),
                "source": ex["row"].get("cand_source", ""),
                "confidence": ex["label"].get("confidence", ""),
            }
            for ex in examples
        ],
    )
    metadata = {
        "labels": str(args.labels),
        "results_dir": str(args.results_dir),
        "clip": args.clip,
        "max_rank": args.max_rank,
        "frames": len(labels),
        "examples": len(examples),
        "positive_examples": int(np.sum(y == 1)),
        "negative_examples": int(np.sum(y == 0)),
        "ignored_near_examples": ignored_near,
        "numeric_features": numeric,
        "source_features": sources,
        "best_model_cv": best_name,
        "model_path": str(model_path),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "README.md").write_text(
        "# Manual XY Tube Ranker\n\n"
        f"Best CV model: `{best_name}`\n\n"
        "See `cv_summary.csv`, `cv_predictions.csv`, and `metadata.json`.\n"
    )
    print(out_dir / "cv_summary.csv")
    print(model_path)


if __name__ == "__main__":
    main()
