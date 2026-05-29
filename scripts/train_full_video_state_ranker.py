#!/usr/bin/env python3
"""Train out-of-fold candidate scores for full-video state-machine evaluation.

This bridges the gap between visible-frame tube ranking and complete-video
acquire/null behavior. It labels exported ``top_tubes.csv`` rows from frame-level
target boxes:

- visible frame + near labeled target center => positive candidate;
- visible frame + far from target => negative candidate;
- no-target frame => negative candidate.

It then emits out-of-fold per-candidate scores and one best candidate per frame
so ``evaluate_lock_state_machine.py`` can sweep acquisition logic without using
in-sample detector scores.
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
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_NUMERIC_COLUMNS = {
    "frame",
    "track_id",
    "x",
    "y",
    "selected",
    "eligible",
    "passes_floor",
    "cand_frame",
    "candidate_frame",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, help="Frame-level labels with visible and det_x/det_y/det_w/det_h.")
    p.add_argument("--top_tubes", required=True, help="Exported top_tubes.csv from tbd_motion_detector.py.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="", help="Optional clip id filter for labels.")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--positive_tol_px", type=float, default=8.0)
    p.add_argument("--negative_min_dist_px", type=float, default=16.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold_strategy", choices=("stratified_blocks", "frame_mod"), default="stratified_blocks")
    p.add_argument("--random_state", type=int, default=17)
    p.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "hist_gbdt", "extra_trees"),
        default=("logistic", "hist_gbdt", "extra_trees"),
    )
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


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    out = safe_float(value)
    return default if out is None else int(out)


def label_visible(row: dict[str, str]) -> bool:
    raw = str(row.get("visible", "")).strip().lower()
    if raw in {"1", "true", "yes", "visible", "target"}:
        return True
    if raw in {"0", "false", "no", "empty", "none", "not_visible"}:
        return False
    return bool(safe_int(raw, 0))


def row_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        safe_float(row.get("x"), 0.0) or 0.0,
        safe_float(row.get("y"), 0.0) or 0.0,
        safe_float(row.get("w"), 1.0) or 1.0,
        safe_float(row.get("h"), 1.0) or 1.0,
    )


def label_bbox(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    x = safe_float(row.get("det_x", row.get("x")))
    y = safe_float(row.get("det_y", row.get("y")))
    w = safe_float(row.get("det_w", row.get("w")), 1.0)
    h = safe_float(row.get("det_h", row.get("h")), 1.0)
    if x is None or y is None or w is None or h is None:
        return None
    return x, y, w, h


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return float(math.hypot((ax + 0.5 * aw) - (bx + 0.5 * bw), (ay + 0.5 * ah) - (by + 0.5 * bh)))


def load_labels(path: Path, clip: str) -> dict[int, dict[str, Any]]:
    rows = read_csv(path)
    if clip:
        rows = [r for r in rows if r.get("clip", "") == clip]
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame = safe_int(row.get("frame"), -1)
        if frame < 0:
            continue
        visible = label_visible(row)
        bbox = label_bbox(row) if visible else None
        out[frame] = {
            "clip": row.get("clip", clip),
            "frame": frame,
            "visible": visible and bbox is not None,
            "bbox": bbox,
            "confidence": row.get("confidence", ""),
            "source": row.get("source", ""),
        }
    return out


def load_top_tubes(path: Path, max_rank: int) -> dict[int, list[dict[str, str]]]:
    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(path):
        rank = safe_int(row.get("rank"), 999999)
        if rank > max_rank:
            continue
        frame = safe_int(row.get("frame"), -1)
        if frame < 0:
            continue
        by_frame.setdefault(frame, []).append(row)
    for rows in by_frame.values():
        rows.sort(key=lambda r: safe_int(r.get("rank"), 999999))
    return by_frame


def make_examples(
    labels: dict[int, dict[str, Any]],
    top_by_frame: dict[int, list[dict[str, str]]],
    positive_tol_px: float,
    negative_min_dist_px: float,
) -> tuple[list[dict[str, Any]], int]:
    examples: list[dict[str, Any]] = []
    ignored_near = 0
    for frame, rows in sorted(top_by_frame.items()):
        lab = labels.get(frame)
        if lab is None:
            continue
        for row in rows:
            dist = None
            y = 0
            reason = "no_target_negative"
            if lab["visible"] and lab["bbox"] is not None:
                dist = center_dist(row_bbox(row), lab["bbox"])
                if dist <= positive_tol_px:
                    y = 1
                    reason = "target_positive"
                elif dist >= negative_min_dist_px:
                    y = 0
                    reason = "far_negative"
                else:
                    ignored_near += 1
                    continue
            examples.append(
                {
                    "frame": frame,
                    "label": lab,
                    "row": row,
                    "y": y,
                    "dist_px": dist,
                    "reason": reason,
                }
            )
    return examples, ignored_near


def infer_features(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    sources: set[str] = set()
    for row in rows:
        source = row.get("cand_source", "")
        if source:
            sources.add(source)
        for key, value in row.items():
            if key in EXCLUDE_NUMERIC_COLUMNS or key == "cand_source":
                continue
            if safe_float(value) is not None and key not in numeric:
                numeric.append(key)
    return numeric, [f"src_{s}" for s in sorted(sources)]


def vectorize_rows(rows: list[dict[str, str]], numeric: list[str], sources: list[str]) -> np.ndarray:
    data: list[list[float]] = []
    for row in rows:
        vals: list[float] = []
        for name in numeric:
            value = safe_float(row.get(name))
            vals.append(np.nan if value is None else value)
        source = row.get("cand_source", "")
        vals.extend(1.0 if name == f"src_{source}" else 0.0 for name in sources)
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
                        max_iter=90,
                        learning_rate=0.045,
                        max_leaf_nodes=6,
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
                        max_depth=5,
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


def make_frame_folds(labels: dict[int, dict[str, Any]], folds: int, strategy: str) -> dict[int, int]:
    frames = sorted(labels)
    n = max(2, folds)
    if strategy == "frame_mod":
        return {frame: i % n for i, frame in enumerate(frames)}
    visible = [f for f in frames if labels[f]["visible"]]
    invisible = [f for f in frames if not labels[f]["visible"]]
    out: dict[int, int] = {}
    for group in (visible, invisible):
        chunks = np.array_split(np.asarray(group, dtype=int), n)
        for fold, chunk in enumerate(chunks):
            for frame in chunk:
                out[int(frame)] = int(fold)
    return out


def frame_best_rows(
    examples: list[dict[str, Any]],
    scores: np.ndarray,
    model_name: str,
    score_column: str,
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        by_frame.setdefault(int(ex["frame"]), []).append(i)
    rows: list[dict[str, Any]] = []
    for frame, idxs in sorted(by_frame.items()):
        best_i = max(idxs, key=lambda i: float(scores[i]))
        ex = examples[best_i]
        row = ex["row"]
        lab = ex["label"]
        strict_hit = bool(ex["y"] == 1)
        rows.append(
            {
                "model": model_name,
                "clip": lab.get("clip", ""),
                "frame": frame,
                "visible": int(lab["visible"]),
                "rank": safe_int(row.get("rank"), 999999),
                score_column: round(float(scores[best_i]), 9),
                "score": round(float(scores[best_i]), 9),
                "x": row.get("x", ""),
                "y": row.get("y", ""),
                "w": row.get("w", ""),
                "h": row.get("h", ""),
                "source": row.get("cand_source", ""),
                "target_candidate": int(ex["y"]),
                "strict_hit_if_selected": int(strict_hit),
                "dist_px": "" if ex["dist_px"] is None else round(float(ex["dist_px"]), 3),
                "verified_score": row.get("verified_score", ""),
                "tube_verifier_score": row.get("tube_verifier_score", ""),
            }
        )
    return rows


def summarize_best_rows(
    labels: dict[int, dict[str, Any]],
    best_rows: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    by_frame = {safe_int(r["frame"]): r for r in best_rows}
    visible_frames = [f for f, lab in labels.items() if lab["visible"]]
    invisible_frames = [f for f, lab in labels.items() if not lab["visible"]]
    visible_with_candidate = [f for f in visible_frames if f in by_frame]
    invisible_with_candidate = [f for f in invisible_frames if f in by_frame]
    strict = sum(int(by_frame[f].get("strict_hit_if_selected", 0)) for f in visible_with_candidate)
    return {
        "model": model_name,
        "label_frames": len(labels),
        "visible_frames": len(visible_frames),
        "invisible_frames": len(invisible_frames),
        "visible_frames_with_candidates": len(visible_with_candidate),
        "invisible_frames_with_candidates": len(invisible_with_candidate),
        "visible_best_strict": strict,
        "visible_best_strict_rate": round(strict / max(1, len(visible_with_candidate)), 4),
        "candidate_frames": len(best_rows),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(Path(args.labels), args.clip)
    top_by_frame = load_top_tubes(Path(args.top_tubes), args.max_rank)
    examples, ignored_near = make_examples(labels, top_by_frame, args.positive_tol_px, args.negative_min_dist_px)
    if not examples:
        raise SystemExit("no candidate examples")
    rows = [ex["row"] for ex in examples]
    numeric, sources = infer_features(rows)
    x = vectorize_rows(rows, numeric, sources)
    y = np.asarray([int(ex["y"]) for ex in examples], dtype=np.int32)
    frames = np.asarray([int(ex["frame"]) for ex in examples], dtype=np.int32)
    frame_to_fold = make_frame_folds(labels, args.folds, args.fold_strategy)

    write_csv(
        out_dir / "candidate_training_examples.csv",
        [
            {
                "clip": ex["label"].get("clip", ""),
                "frame": ex["frame"],
                "rank": ex["row"].get("rank", ""),
                "y": ex["y"],
                "reason": ex["reason"],
                "dist_px": "" if ex["dist_px"] is None else round(float(ex["dist_px"]), 3),
                "visible": int(ex["label"]["visible"]),
                "source": ex["row"].get("cand_source", ""),
                "verified_score": ex["row"].get("verified_score", ""),
            }
            for ex in examples
        ],
    )

    summaries: list[dict[str, Any]] = []
    baseline_scores = np.asarray([safe_float(ex["row"].get("verified_score"), -1e9) or -1e9 for ex in examples])
    baseline_best = frame_best_rows(examples, baseline_scores, "baseline_verified_score", "learned_score")
    write_csv(out_dir / "best_per_frame_baseline_verified_score.csv", baseline_best)
    summaries.append(summarize_best_rows(labels, baseline_best, "baseline_verified_score"))

    specs = make_models(args.random_state)
    selected_specs = {name: specs[name] for name in args.models}
    model_score_columns: dict[str, str] = {}
    for model_name, spec in selected_specs.items():
        scores = np.full(len(examples), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []
        for fold in sorted(set(frame_to_fold.values())):
            test_idx = np.asarray([i for i, f in enumerate(frames) if frame_to_fold.get(int(f)) == fold], dtype=int)
            train_idx = np.asarray([i for i, f in enumerate(frames) if frame_to_fold.get(int(f)) != fold], dtype=int)
            if len(test_idx) == 0:
                continue
            if len(set(y[train_idx])) < 2:
                fold_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "status": "skipped_one_class_train",
                        "test_examples": int(len(test_idx)),
                        "train_examples": int(len(train_idx)),
                        "train_pos": int(y[train_idx].sum()),
                        "train_neg": int(len(train_idx) - y[train_idx].sum()),
                    }
                )
                continue
            model = clone(spec)
            model.fit(x[train_idx], y[train_idx])
            scores[test_idx] = predict_score(model, x[test_idx])
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "status": "ok",
                    "test_examples": int(len(test_idx)),
                    "train_examples": int(len(train_idx)),
                    "train_pos": int(y[train_idx].sum()),
                    "train_neg": int(len(train_idx) - y[train_idx].sum()),
                    "test_pos": int(y[test_idx].sum()),
                    "test_neg": int(len(test_idx) - y[test_idx].sum()),
                }
            )

        valid_idx = np.where(np.isfinite(scores))[0]
        if len(valid_idx) == 0:
            continue
        oof_examples = [examples[int(i)] for i in valid_idx]
        oof_scores = scores[valid_idx]
        score_column = f"oof_{model_name}_score"
        model_score_columns[model_name] = score_column
        cand_rows = []
        for local_i, ex in enumerate(oof_examples):
            row = ex["row"]
            cand_rows.append(
                {
                    "clip": ex["label"].get("clip", ""),
                    "frame": ex["frame"],
                    "rank": row.get("rank", ""),
                    score_column: round(float(oof_scores[local_i]), 9),
                    "score": round(float(oof_scores[local_i]), 9),
                    "x": row.get("x", ""),
                    "y": row.get("y", ""),
                    "w": row.get("w", ""),
                    "h": row.get("h", ""),
                    "target_candidate": int(ex["y"]),
                    "dist_px": "" if ex["dist_px"] is None else round(float(ex["dist_px"]), 3),
                    "visible": int(ex["label"]["visible"]),
                    "source": row.get("cand_source", ""),
                    "verified_score": row.get("verified_score", ""),
                }
            )
        best_rows = frame_best_rows(oof_examples, oof_scores, model_name, score_column)
        write_csv(out_dir / f"oof_candidate_scores_{model_name}.csv", cand_rows)
        write_csv(out_dir / f"oof_best_per_frame_{model_name}.csv", best_rows)
        write_csv(out_dir / f"folds_{model_name}.csv", fold_rows)
        summary = summarize_best_rows(labels, best_rows, model_name)
        summary["oof_scored_examples"] = int(len(valid_idx))
        summary["oof_score_column"] = score_column
        summaries.append(summary)

        final_model = clone(spec)
        final_model.fit(x, y)
        joblib.dump(
            {
                "model": final_model,
                "numeric_features": numeric,
                "source_features": sources,
                "max_rank": args.max_rank,
                "positive_tol_px": args.positive_tol_px,
                "negative_min_dist_px": args.negative_min_dist_px,
                "model_name": model_name,
            },
            out_dir / f"{model_name}_full_fit_model.joblib",
        )

    write_csv(out_dir / "model_summary.csv", summaries)
    metadata = {
        "labels": str(args.labels),
        "top_tubes": str(args.top_tubes),
        "clip": args.clip,
        "max_rank": args.max_rank,
        "positive_tol_px": args.positive_tol_px,
        "negative_min_dist_px": args.negative_min_dist_px,
        "folds": args.folds,
        "fold_strategy": args.fold_strategy,
        "label_frames": len(labels),
        "candidate_frames": len(top_by_frame),
        "examples": len(examples),
        "positive_examples": int(y.sum()),
        "negative_examples": int(len(y) - y.sum()),
        "ignored_near_examples": ignored_near,
        "numeric_features": numeric,
        "source_features": sources,
        "score_columns": model_score_columns,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "README.md").write_text(
        "# Full-Video State Ranker\n\n"
        "This artifact trains out-of-fold candidate scores for state-machine evaluation.\n"
        "It is not leave-one-clip-out because this benchmark currently uses one full-video\n"
        "label set; treat it as a null/acquisition harness, not final generalization proof.\n\n"
        "Use `oof_best_per_frame_<model>.csv` with `scripts/evaluate_lock_state_machine.py`.\n"
    )
    print(out_dir / "model_summary.csv")


if __name__ == "__main__":
    main()
