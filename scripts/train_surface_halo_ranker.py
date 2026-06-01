#!/usr/bin/env python3
"""Train/evaluate a ranker for surface-halo proposal candidates.

The recenter/surface-halo branch can put the aaf1 terrain target into the
candidate set, but the existing held-out crop model ranks those candidates
poorly. This script answers the next question: are the recovered candidates
learnable from the new frame-by-frame terrain labels, using only candidate/tube
features and not absolute target coordinates?

This is an offline experiment. It does not change runtime defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_NUMERIC = {
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "selected",
    "eligible",
    "passes_floor",
    "cand_frame",
    "cand_is_current",
    "target_cx",
    "target_cy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--max_rank", type=int, default=180)
    p.add_argument("--center_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--negative_min_dist_px", type=float, default=20.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--fold_mode",
        choices=("interleaved", "blocked"),
        default="interleaved",
        help=(
            "Frame-fold assignment. interleaved is useful for quick same-mode checks; "
            "blocked holds out contiguous frame ranges and is the stricter tracking sanity gate."
        ),
    )
    p.add_argument("--models", nargs="+", choices=("logistic", "hist_gbdt", "extra_trees"), default=["hist_gbdt", "extra_trees"])
    p.add_argument("--random_state", type=int, default=31)
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


def fnum(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def fint(value: Any, default: int = 0) -> int:
    out = fnum(value)
    return default if out is None else int(round(out))


def label_visible(row: dict[str, str]) -> bool:
    raw = str(row.get("visible", "")).strip().lower()
    return raw in {"1", "true", "yes", "visible", "target"}


def label_center(row: dict[str, str]) -> tuple[float, float]:
    x = fnum(row.get("det_x", row.get("x")), 0.0) or 0.0
    y = fnum(row.get("det_y", row.get("y")), 0.0) or 0.0
    w = fnum(row.get("det_w", row.get("w")), 3.0) or 3.0
    h = fnum(row.get("det_h", row.get("h")), 3.0) or 3.0
    return x + 0.5 * w, y + 0.5 * h


def row_center(row: dict[str, str]) -> tuple[float, float]:
    x = fnum(row.get("x"), 0.0) or 0.0
    y = fnum(row.get("y"), 0.0) or 0.0
    w = fnum(row.get("w"), 3.0) or 3.0
    h = fnum(row.get("h"), 3.0) or 3.0
    return x + 0.5 * w, y + 0.5 * h


def center_dist(row: dict[str, str], target: tuple[float, float]) -> float:
    cx, cy = row_center(row)
    return float(math.hypot(cx - target[0], cy - target[1]))


def load_visible_labels(path: Path, clip: str) -> list[dict[str, str]]:
    labels: list[dict[str, str]] = []
    for row in read_csv(path):
        if clip and row.get("clip", "") != clip:
            continue
        if label_visible(row):
            labels.append(row)
    labels.sort(key=lambda r: fint(r.get("frame"), -1))
    return labels


def load_candidates(path: Path, clip: str, max_rank: int) -> dict[int, list[dict[str, str]]]:
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if clip and row.get("clip", clip) != clip:
            continue
        frame = fint(row.get("frame"), -1)
        rank = fint(row.get("rank"), 999999)
        if frame >= 0 and rank <= max_rank:
            by_frame[frame].append(row)
    for rows in by_frame.values():
        rows.sort(key=lambda r: fint(r.get("rank"), 999999))
    return by_frame


def build_examples(
    labels: list[dict[str, str]],
    candidates: dict[int, list[dict[str, str]]],
    center_tol_px: float,
    negative_min_dist_px: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    examples: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, str]] = []
    for lab in labels:
        frame = fint(lab.get("frame"), -1)
        target = label_center(lab)
        for row in candidates.get(frame, []):
            d = center_dist(row, target)
            if d <= center_tol_px:
                y = 1
            elif d >= negative_min_dist_px:
                y = 0
            else:
                continue
            examples.append({"clip": lab.get("clip", ""), "frame": frame, "row": row, "dist_px": d, "y": y})
            candidate_rows.append(row)
    return examples, candidate_rows


def infer_features(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    numeric: list[str] = []
    sources: set[str] = set()
    variants: set[str] = set()
    for row in rows:
        if row.get("cand_source"):
            sources.add(str(row.get("cand_source")))
        if row.get("proposal_variant"):
            variants.add(str(row.get("proposal_variant")))
        for key, value in row.items():
            if key in EXCLUDE_NUMERIC or key in {"clip", "cand_source", "proposal_variant", "cand_router_state", "crop_pred_class"}:
                continue
            if fnum(value) is not None and key not in numeric:
                numeric.append(key)
    return numeric, [f"src_{s}" for s in sorted(sources)], [f"variant_{v}" for v in sorted(variants)]


def vectorize(rows: list[dict[str, str]], numeric: list[str], sources: list[str], variants: list[str]) -> np.ndarray:
    data: list[list[float]] = []
    for row in rows:
        vals: list[float] = []
        for key in numeric:
            value = fnum(row.get(key))
            vals.append(np.nan if value is None else value)
        src = str(row.get("cand_source", ""))
        vals.extend(1.0 if key == f"src_{src}" else 0.0 for key in sources)
        variant = str(row.get("proposal_variant", ""))
        vals.extend(1.0 if key == f"variant_{variant}" else 0.0 for key in variants)
        data.append(vals)
    return np.asarray(data, dtype=np.float64)


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed)),
            ]
        ),
        "hist_gbdt": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=16, l2_regularization=0.8, random_state=seed)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=350, max_depth=7, min_samples_leaf=4, class_weight="balanced", random_state=seed)),
            ]
        ),
    }


def predict_score(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def baseline_score(row: dict[str, str], name: str) -> float:
    if name == "rank":
        return -float(fint(row.get("rank"), 999999))
    if name == "verified":
        return fnum(row.get("verified_score"), fnum(row.get("score"), 0.0)) or 0.0
    if name == "crop_llr":
        t = fnum(row.get("crop_t_logit"), -6.0) or -6.0
        vals = [fnum(row.get(f"crop_{c}_logit"), -6.0) or -6.0 for c in ("s", "e", "h", "g")]
        m = max(vals)
        return t - (m + math.log(sum(math.exp(v - m) for v in vals)))
    raise ValueError(name)


def summarize_selection(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    total = len(rows)
    strict = sum(bool(r.get("strict_hit")) for r in rows)
    loose = sum(bool(r.get("loose_hit")) for r in rows)
    oracle = sum(bool(r.get("oracle_hit")) for r in rows)
    return {
        "model": model,
        "frames": total,
        "strict": strict,
        "strict_rate": round(strict / max(1, total), 4),
        "loose": loose,
        "loose_rate": round(loose / max(1, total), 4),
        "oracle": oracle,
        "oracle_rate": round(oracle / max(1, total), 4),
    }


def assign_frame_folds(labels: list[dict[str, str]], folds: int, mode: str) -> dict[int, int]:
    """Assign each labeled frame to a validation fold.

    Interleaved folds are optimistic for continuous video because adjacent
    frames leak near-identical examples into train and test. Blocked folds are
    stricter: each fold is a contiguous slice of the reviewed timeline.
    """

    frames = sorted({fint(lab.get("frame"), -1) for lab in labels if fint(lab.get("frame"), -1) >= 0})
    if not frames:
        return {}
    folds = max(2, min(int(folds), len(frames)))
    if mode == "interleaved":
        return {frame: idx % folds for idx, frame in enumerate(frames)}
    if mode != "blocked":
        raise ValueError(f"unknown fold mode: {mode}")

    out: dict[int, int] = {}
    for idx, frame in enumerate(frames):
        fold = min(folds - 1, int(idx * folds / len(frames)))
        out[frame] = fold
    return out


def evaluate_baseline(
    labels: list[dict[str, str]],
    candidates: dict[int, list[dict[str, str]]],
    score_name: str,
    center_tol_px: float,
    loose_tol_px: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lab in labels:
        frame = fint(lab.get("frame"), -1)
        target = label_center(lab)
        rows = candidates.get(frame, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: baseline_score(r, score_name))
        d = center_dist(best, target)
        oracle_d = min((center_dist(row, target) for row in rows), default=float("inf"))
        out.append(
            {
                "model": f"baseline_{score_name}",
                "clip": lab.get("clip", ""),
                "frame": frame,
                "rank": best.get("rank", ""),
                "score": round(baseline_score(best, score_name), 6),
                "source": best.get("cand_source", ""),
                "proposal_variant": best.get("proposal_variant", ""),
                "dist_px": round(d, 3),
                "strict_hit": d <= center_tol_px,
                "loose_hit": d <= loose_tol_px,
                "oracle_hit": oracle_d <= center_tol_px,
            }
        )
    return out


def evaluate_model(
    model: Pipeline,
    labels: list[dict[str, str]],
    candidates: dict[int, list[dict[str, str]]],
    numeric: list[str],
    sources: list[str],
    variants: list[str],
    center_tol_px: float,
    loose_tol_px: float,
    model_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lab in labels:
        frame = fint(lab.get("frame"), -1)
        target = label_center(lab)
        rows = candidates.get(frame, [])
        if not rows:
            continue
        scores = predict_score(model, vectorize(rows, numeric, sources, variants))
        best_i = int(np.argmax(scores))
        best = rows[best_i]
        d = center_dist(best, target)
        oracle_d = min((center_dist(row, target) for row in rows), default=float("inf"))
        out.append(
            {
                "model": model_name,
                "clip": lab.get("clip", ""),
                "frame": frame,
                "rank": best.get("rank", ""),
                "score": round(float(scores[best_i]), 6),
                "source": best.get("cand_source", ""),
                "proposal_variant": best.get("proposal_variant", ""),
                "dist_px": round(d, 3),
                "strict_hit": d <= center_tol_px,
                "loose_hit": d <= loose_tol_px,
                "oracle_hit": oracle_d <= center_tol_px,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_visible_labels(Path(args.labels), args.clip)
    candidates = load_candidates(Path(args.candidates), args.clip, args.max_rank)
    examples, example_rows = build_examples(labels, candidates, args.center_tol_px, args.negative_min_dist_px)
    if not examples:
        raise SystemExit("no examples")
    numeric, sources, variants = infer_features(example_rows)
    x = vectorize([ex["row"] for ex in examples], numeric, sources, variants)
    y = np.asarray([int(ex["y"]) for ex in examples], dtype=np.int32)

    predictions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for baseline in ("rank", "verified", "crop_llr"):
        rows = evaluate_baseline(labels, candidates, baseline, args.center_tol_px, args.loose_tol_px)
        predictions.extend(rows)
        summaries.append(summarize_selection(rows, f"baseline_{baseline}"))

    folds = max(2, args.folds)
    frame_folds = assign_frame_folds(labels, folds, args.fold_mode)
    models = make_models(args.random_state)
    for model_name in args.models:
        fold_predictions: list[dict[str, Any]] = []
        fold_model_name = f"{model_name}_{args.fold_mode}"
        for fold in range(folds):
            train_idx = np.asarray([i for i, ex in enumerate(examples) if frame_folds.get(int(ex["frame"]), 0) != fold], dtype=int)
            test_labels = [lab for lab in labels if frame_folds.get(fint(lab.get("frame"), -1), 0) == fold]
            if train_idx.size == 0 or len(set(y[train_idx].tolist())) < 2:
                continue
            model = make_models(args.random_state)[model_name]
            model.fit(x[train_idx], y[train_idx])
            fold_predictions.extend(
                evaluate_model(
                    model,
                    test_labels,
                    candidates,
                    numeric,
                    sources,
                    variants,
                    args.center_tol_px,
                    args.loose_tol_px,
                    fold_model_name,
                )
            )
        predictions.extend(fold_predictions)
        summaries.append(summarize_selection(fold_predictions, fold_model_name))

    best_name = max(
        [s for s in summaries if s["model"].endswith(f"_{args.fold_mode}")],
        key=lambda s: (s["strict_rate"], s["loose_rate"]),
    )["model"].removesuffix(f"_{args.fold_mode}")
    final_model = make_models(args.random_state)[best_name]
    final_model.fit(x, y)
    model_path = out_dir / f"{best_name}_surface_halo_ranker.joblib"
    joblib.dump(
        {
            "model": final_model,
            "numeric_features": numeric,
            "source_features": sources,
            "variant_features": variants,
            "max_rank": args.max_rank,
            "center_tol_px": args.center_tol_px,
            "negative_min_dist_px": args.negative_min_dist_px,
            "best_model_interleaved": best_name,
        },
        model_path,
    )

    write_csv(out_dir / "interleaved_predictions.csv", predictions)
    write_csv(out_dir / "summary.csv", summaries)
    write_csv(
        out_dir / "training_examples.csv",
        [
            {
                "clip": ex["clip"],
                "frame": ex["frame"],
                "rank": ex["row"].get("rank", ""),
                "source": ex["row"].get("cand_source", ""),
                "proposal_variant": ex["row"].get("proposal_variant", ""),
                "dist_px": round(float(ex["dist_px"]), 3),
                "y": ex["y"],
            }
            for ex in examples
        ],
    )
    metadata = {
        "labels": args.labels,
        "candidates": args.candidates,
        "clip": args.clip,
        "frames": len(labels),
        "examples": len(examples),
        "class_counts": dict(Counter(y.tolist())),
        "numeric_features": numeric,
        "source_features": sources,
        "variant_features": variants,
        "best_model_interleaved": best_name,
        "fold_mode": args.fold_mode,
        "model_path": str(model_path),
        "metric_caveat": f"{args.fold_mode} frame-fold result for one terrain segment; not a production/generalization metric",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# Surface Halo Ranker\n\n"
        f"{args.fold_mode} frame-fold ranker check for terrain surface-halo proposals.\n\n"
        f"Best model: `{best_name}`\n\n"
        "This proves whether the recovered candidates are learnable from the new labels; it is not a production metric.\n"
    )
    print(out_dir / "summary.csv")
    print(model_path)


if __name__ == "__main__":
    main()
