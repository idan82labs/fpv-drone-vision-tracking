#!/usr/bin/env python3
"""Train/evaluate a small crop-stack verifier on hard top-tube alternatives.

This is an offline observation-model probe, not a runtime path. It tests the
next hypothesis after scalar CLBA features stalled:

    can target-aligned/background-aligned crop stacks separate true tiny-drone
    tubes from the false tree/terrain/branch tubes that beat scalar rankers?

The script can either consume an existing ``hard_label`` candidate CSV or build
hard alternatives from frame labels plus top_tubes. It reports leave-one-clip-out
hard-example performance and pairwise frame win rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import profile_tube_alignment_features as align
    import train_surface_xy_ranker as xy_ranker
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import profile_tube_alignment_features as align
    from scripts import train_surface_xy_ranker as xy_ranker


SCALAR_COLUMNS = [
    "rank",
    "verified_score",
    "tube_verifier_score",
    "learned_score",
    "score",
    "cand_texture",
    "cand_line_context",
    "cand_attached_support",
    "cand_map_score",
    "cand_native_dark_score",
    "tube_mean_line_context",
    "tube_mean_attached_support",
    "tube_mean_texture",
    "tube_mean_pair_score",
    "tube_mean_pair_bg",
    "tube_mean_bg_dist",
    "tube_log_cand_density",
    "clba_gain_norm",
    "clba_target_likelihood",
    "clba_bg_static_likelihood",
    "clba_attached_likelihood",
]

SOURCE_CATEGORIES = ["motion", "map", "appearance", "temporal_stack", "large_dark", "hybrid_coast", ""]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--examples", nargs="+", help="Existing candidate CSVs with hard_label.")
    source.add_argument("--labels", help="Frame-label CSV used to build hard alternatives.")
    p.add_argument("--results_dir", help="Directory containing clip/top_tubes.csv files for --labels.")
    p.add_argument("--tube_csv", nargs="*", default=[], help="top_tubes CSVs used to recover track paths.")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--center_tol_px", type=float, default=8.0)
    p.add_argument("--negative_min_dist_px", type=float, default=24.0)
    p.add_argument("--negatives_per_frame", type=int, default=3)
    p.add_argument("--null_negatives_per_frame", type=int, default=3)
    p.add_argument("--confidence", nargs="*", default=["high", "medium_high"])
    p.add_argument("--null_confidence", nargs="*", default=["high_not_visible"])
    p.add_argument("--window_radius", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=31)
    p.add_argument("--patch_size", type=int, default=11)
    p.add_argument(
        "--source_geometry_features",
        action="store_true",
        help="Append candidate box geometry and candidate-source one-hot flags. Off by default for legacy model compatibility.",
    )
    p.add_argument(
        "--save_loco_models",
        action="store_true",
        help="Save one held-out-clip model bundle per model/clip for honest downstream replay.",
    )
    p.add_argument("--detector_scale", type=float, default=0.5)
    p.add_argument("--orb_features", type=int, default=900)
    p.add_argument("--min_matches", type=int, default=18)
    p.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "hist_gbdt", "pairwise_logistic"),
        default=["logistic", "hist_gbdt", "pairwise_logistic"],
    )
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--random_state", type=int, default=19)
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


def safe_float(value: Any, default: float = 0.0) -> float:
    return align.safe_float(value, default)


def safe_int(value: Any, default: int = 0) -> int:
    return align.safe_int(value, default)


def label_visible(row: dict[str, str]) -> bool:
    return xy_ranker.label_visible(row)


def clip_matches(a: str, b: str) -> bool:
    return xy_ranker.clip_matches(a, b)


def top_tubes_path(results_dir: Path, clip: str) -> Path | None:
    return xy_ranker.top_tubes_path(results_dir, clip)


def load_top_tubes(results_dir: Path, clip: str, max_rank: int) -> dict[int, list[dict[str, str]]]:
    return xy_ranker.load_top_tubes(results_dir, clip, max_rank)


def row_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return xy_ranker.row_bbox(row)


def label_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return xy_ranker.label_bbox(row)


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return xy_ranker.center_dist(a, b)


def dist_to_label(row: dict[str, str], label: dict[str, str]) -> float:
    return center_dist(row_bbox(row), label_bbox(label))


def build_examples_from_labels(
    labels_path: Path,
    results_dir: Path,
    max_rank: int,
    center_tol_px: float,
    negative_min_dist_px: float,
    negatives_per_frame: int,
    null_negatives_per_frame: int,
    confidences: set[str],
    null_confidences: set[str],
) -> list[dict[str, str]]:
    labels = read_csv(labels_path)
    clips = sorted({r["clip"] for r in labels if r.get("clip")})
    top_by_clip = {clip: load_top_tubes(results_dir, clip, max_rank) for clip in clips}
    examples: list[dict[str, str]] = []

    for lab in labels:
        clip = lab.get("clip", "")
        frame = safe_int(lab.get("frame"), -1)
        if not clip or frame < 0:
            continue
        rows = top_by_clip.get(clip, {}).get(frame, [])
        if not rows:
            continue
        if label_visible(lab):
            if confidences and lab.get("confidence") not in confidences:
                continue
            scored = [(dist_to_label(row, lab), row) for row in rows]
            positives = [(d, row) for d, row in scored if d <= center_tol_px]
            negatives = [(d, row) for d, row in scored if d >= negative_min_dist_px]
            if not positives or not negatives:
                continue
            pos_d, pos = min(positives, key=lambda item: (safe_int(item[1].get("rank"), 999999), item[0]))
            out = dict(pos)
            out.update(
                {
                    "clip": clip,
                    "hard_label": "1",
                    "hard_kind": "nearest_true",
                    "label_x": str(lab.get("det_x", lab.get("x", ""))),
                    "label_y": str(lab.get("det_y", lab.get("y", ""))),
                    "label_w": str(lab.get("det_w", lab.get("w", ""))),
                    "label_h": str(lab.get("det_h", lab.get("h", ""))),
                    "candidate_dist_px": round(pos_d, 3),
                }
            )
            examples.append(out)
            negatives_sorted = sorted(
                negatives,
                key=lambda item: (
                    str(item[1].get("selected", "0")) != "1",
                    -safe_float(item[1].get("learned_score"), safe_float(item[1].get("verified_score"), safe_float(item[1].get("score")))),
                    safe_int(item[1].get("rank"), 999999),
                ),
            )
            for neg_i, (dist, neg) in enumerate(negatives_sorted[: max(1, negatives_per_frame)]):
                out = dict(neg)
                out.update(
                    {
                        "clip": clip,
                        "hard_label": "0",
                        "hard_kind": "false_competitor",
                        "label_x": str(lab.get("det_x", lab.get("x", ""))),
                        "label_y": str(lab.get("det_y", lab.get("y", ""))),
                        "label_w": str(lab.get("det_w", lab.get("w", ""))),
                        "label_h": str(lab.get("det_h", lab.get("h", ""))),
                        "candidate_dist_px": round(dist, 3),
                        "negative_rank_in_frame": neg_i + 1,
                    }
                )
                examples.append(out)
        else:
            if null_confidences and lab.get("confidence") not in null_confidences:
                continue
            rows_sorted = sorted(
                rows,
                key=lambda row: (
                    -safe_float(row.get("learned_score"), safe_float(row.get("verified_score"), safe_float(row.get("score")))),
                    safe_int(row.get("rank"), 999999),
                ),
            )
            for neg_i, neg in enumerate(rows_sorted[: max(1, null_negatives_per_frame)]):
                out = dict(neg)
                out.update(
                    {
                        "clip": clip,
                        "hard_label": "0",
                        "hard_kind": "null_competitor",
                        "candidate_dist_px": "",
                        "negative_rank_in_frame": neg_i + 1,
                    }
                )
                examples.append(out)
    return examples


def load_examples(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for row in read_csv(path):
            if row.get("hard_label", "") in {"0", "1"} and row.get("clip") and row.get("frame"):
                rows.append(row)
    return rows


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    vals = crop.astype(np.float32)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = max(1.4826 * mad, 3.0)
    return np.clip((vals - med) / sigma, -4.0, 4.0)


def resize_patch(img: np.ndarray, patch_size: int) -> np.ndarray:
    import cv2

    if img.shape[0] == patch_size and img.shape[1] == patch_size:
        return img.astype(np.float32)
    return cv2.resize(img.astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_AREA)


def source_geometry_vector(row: dict[str, str]) -> np.ndarray:
    source = str(row.get("cand_source", "")).strip().lower()
    source_flags = [1.0 if source == cat else 0.0 for cat in SOURCE_CATEGORIES]
    source_other = 0.0 if source in SOURCE_CATEGORIES else 1.0
    area = safe_float(row.get("cand_area"), safe_float(row.get("w")) * safe_float(row.get("h")))
    geom = [
        safe_float(row.get("w")),
        safe_float(row.get("h")),
        area,
        safe_float(row.get("cand_aspect")),
        safe_float(row.get("cand_fill")),
        source_other,
    ]
    return np.asarray(geom + source_flags, dtype=np.float32)


def extract_stack_features(
    row: dict[str, str],
    frame_cache: align.FrameCache,
    transforms: align.TransformCache,
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    window_radius: int,
    crop_size: int,
    patch_size: int,
    source_geometry_features: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    clip = row.get("clip", "")
    frame = safe_int(row.get("frame"), -1)
    anchor = align.center(row)
    target_crops: list[np.ndarray] = []
    bg_crops: list[np.ndarray] = []
    path_bg_dist: list[float] = []
    failures = 0
    for target_frame in range(frame - window_radius, frame + 1):
        gray = frame_cache.gray(clip, target_frame)
        if gray is None:
            continue
        target_pt = align.target_point(row, tube_rows, clip, frame, target_frame)
        mat = transforms.transform(clip, frame, target_frame)
        bg_pt = align.project(mat, anchor)
        if np.allclose(mat, np.eye(3), atol=1e-6) and target_frame != frame:
            failures += 1
        target_crops.append(align.extract_crop(gray, target_pt, crop_size))
        bg_crops.append(align.extract_crop(gray, bg_pt, crop_size))
        path_bg_dist.append(float(math.hypot(target_pt[0] - bg_pt[0], target_pt[1] - bg_pt[1])))
    tq = align.stack_quality(target_crops, crop_size)
    bq = align.stack_quality(bg_crops, crop_size)
    if target_crops:
        t_arr = np.stack([normalize_crop(c) for c in target_crops]).astype(np.float32)
        t_mean = np.mean(t_arr, axis=0)
        t_std = np.std(t_arr, axis=0)
    else:
        t_mean = np.zeros((crop_size, crop_size), dtype=np.float32)
        t_std = np.zeros((crop_size, crop_size), dtype=np.float32)
    if bg_crops:
        b_arr = np.stack([normalize_crop(c) for c in bg_crops]).astype(np.float32)
        b_mean = np.mean(b_arr, axis=0)
        b_std = np.std(b_arr, axis=0)
    else:
        b_mean = np.zeros((crop_size, crop_size), dtype=np.float32)
        b_std = np.zeros((crop_size, crop_size), dtype=np.float32)
    diff = t_mean - b_mean
    patch_parts = [
        resize_patch(t_mean, patch_size),
        resize_patch(b_mean, patch_size),
        resize_patch(diff, patch_size),
        resize_patch(t_std - b_std, patch_size),
    ]
    scalar = np.asarray(
        [
            tq["q"],
            bq["q"],
            tq["q"] - bq["q"],
            tq["stack_dark_z"],
            bq["stack_dark_z"],
            tq["anisotropy"],
            bq["anisotropy"],
            float(np.mean(path_bg_dist)) if path_bg_dist else 0.0,
            float(np.max(path_bg_dist)) if path_bg_dist else 0.0,
            float(len(target_crops)),
            float(failures),
            *[safe_float(row.get(name)) for name in SCALAR_COLUMNS],
        ],
        dtype=np.float32,
    )
    vector_parts = [p.reshape(-1) for p in patch_parts] + [scalar]
    if source_geometry_features:
        vector_parts.append(source_geometry_vector(row))
    vector = np.concatenate(vector_parts)
    meta = {
        "crop_target_q": round(float(tq["q"]), 6),
        "crop_bg_q": round(float(bq["q"]), 6),
        "crop_gain": round(float(tq["q"] - bq["q"]), 6),
        "crop_target_stack_dark_z": round(float(tq["stack_dark_z"]), 6),
        "crop_bg_stack_dark_z": round(float(bq["stack_dark_z"]), 6),
        "crop_path_bg_dist_mean": round(float(np.mean(path_bg_dist)) if path_bg_dist else 0.0, 6),
        "crop_used_frames": len(target_crops),
        "crop_transform_failures": failures,
    }
    return vector, meta


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.18, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed)),
            ]
        ),
        "hist_gbdt": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.035, max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1.0, random_state=seed)),
            ]
        ),
        "pairwise_logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.08, max_iter=2000, solver="liblinear", random_state=seed)),
            ]
        ),
    }


def predict_model_score(model: Pipeline, x: np.ndarray, score_mode: str = "auto") -> np.ndarray:
    if score_mode == "decision_function":
        return model.decision_function(x)
    if score_mode == "probability":
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def predict_score(model: Pipeline, x: np.ndarray) -> np.ndarray:
    return predict_model_score(model, x)


def pairwise_training_data(rows: list[dict[str, Any]], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    by_frame: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_frame[(str(row.get("clip", "")), safe_int(row.get("frame"), -1))].append(idx)
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    for idxs in by_frame.values():
        pos = [i for i in idxs if safe_int(rows[i].get("hard_label"), 0) == 1]
        neg = [i for i in idxs if safe_int(rows[i].get("hard_label"), 0) == 0]
        for p_idx in pos:
            for n_idx in neg:
                diffs.append(x[p_idx] - x[n_idx])
                labels.append(1)
                diffs.append(x[n_idx] - x[p_idx])
                labels.append(0)
    if not diffs:
        return np.empty((0, x.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.vstack(diffs).astype(np.float32), np.asarray(labels, dtype=np.int32)


def fit_verifier_model(model_name: str, model: Pipeline, x_train: np.ndarray, y_train: np.ndarray, train_rows: list[dict[str, Any]]) -> Pipeline:
    if model_name != "pairwise_logistic":
        model.fit(x_train, y_train)
        return model
    x_pair, y_pair = pairwise_training_data(train_rows, x_train)
    if x_pair.shape[0] == 0 or len(set(y_pair.tolist())) < 2:
        model.fit(x_train, y_train)
        return model
    model.fit(x_pair, y_pair)
    return model


def score_mode_for_model(model_name: str) -> str:
    return "decision_function" if model_name == "pairwise_logistic" else "auto"


def model_bundle(
    model: Pipeline,
    model_name: str,
    args: argparse.Namespace,
    *,
    heldout_clip: str | None = None,
) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "model": model,
        "best_model_loco": model_name,
        "window_radius": args.window_radius,
        "crop_size": args.crop_size,
        "patch_size": args.patch_size,
        "detector_scale": args.detector_scale,
        "scalar_columns": SCALAR_COLUMNS,
        "score_mode": score_mode_for_model(model_name),
        "source_geometry_features": bool(args.source_geometry_features),
        "source_categories": SOURCE_CATEGORIES,
    }
    if heldout_clip is not None:
        bundle["heldout_clip"] = heldout_clip
        bundle["training_mode"] = "leave_one_clip_out"
    return bundle


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    wins = 0.0
    total = 0
    for p in pos:
        total += int(neg.size)
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / max(1, total)


def pairwise_summary(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[(str(row.get("clip", "")), safe_int(row.get("frame"), -1))].append(row)
    wins = 0
    ties = 0
    total = 0
    positive_frames = 0
    for group in by_frame.values():
        pos = [r for r in group if safe_int(r.get("hard_label"), 0) == 1]
        neg = [r for r in group if safe_int(r.get("hard_label"), 0) == 0]
        if pos and neg:
            positive_frames += 1
        for p in pos:
            ps = safe_float(p.get(score_key))
            for n in neg:
                ns = safe_float(n.get(score_key))
                total += 1
                if ps > ns:
                    wins += 1
                elif ps == ns:
                    ties += 1
    return {
        "pairwise_wins": wins,
        "pairwise_ties": ties,
        "pairwise_total": total,
        "pairwise_win_rate": round((wins + 0.5 * ties) / max(1, total), 6),
        "positive_frames_with_negatives": positive_frames,
    }


def summarize_predictions(rows: list[dict[str, Any]], score_key: str, model: str) -> dict[str, Any]:
    y = np.asarray([safe_int(r.get("hard_label"), 0) for r in rows], dtype=np.int32)
    s = np.asarray([safe_float(r.get(score_key)) for r in rows], dtype=np.float32)
    pair = pairwise_summary(rows, score_key)
    pos = [r for r in rows if safe_int(r.get("hard_label"), 0) == 1]
    neg = [r for r in rows if safe_int(r.get("hard_label"), 0) == 0]
    return {
        "model": model,
        "rows": len(rows),
        "positives": len(pos),
        "negatives": len(neg),
        "auc": round(auc_score(y, s), 6),
        "pos_mean": round(float(np.mean([safe_float(r.get(score_key)) for r in pos])) if pos else 0.0, 6),
        "neg_mean": round(float(np.mean([safe_float(r.get(score_key)) for r in neg])) if neg else 0.0, 6),
        **pair,
    }


def summarize_by_clip(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[str(row.get("clip", ""))].append(row)
    out: list[dict[str, Any]] = []
    for clip, group in sorted(by_clip.items()):
        rec = summarize_predictions(group, score_key, str(group[0].get("model", "")) if group else "")
        rec["clip"] = clip
        out.append(rec)
    return out


def pair_failures(rows: list[dict[str, Any]], score_key: str, model_name: str) -> list[dict[str, Any]]:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("model") == model_name:
            by_frame[(str(row.get("clip", "")), safe_int(row.get("frame"), -1))].append(row)
    failures: list[dict[str, Any]] = []
    for (clip, frame), group in sorted(by_frame.items()):
        positives = [r for r in group if safe_int(r.get("hard_label"), 0) == 1]
        negatives = [r for r in group if safe_int(r.get("hard_label"), 0) == 0]
        if not positives or not negatives:
            continue
        pos = max(positives, key=lambda r: safe_float(r.get(score_key)))
        neg = max(negatives, key=lambda r: safe_float(r.get(score_key)))
        pos_score = safe_float(pos.get(score_key))
        neg_score = safe_float(neg.get(score_key))
        if pos_score > neg_score:
            continue
        failures.append(
            {
                "clip": clip,
                "frame": frame,
                "model": model_name,
                "score_gap_pos_minus_neg": round(pos_score - neg_score, 6),
                "pos_score": round(pos_score, 6),
                "pos_rank": pos.get("rank", ""),
                "pos_source": pos.get("cand_source", ""),
                "pos_x": pos.get("x", ""),
                "pos_y": pos.get("y", ""),
                "pos_w": pos.get("w", ""),
                "pos_h": pos.get("h", ""),
                "pos_candidate_dist_px": pos.get("candidate_dist_px", ""),
                "neg_score": round(neg_score, 6),
                "neg_rank": neg.get("rank", ""),
                "neg_source": neg.get("cand_source", ""),
                "neg_kind": neg.get("hard_kind", ""),
                "neg_x": neg.get("x", ""),
                "neg_y": neg.get("y", ""),
                "neg_w": neg.get("w", ""),
                "neg_h": neg.get("h", ""),
                "neg_candidate_dist_px": neg.get("candidate_dist_px", ""),
            }
        )
    failures.sort(key=lambda r: (r["clip"], r["frame"], r["score_gap_pos_minus_neg"]))
    return failures


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.examples:
        examples = load_examples([Path(p) for p in args.examples])
        tube_paths = [Path(p) for p in args.tube_csv]
    else:
        if not args.results_dir:
            raise SystemExit("--results_dir is required with --labels")
        examples = build_examples_from_labels(
            Path(args.labels),
            Path(args.results_dir),
            args.max_rank,
            args.center_tol_px,
            args.negative_min_dist_px,
            args.negatives_per_frame,
            args.null_negatives_per_frame,
            set(args.confidence or []),
            set(args.null_confidence or []),
        )
        clips = sorted({r.get("clip", "") for r in examples if r.get("clip")})
        tube_paths = []
        for clip in clips:
            path = top_tubes_path(Path(args.results_dir), clip)
            if path is not None:
                tube_paths.append(path)
    if args.max_examples and len(examples) > args.max_examples:
        rng = np.random.default_rng(args.random_state)
        idx = np.sort(rng.choice(len(examples), size=args.max_examples, replace=False))
        examples = [examples[int(i)] for i in idx]
    if not examples:
        raise SystemExit("no hard examples")

    tube_rows = align.load_tube_rows(tube_paths)
    frame_cache = align.FrameCache(Path(args.video_dir), args.detector_scale)
    transform_cache = align.TransformCache(frame_cache, args.orb_features, args.min_matches)
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    try:
        for row in examples:
            vector, meta = extract_stack_features(
                row,
                frame_cache,
                transform_cache,
                tube_rows,
                args.window_radius,
                args.crop_size,
                args.patch_size,
                args.source_geometry_features,
            )
            out = dict(row)
            out.update(meta)
            vectors.append(vector)
            rows.append(out)
    finally:
        frame_cache.close()

    x = np.vstack(vectors).astype(np.float32)
    y = np.asarray([safe_int(r.get("hard_label"), 0) for r in rows], dtype=np.int32)
    clips = sorted({str(r.get("clip", "")) for r in rows})
    models = make_models(args.random_state)
    prediction_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for model_name in args.models:
        model_rows: list[dict[str, Any]] = []
        for held_clip in clips:
            train_idx = np.asarray([i for i, row in enumerate(rows) if row.get("clip") != held_clip], dtype=int)
            test_idx = np.asarray([i for i, row in enumerate(rows) if row.get("clip") == held_clip], dtype=int)
            if train_idx.size == 0 or test_idx.size == 0 or len(set(y[train_idx])) < 2:
                continue
            model = clone(models[model_name])
            train_rows = [rows[int(idx)] for idx in train_idx]
            model = fit_verifier_model(model_name, model, x[train_idx], y[train_idx], train_rows)
            if args.save_loco_models:
                loco_dir = out_dir / "loco_models" / model_name
                loco_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    model_bundle(model, model_name, args, heldout_clip=held_clip),
                    loco_dir / f"{held_clip}.joblib",
                )
            scores = predict_model_score(model, x[test_idx], score_mode_for_model(model_name))
            for idx, score in zip(test_idx, scores):
                out = dict(rows[int(idx)])
                out["crop_stack_score"] = round(float(score), 6)
                out["model"] = model_name
                model_rows.append(out)
                prediction_rows.append(out)
        summary.append(summarize_predictions(model_rows, "crop_stack_score", model_name))

    best_model_name = max(summary, key=lambda r: (r["pairwise_win_rate"], r["auc"]))["model"] if summary else args.models[0]
    final_model = clone(models[best_model_name])
    final_model = fit_verifier_model(best_model_name, final_model, x, y, rows)
    model_path = out_dir / f"{best_model_name}_crop_stack_verifier.joblib"
    joblib.dump(
        model_bundle(final_model, best_model_name, args),
        model_path,
    )

    write_csv(out_dir / "hard_examples_with_crop_features.csv", rows)
    write_csv(out_dir / "loco_predictions.csv", prediction_rows)
    write_csv(out_dir / "loco_summary.csv", summary)
    for model_name in args.models:
        model_rows = [r for r in prediction_rows if r.get("model") == model_name]
        if model_rows:
            write_csv(out_dir / f"{model_name}_by_clip.csv", summarize_by_clip(model_rows, "crop_stack_score"))
            write_csv(out_dir / f"{model_name}_pair_failures.csv", pair_failures(model_rows, "crop_stack_score", model_name))
    metadata = {
        "examples": len(rows),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "clips": clips,
        "feature_dim": int(x.shape[1]),
        "examples_source": args.examples or args.labels,
        "results_dir": args.results_dir,
        "tube_csv": [str(p) for p in tube_paths],
        "video_dir": args.video_dir,
        "best_model_loco": best_model_name,
        "model_path": str(model_path),
        "score_mode": score_mode_for_model(best_model_name),
        "source_geometry_features": bool(args.source_geometry_features),
        "metric_caveat": "hard-alternative separation only; not full selected-box tracking accuracy",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# Crop-Stack Verifier Probe\n\n"
        "Offline hard-alternative target/background crop-stack verifier.\n\n"
        f"Examples: `{len(rows)}`\n\n"
        f"Best LOCO model: `{best_model_name}`\n\n"
        "See `loco_summary.csv`, `loco_predictions.csv`, and `metadata.json`.\n"
    )
    print(out_dir / "loco_summary.csv")
    print(model_path)


if __name__ == "__main__":
    main()
