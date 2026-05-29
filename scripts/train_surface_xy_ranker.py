#!/usr/bin/env python3
"""Train/evaluate a multi-clip XY tube ranker for hard surface backgrounds."""

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
    "cand_frame",
    "cand_is_current",
    "candidate_frame",
    "candidate_is_current",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--center_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--negative_min_dist_px", type=float, default=24.0)
    p.add_argument("--confidence", nargs="*", default=["high", "medium_high"])
    p.add_argument(
        "--include_null_frames",
        action="store_true",
        help="Add candidates from visible=0 label frames as negative training examples.",
    )
    p.add_argument(
        "--null_confidence",
        nargs="*",
        default=["high_not_visible"],
        help="Confidence values accepted for null-frame negative examples.",
    )
    p.add_argument("--models", nargs="+", choices=("logistic", "hist_gbdt", "extra_trees"), default=["logistic", "hist_gbdt", "extra_trees"])
    p.add_argument("--fallback_thresholds", default="0:1:0.02", help="Threshold sweep for learned-score fallback rules: start:end:step or comma list.")
    p.add_argument(
        "--fallback_gates",
        default="none,learned_not_map,source_large_dark_or_appearance,high_support,high_texture_support,large_dark_high_support,low_sky_high_support,negative_bg_pair,support_negative_bg_pair",
        help="Comma-separated learned-candidate gates to sweep for state-specific fallback.",
    )
    p.add_argument(
        "--final_exclude_clip",
        nargs="*",
        default=[],
        help="Optional clip ids to exclude when fitting the saved deployment model.",
    )
    p.add_argument(
        "--extra_examples",
        default="",
        help=(
            "Optional candidate-level hard examples CSV for the saved final model. "
            "Rows must contain hard_label plus the same candidate/tube feature columns."
        ),
    )
    p.add_argument(
        "--extra_weight",
        type=int,
        default=1,
        help=(
            "Integer repeat weight for --extra_examples in the saved final model. "
            "This only affects the deployment model, not LOCO evaluation."
        ),
    )
    p.add_argument("--random_state", type=int, default=17)
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


def float_or_default(value: Any, default: float) -> float:
    out = safe_float(value)
    return default if out is None else out


def int_or_default(value: Any, default: int) -> int:
    out = safe_float(value)
    return default if out is None else int(out)


def label_visible(row: dict[str, str]) -> bool:
    raw = str(row.get("visible", "")).strip().lower()
    if raw in {"", "1", "true", "yes", "visible", "target"}:
        return True
    if raw in {"0", "false", "no", "empty", "none", "not_visible", "not visible"}:
        return False
    return True


def clip_matches(a: str, b: str) -> bool:
    return a == b or a.startswith(b) or b.startswith(a)


def top_tubes_path(results_dir: Path, clip: str) -> Path | None:
    direct = results_dir / clip / "top_tubes.csv"
    if direct.exists():
        return direct
    for path in results_dir.glob("*/top_tubes.csv"):
        if clip_matches(clip, path.parent.name):
            return path
    return None


def load_top_tubes(results_dir: Path, clip: str, max_rank: int) -> dict[int, list[dict[str, str]]]:
    path = top_tubes_path(results_dir, clip)
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    if path is None:
        return by_frame
    for row in read_csv(path):
        rank = int_or_default(row.get("rank"), 999)
        if rank > max_rank:
            continue
        frame = int_or_default(row.get("frame"), -1)
        if frame >= 0:
            by_frame[frame].append(row)
    return by_frame


def row_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float_or_default(row.get("x"), 0.0),
        float_or_default(row.get("y"), 0.0),
        float_or_default(row.get("w"), 1.0),
        float_or_default(row.get("h"), 1.0),
    )


def label_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float_or_default(row.get("det_x", row.get("x")), 0.0),
        float_or_default(row.get("det_y", row.get("y")), 0.0),
        float_or_default(row.get("det_w", row.get("w")), 1.0),
        float_or_default(row.get("det_h", row.get("h")), 1.0),
    )


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return float(math.hypot(ax - bx, ay - by))


def dist_to_label(row: dict[str, str], lab: dict[str, str]) -> float:
    return center_dist(row_bbox(row), label_bbox(lab))


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
    return numeric, [f"src_{src}" for src in sorted(sources)]


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


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.32, class_weight="balanced", max_iter=2500, solver="liblinear", random_state=seed)),
            ]
        ),
        "hist_gbdt": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.035, max_leaf_nodes=6, min_samples_leaf=14, l2_regularization=1.0, random_state=seed)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=350, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=seed)),
            ]
        ),
    }


def predict_score(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def summarize(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    total = len(rows)
    oracle = sum(bool(r["oracle_hit"]) for r in rows)
    strict = sum(bool(r["strict_hit"]) for r in rows)
    loose = sum(bool(r["loose_hit"]) for r in rows)
    high = [r for r in rows if r.get("confidence") == "high"]
    return {
        "model": model,
        "frames": total,
        "oracle_recall": round(oracle / max(1, total), 4),
        "strict_hit": strict,
        "strict_recall": round(strict / max(1, total), 4),
        "loose_hit": loose,
        "loose_recall": round(loose / max(1, total), 4),
        "high_frames": len(high),
        "high_strict_recall": round(sum(bool(r["strict_hit"]) for r in high) / max(1, len(high)), 4),
        "high_loose_recall": round(sum(bool(r["loose_hit"]) for r in high) / max(1, len(high)), 4),
    }


def parse_thresholds(raw: str) -> list[float]:
    if ":" in raw:
        start_s, end_s, step_s = raw.split(":", 2)
        start = float(start_s)
        end = float(end_s)
        step = float(step_s)
        if step <= 0:
            raise SystemExit("--fallback_thresholds step must be positive")
        vals: list[float] = []
        value = start
        while value <= end + 1e-9:
            vals.append(round(value, 6))
            value += step
        return vals
    return [float(x) for x in raw.split(",") if x.strip()]


def fallback_rows(
    predictions: list[dict[str, Any]],
    model_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        key = (str(row.get("clip", "")), int_or_default(row.get("frame"), 0))
        by_key[key][str(row.get("model", ""))] = row

    rows: list[dict[str, Any]] = []
    for key in sorted(by_key):
        pair = by_key[key]
        baseline = pair.get("baseline_verified_score")
        learned = pair.get(model_name)
        if baseline is None or learned is None:
            continue
        score_value = safe_float(learned.get("score"), -1e9)
        score = -1e9 if score_value is None else score_value
        chosen = learned if score >= threshold else baseline
        rec = dict(chosen)
        rec["model"] = f"fallback_{model_name}_thr{threshold:.2f}"
        rec["fallback_model"] = model_name
        rec["fallback_threshold"] = round(threshold, 6)
        rec["fallback_used_learned"] = chosen is learned
        rec["learned_score"] = round(score, 6)
        rows.append(rec)
    return rows


def parse_gates(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def top_row_for_prediction(
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    row: dict[str, Any],
) -> dict[str, str] | None:
    clip = str(row.get("clip", ""))
    frame = int_or_default(row.get("frame"), 0)
    rank = str(row.get("rank", ""))
    for cand in top_by_clip.get(clip, {}).get(frame, []):
        if str(cand.get("rank", "")) == rank:
            return cand
    return None


def gate_float(row: dict[str, str] | None, name: str, default: float = 0.0) -> float:
    if row is None:
        return default
    value = safe_float(row.get(name), default)
    return default if value is None else value


def gate_source(row: dict[str, str] | None) -> str:
    return "" if row is None else str(row.get("cand_source", ""))


def gate_allows(row: dict[str, str] | None, gate: str) -> bool:
    if gate == "none":
        return True
    source = gate_source(row)
    cand_support = gate_float(row, "cand_attached_support")
    tube_support = gate_float(row, "tube_mean_attached_support")
    support = max(cand_support, tube_support)
    cand_texture = gate_float(row, "cand_texture")
    tube_texture = gate_float(row, "tube_mean_texture")
    texture = max(cand_texture, tube_texture)
    sky = max(gate_float(row, "cand_sky_like"), gate_float(row, "tube_mean_sky_like"))
    pair_bg = gate_float(row, "tube_mean_pair_bg")
    if gate == "learned_not_map":
        return source != "map"
    if gate == "source_large_dark_or_appearance":
        return source in {"large_dark", "appearance"}
    if gate == "high_support":
        return support >= 8.0
    if gate == "high_texture_support":
        return texture >= 55.0 and support >= 6.0
    if gate == "large_dark_high_support":
        return source == "large_dark" and support >= 6.0
    if gate == "low_sky_high_support":
        return sky <= 0.02 and support >= 6.0
    if gate == "negative_bg_pair":
        return pair_bg <= 0.0
    if gate == "support_negative_bg_pair":
        return support >= 6.0 and pair_bg <= 0.0
    raise SystemExit(f"unknown fallback gate: {gate}")


def gated_fallback_rows(
    predictions: list[dict[str, Any]],
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    model_name: str,
    threshold: float,
    gate: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        key = (str(row.get("clip", "")), int_or_default(row.get("frame"), 0))
        by_key[key][str(row.get("model", ""))] = row

    rows: list[dict[str, Any]] = []
    for key in sorted(by_key):
        pair = by_key[key]
        baseline = pair.get("baseline_verified_score")
        learned = pair.get(model_name)
        if baseline is None or learned is None:
            continue
        learned_top = top_row_for_prediction(top_by_clip, learned)
        score_value = safe_float(learned.get("score"), -1e9)
        score = -1e9 if score_value is None else score_value
        allowed = gate_allows(learned_top, gate)
        chosen = learned if score >= threshold and allowed else baseline
        rec = dict(chosen)
        rec["model"] = f"gated_{gate}_{model_name}_thr{threshold:.2f}"
        rec["fallback_model"] = model_name
        rec["fallback_threshold"] = round(threshold, 6)
        rec["fallback_gate"] = gate
        rec["fallback_gate_allowed"] = allowed
        rec["fallback_used_learned"] = chosen is learned
        rec["learned_score"] = round(score, 6)
        rows.append(rec)
    return rows


def nested_fallback_rows(
    predictions: list[dict[str, Any]],
    model_names: list[str],
    thresholds: list[float],
    clips: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select fallback model/threshold without looking at the held-out clip."""
    fallback_by_config: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for model_name in model_names:
        for threshold in thresholds:
            fallback_by_config[(model_name, threshold)] = fallback_rows(predictions, model_name, threshold)

    nested_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for held_clip in clips:
        best_key: tuple[str, float] | None = None
        best_summary: dict[str, Any] | None = None
        for key, rows in fallback_by_config.items():
            train_rows = [r for r in rows if r.get("clip") != held_clip]
            if not train_rows:
                continue
            rec = summarize(train_rows, f"nested_train_{key[0]}_thr{key[1]:.2f}")
            if best_summary is None or (
                rec["strict_recall"],
                rec["loose_recall"],
                rec["high_strict_recall"],
            ) > (
                best_summary["strict_recall"],
                best_summary["loose_recall"],
                best_summary["high_strict_recall"],
            ):
                best_key = key
                best_summary = rec
        if best_key is None or best_summary is None:
            continue
        model_name, threshold = best_key
        test_rows = [dict(r) for r in fallback_by_config[best_key] if r.get("clip") == held_clip]
        for row in test_rows:
            row["model"] = "nested_fallback"
            row["nested_model"] = model_name
            row["nested_threshold"] = round(threshold, 6)
        nested_rows.extend(test_rows)
        test_summary = summarize(test_rows, f"nested_test_{held_clip}")
        selection_rows.append(
            {
                "heldout_clip": held_clip,
                "selected_model": model_name,
                "selected_threshold": round(threshold, 6),
                "train_strict_recall": best_summary["strict_recall"],
                "train_loose_recall": best_summary["loose_recall"],
                "train_high_strict_recall": best_summary["high_strict_recall"],
                "test_frames": test_summary["frames"],
                "test_strict_recall": test_summary["strict_recall"],
                "test_loose_recall": test_summary["loose_recall"],
                "test_high_strict_recall": test_summary["high_strict_recall"],
            }
        )
    return nested_rows, selection_rows


def nested_gated_fallback_rows(
    predictions: list[dict[str, Any]],
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    model_names: list[str],
    thresholds: list[float],
    gates: list[str],
    clips: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    fallback_by_config: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for model_name in model_names:
        for threshold in thresholds:
            for gate in gates:
                fallback_by_config[(model_name, threshold, gate)] = gated_fallback_rows(
                    predictions,
                    top_by_clip,
                    model_name,
                    threshold,
                    gate,
                )

    nested_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for held_clip in clips:
        best_key: tuple[str, float, str] | None = None
        best_summary: dict[str, Any] | None = None
        for key, rows in fallback_by_config.items():
            train_rows = [r for r in rows if r.get("clip") != held_clip]
            if not train_rows:
                continue
            rec = summarize(train_rows, f"nested_gate_train_{key[0]}_{key[2]}_thr{key[1]:.2f}")
            if best_summary is None or (
                rec["strict_recall"],
                rec["loose_recall"],
                rec["high_strict_recall"],
            ) > (
                best_summary["strict_recall"],
                best_summary["loose_recall"],
                best_summary["high_strict_recall"],
            ):
                best_key = key
                best_summary = rec
        if best_key is None or best_summary is None:
            continue
        model_name, threshold, gate = best_key
        test_rows = [dict(r) for r in fallback_by_config[best_key] if r.get("clip") == held_clip]
        for row in test_rows:
            row["model"] = "nested_gated_fallback"
            row["nested_model"] = model_name
            row["nested_threshold"] = round(threshold, 6)
            row["nested_gate"] = gate
        nested_rows.extend(test_rows)
        test_summary = summarize(test_rows, f"nested_gate_test_{held_clip}")
        selection_rows.append(
            {
                "heldout_clip": held_clip,
                "selected_model": model_name,
                "selected_threshold": round(threshold, 6),
                "selected_gate": gate,
                "train_strict_recall": best_summary["strict_recall"],
                "train_loose_recall": best_summary["loose_recall"],
                "train_high_strict_recall": best_summary["high_strict_recall"],
                "test_frames": test_summary["frames"],
                "test_strict_recall": test_summary["strict_recall"],
                "test_loose_recall": test_summary["loose_recall"],
                "test_high_strict_recall": test_summary["high_strict_recall"],
            }
        )
    summary = summarize(nested_rows, "nested_gated_fallback") if nested_rows else None
    if summary:
        summary["selection"] = "per-heldout-clip model/threshold/gate selected on other clips"
    return nested_rows, selection_rows, summary


def summarize_by_clip(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[str(row.get("clip", ""))].append(row)
    out: list[dict[str, Any]] = []
    for clip, group in sorted(by_clip.items()):
        rec = summarize(group, model)
        rec["clip"] = clip
        out.append(rec)
    return out


def evaluate_by_score(
    labels: list[dict[str, str]],
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    score_name: str,
    center_tol: float,
    loose_tol: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lab in labels:
        frame = int_or_default(lab.get("frame"), 0)
        rows = top_by_clip.get(lab["clip"], {}).get(frame, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: float_or_default(r.get(score_name), -1e9))
        d = dist_to_label(best, lab)
        oracle_d = min((dist_to_label(r, lab) for r in rows), default=float("inf"))
        out.append(
            {
                "model": f"baseline_{score_name}",
                "clip": lab["clip"],
                "frame": frame,
                "confidence": lab.get("confidence", ""),
                "rank": best.get("rank", ""),
                "score": safe_float(best.get(score_name), ""),
                "source": best.get("cand_source", ""),
                "dist_px": round(d, 3),
                "strict_hit": d <= center_tol,
                "loose_hit": d <= loose_tol,
                "oracle_hit": oracle_d <= center_tol,
            }
        )
    return out


def load_extra_examples(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    y_vals: list[int] = []
    if not path.exists():
        raise SystemExit(f"--extra_examples not found: {path}")
    for row in read_csv(path):
        raw = row.get("hard_label", row.get("y", ""))
        if raw == "":
            continue
        label = int(float(raw))
        if label not in {0, 1}:
            continue
        rows.append(row)
        y_vals.append(label)
    return rows, np.asarray(y_vals, dtype=np.int32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_labels = read_csv(Path(args.labels))
    labels = [r for r in raw_labels if label_visible(r)]
    if args.confidence:
        labels = [r for r in labels if r.get("confidence") in set(args.confidence)]
    null_labels: list[dict[str, str]] = []
    if args.include_null_frames:
        null_labels = [r for r in raw_labels if not label_visible(r)]
        if args.null_confidence:
            null_labels = [r for r in null_labels if r.get("confidence") in set(args.null_confidence)]
    labels.sort(key=lambda r: (r.get("clip", ""), int_or_default(r.get("frame"), 0)))
    null_labels.sort(key=lambda r: (r.get("clip", ""), int_or_default(r.get("frame"), 0)))
    clips = sorted({r["clip"] for r in labels + null_labels})
    top_by_clip = {clip: load_top_tubes(Path(args.results_dir), clip, args.max_rank) for clip in clips}

    examples: list[dict[str, Any]] = []
    all_rows_for_features: list[dict[str, str]] = []
    ignored_near = 0
    missing_frames = 0
    null_examples = 0
    null_missing_frames = 0
    for lab in labels:
        frame = int_or_default(lab.get("frame"), 0)
        rows = top_by_clip.get(lab["clip"], {}).get(frame, [])
        if not rows:
            missing_frames += 1
            continue
        for row in rows:
            all_rows_for_features.append(row)
            d = dist_to_label(row, lab)
            if d <= args.center_tol_px:
                y = 1
            elif d >= args.negative_min_dist_px:
                y = 0
            else:
                ignored_near += 1
                continue
            examples.append({"clip": lab["clip"], "frame": frame, "label": lab, "row": row, "dist_px": d, "y": y})
    for lab in null_labels:
        frame = int_or_default(lab.get("frame"), 0)
        rows = top_by_clip.get(lab["clip"], {}).get(frame, [])
        if not rows:
            null_missing_frames += 1
            continue
        for row in rows:
            all_rows_for_features.append(row)
            examples.append({"clip": lab["clip"], "frame": frame, "label": lab, "row": row, "dist_px": "", "y": 0})
            null_examples += 1
    if not examples:
        raise SystemExit("no examples")

    numeric, sources = infer_features(all_rows_for_features)
    x = vectorize([e["row"] for e in examples], numeric, sources)
    y = np.asarray([int(e["y"]) for e in examples], dtype=np.int32)
    ex_clips = np.asarray([e["clip"] for e in examples], dtype=object)

    predictions: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    baseline = evaluate_by_score(labels, top_by_clip, "verified_score", args.center_tol_px, args.loose_tol_px)
    predictions.extend(baseline)
    summary.append(summarize(baseline, "baseline_verified_score"))
    thresholds = parse_thresholds(args.fallback_thresholds)
    gates = parse_gates(args.fallback_gates)

    models = make_models(args.random_state)
    fallback_summaries: list[dict[str, Any]] = []
    fallback_predictions_by_name: dict[str, list[dict[str, Any]]] = {}
    for model_name in args.models:
        model_rows: list[dict[str, Any]] = []
        for held_clip in clips:
            train_idx = np.asarray([i for i, clip in enumerate(ex_clips) if clip != held_clip], dtype=int)
            if len(train_idx) == 0 or len(set(y[train_idx])) < 2:
                continue
            model = Pipeline(models[model_name].steps)
            model.fit(x[train_idx], y[train_idx])
            for lab in [r for r in labels if r["clip"] == held_clip]:
                frame = int_or_default(lab.get("frame"), 0)
                rows = top_by_clip.get(held_clip, {}).get(frame, [])
                if not rows:
                    continue
                scores = predict_score(model, vectorize(rows, numeric, sources))
                best_i = int(np.argmax(scores))
                best = rows[best_i]
                d = dist_to_label(best, lab)
                oracle_d = min((dist_to_label(r, lab) for r in rows), default=float("inf"))
                rec = {
                    "model": model_name,
                    "clip": held_clip,
                    "frame": frame,
                    "confidence": lab.get("confidence", ""),
                    "rank": best.get("rank", ""),
                    "score": round(float(scores[best_i]), 6),
                    "source": best.get("cand_source", ""),
                    "dist_px": round(d, 3),
                    "strict_hit": d <= args.center_tol_px,
                    "loose_hit": d <= args.loose_tol_px,
                    "oracle_hit": oracle_d <= args.center_tol_px,
                }
                model_rows.append(rec)
                predictions.append(rec)
        summary.append(summarize(model_rows, model_name))

    for model_name in args.models:
        for threshold in thresholds:
            rows = fallback_rows(predictions, model_name, threshold)
            if not rows:
                continue
            rec = summarize(rows, f"fallback_{model_name}_thr{threshold:.2f}")
            rec["fallback_model"] = model_name
            rec["fallback_threshold"] = round(threshold, 6)
            rec["used_learned_frames"] = sum(bool(r.get("fallback_used_learned")) for r in rows)
            rec["used_learned_rate"] = round(rec["used_learned_frames"] / max(1, len(rows)), 4)
            fallback_summaries.append(rec)
            fallback_predictions_by_name[str(rec["model"])] = rows

    nested_rows, nested_selection_rows = nested_fallback_rows(predictions, args.models, thresholds, clips)
    nested_fallback_summary = summarize(nested_rows, "nested_fallback") if nested_rows else None
    if nested_fallback_summary:
        nested_fallback_summary["selection"] = "per-heldout-clip model/threshold selected on other clips"
    nested_gated_rows, nested_gated_selection_rows, nested_gated_fallback_summary = nested_gated_fallback_rows(
        predictions,
        top_by_clip,
        args.models,
        thresholds,
        gates,
        clips,
    )

    best_direct = max(summary[1:], key=lambda r: (r["strict_recall"], r["high_strict_recall"]))["model"]
    best_fallback = max(fallback_summaries, key=lambda r: (r["strict_recall"], r["loose_recall"], r["high_strict_recall"]), default=None)
    best_name = best_direct
    final_model = make_models(args.random_state)[best_name]
    final_exclude = set(args.final_exclude_clip or [])
    final_train_idx = np.asarray([i for i, clip in enumerate(ex_clips) if clip not in final_exclude], dtype=int)
    if len(final_train_idx) == 0 or len(set(y[final_train_idx])) < 2:
        raise SystemExit("--final_exclude_clip leaves no usable final training examples")
    final_x = x[final_train_idx]
    final_y = y[final_train_idx]
    extra_rows: list[dict[str, str]] = []
    extra_y = np.asarray([], dtype=np.int32)
    unique_extra_y = np.asarray([], dtype=np.int32)
    extra_repeat = max(1, int(args.extra_weight))
    if args.extra_examples:
        extra_rows, extra_y = load_extra_examples(Path(args.extra_examples))
        unique_extra_y = extra_y.copy()
        if extra_rows:
            extra_x = vectorize(extra_rows, numeric, sources)
            if extra_repeat > 1:
                extra_x = np.repeat(extra_x, extra_repeat, axis=0)
                extra_y = np.repeat(extra_y, extra_repeat, axis=0)
            final_x = np.vstack([final_x, extra_x])
            final_y = np.concatenate([final_y, extra_y])
    final_model.fit(final_x, final_y)
    model_path = out_dir / f"{best_name}_surface_xy_ranker.joblib"
    joblib.dump(
        {
            "model": final_model,
            "numeric_features": numeric,
            "source_features": sources,
            "max_rank": args.max_rank,
            "center_tol_px": args.center_tol_px,
            "loose_tol_px": args.loose_tol_px,
            "negative_min_dist_px": args.negative_min_dist_px,
            "best_model_loco": best_name,
            "best_direct_model_loco": best_direct,
            "best_fallback_loco": best_fallback,
            "nested_fallback_loco": nested_fallback_summary,
            "nested_gated_fallback_loco": nested_gated_fallback_summary,
            "fallback_gates": gates,
            "final_exclude_clip": sorted(final_exclude),
            "final_training_examples": int(len(final_y)),
            "base_final_training_examples": int(len(final_train_idx)),
            "extra_examples": args.extra_examples,
            "extra_weight": extra_repeat,
            "extra_training_examples": int(len(extra_y)),
            "unique_extra_examples": int(len(extra_rows)),
        },
        model_path,
    )

    write_csv(out_dir / "loco_predictions.csv", predictions)
    write_csv(out_dir / "loco_summary.csv", summary)
    write_csv(out_dir / "fallback_sweep.csv", fallback_summaries)
    if nested_rows:
        write_csv(out_dir / "nested_fallback_predictions.csv", nested_rows)
        write_csv(out_dir / "nested_fallback_by_clip.csv", summarize_by_clip(nested_rows, "nested_fallback"))
    if nested_selection_rows:
        write_csv(out_dir / "nested_fallback_selection.csv", nested_selection_rows)
    if nested_gated_rows:
        write_csv(out_dir / "nested_gated_fallback_predictions.csv", nested_gated_rows)
        write_csv(out_dir / "nested_gated_fallback_by_clip.csv", summarize_by_clip(nested_gated_rows, "nested_gated_fallback"))
    if nested_gated_selection_rows:
        write_csv(out_dir / "nested_gated_fallback_selection.csv", nested_gated_selection_rows)
    if best_fallback:
        best_fallback_name = str(best_fallback["model"])
        best_fallback_rows = fallback_predictions_by_name.get(best_fallback_name, [])
        write_csv(out_dir / "best_fallback_predictions.csv", best_fallback_rows)
        write_csv(out_dir / "best_fallback_by_clip.csv", summarize_by_clip(best_fallback_rows, best_fallback_name))
    write_csv(
        out_dir / "training_examples.csv",
        [
            {
                "clip": ex["clip"],
                "frame": ex["frame"],
                "y": ex["y"],
                "dist_px": round(ex["dist_px"], 3) if ex["dist_px"] != "" else "",
                "rank": ex["row"].get("rank", ""),
                "source": ex["row"].get("cand_source", ""),
                "confidence": ex["label"].get("confidence", ""),
                "visible": int(label_visible(ex["label"])),
            }
            for ex in examples
        ],
    )
    if extra_rows:
        write_csv(
            out_dir / "extra_training_examples.csv",
            [
                {
                    "clip": row.get("clip", ""),
                    "frame": row.get("frame", ""),
                    "y": int(unique_extra_y[i]),
                    "kind": row.get("hard_kind", ""),
                    "rank": row.get("rank", ""),
                    "source": row.get("cand_source", ""),
                    "candidate_dist_px": row.get("candidate_dist_px", ""),
                    "old_selected_dist_px": row.get("old_selected_dist_px", ""),
                    "weight": extra_repeat,
                }
                for i, row in enumerate(extra_rows)
            ],
        )
    metadata = {
        "labels": str(args.labels),
        "results_dir": str(args.results_dir),
        "clips": clips,
        "frames": len(labels),
        "missing_frames": missing_frames,
        "null_labels": len(null_labels),
        "null_missing_frames": null_missing_frames,
        "examples": len(examples),
        "positive_examples": int(np.sum(y == 1)),
        "negative_examples": int(np.sum(y == 0)),
        "null_negative_examples": null_examples,
        "ignored_near_examples": ignored_near,
        "numeric_features": numeric,
        "source_features": sources,
        "best_model_loco": best_name,
        "best_direct_model_loco": best_direct,
        "best_fallback_loco": best_fallback,
        "nested_fallback_loco": nested_fallback_summary,
        "nested_gated_fallback_loco": nested_gated_fallback_summary,
        "fallback_gates": gates,
        "final_exclude_clip": sorted(final_exclude),
        "final_training_examples": int(len(final_y)),
        "base_final_training_examples": int(len(final_train_idx)),
        "extra_examples": args.extra_examples,
        "extra_weight": extra_repeat,
        "extra_training_examples": int(len(extra_y)),
        "unique_extra_examples": int(len(extra_rows)),
        "model_path": str(model_path),
        "metric_caveat": "fallback threshold is selected from this LOCO sweep; use nested threshold selection or a separate holdout for unbiased reporting",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "README.md").write_text(
        "# Surface XY Tube Ranker\n\n"
        "Leave-one-clip-out evaluation for textured/non-sky labels.\n\n"
        f"Best direct LOCO model: `{best_direct}`\n\n"
        f"Best confidence fallback: `{best_fallback['model'] if best_fallback else ''}`\n\n"
        f"Nested confidence fallback: `{nested_fallback_summary['strict_recall'] if nested_fallback_summary else ''}` strict recall\n\n"
        f"Nested gated fallback: `{nested_gated_fallback_summary['strict_recall'] if nested_gated_fallback_summary else ''}` strict recall\n\n"
        "Caveat: the fallback threshold is selected from this LOCO sweep. Treat it\n"
        "as model-selection evidence unless rerun with nested threshold selection\n"
        "or a separate holdout.\n\n"
        "See `loco_summary.csv`, `fallback_sweep.csv`, "
        "`nested_fallback_selection.csv`, `nested_gated_fallback_selection.csv`, `best_fallback_by_clip.csv`, "
        "`loco_predictions.csv`, and `metadata.json`.\n"
    )
    print(out_dir / "loco_summary.csv")
    print(model_path)


if __name__ == "__main__":
    main()
