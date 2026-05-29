#!/usr/bin/env python3
"""Train/evaluate a candidate acquisition/null ranker on top-tube exports.

This differs from the surface XY ranker: it evaluates the whole decision
"select a candidate or emit no target" so invisible/no-target frames count.
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
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--negative_min_dist_px", type=float, default=24.0)
    p.add_argument("--thresholds", default="0:1:0.02")
    p.add_argument(
        "--decision_mode",
        choices=("select_best", "gate_selected"),
        default="select_best",
        help=(
            "select_best reselects the highest learned-score candidate; "
            "gate_selected only accepts/rejects the detector's selected row."
        ),
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "hist_gbdt", "extra_trees"),
        default=["logistic", "hist_gbdt", "extra_trees"],
    )
    p.add_argument("--random_state", type=int, default=29)
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


def visible(row: dict[str, str]) -> bool:
    raw = str(row.get("visible", "")).strip().lower()
    if raw in {"0", "false", "no", "empty", "none", "not_visible", "not visible"}:
        return False
    return True


def confidence_allowed(row: dict[str, str]) -> bool:
    raw = str(row.get("confidence", "")).strip().lower()
    return raw not in {"low", "low_review_required", "uncertain_bad"}


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


def bbox(row: dict[str, str], label: bool = False) -> tuple[float, float, float, float]:
    if label:
        return (
            float_or_default(row.get("det_x", row.get("x")), 0.0),
            float_or_default(row.get("det_y", row.get("y")), 0.0),
            float_or_default(row.get("det_w", row.get("w")), 1.0),
            float_or_default(row.get("det_h", row.get("h")), 1.0),
        )
    return (
        float_or_default(row.get("x"), 0.0),
        float_or_default(row.get("y"), 0.0),
        float_or_default(row.get("w"), 1.0),
        float_or_default(row.get("h"), 1.0),
    )


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return float(math.hypot(ax - bx, ay - by))


def parse_thresholds(raw: str) -> list[float]:
    if ":" in raw:
        start_s, end_s, step_s = raw.split(":", 2)
        start = float(start_s)
        end = float(end_s)
        step = float(step_s)
        vals: list[float] = []
        cur = start
        while cur <= end + 1e-9:
            vals.append(round(cur, 6))
            cur += step
        return vals
    return [float(v) for v in raw.split(",") if v.strip()]


def load_labels(paths: list[str]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, int], dict[str, str]] = {}
    priority: dict[tuple[str, int], int] = {}
    for idx, raw in enumerate(paths):
        for row in read_csv(Path(raw)):
            clip = row.get("clip", "")
            frame = safe_float(row.get("frame"))
            if not clip or frame is None:
                continue
            if not confidence_allowed(row):
                continue
            key = (clip, int(frame))
            # Later files override earlier files. This lets full-video labels
            # replace partial rows for the same clip/frame.
            if priority.get(key, -1) <= idx:
                by_key[key] = row
                priority[key] = idx
    return [by_key[k] for k in sorted(by_key, key=lambda x: (x[0], x[1]))]


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
                ("model", LogisticRegression(C=0.4, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=seed)),
            ]
        ),
        "hist_gbdt": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.035, max_leaf_nodes=8, min_samples_leaf=18, l2_regularization=1.2, random_state=seed)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=240, max_depth=6, min_samples_leaf=6, class_weight="balanced", random_state=seed)),
            ]
        ),
    }


def predict_score(model: Pipeline, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def build_examples(
    labels: list[dict[str, str]],
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    strict_tol: float,
    negative_min_dist: float,
    decision_mode: str = "select_best",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    examples: list[dict[str, Any]] = []
    feature_rows: list[dict[str, str]] = []
    for lab in labels:
        clip = lab["clip"]
        frame = int_or_default(lab.get("frame"), -1)
        rows = top_by_clip.get(clip, {}).get(frame, [])
        if not rows:
            continue
        if decision_mode == "gate_selected":
            selected = selected_baseline_row(rows)
            if selected is None:
                continue
            feature_rows.append(selected)
            is_visible = visible(lab)
            if not is_visible:
                examples.append({"clip": clip, "frame": frame, "row": selected, "label": lab, "y": 0, "dist_px": ""})
                continue
            dist = center_dist(bbox(selected), bbox(lab, label=True))
            y = 1 if dist <= strict_tol else 0
            examples.append({"clip": clip, "frame": frame, "row": selected, "label": lab, "y": y, "dist_px": dist})
            continue
        is_visible = visible(lab)
        gt = bbox(lab, label=True) if is_visible else None
        for row in rows:
            feature_rows.append(row)
            if not is_visible:
                examples.append({"clip": clip, "frame": frame, "row": row, "label": lab, "y": 0, "dist_px": ""})
                continue
            dist = center_dist(bbox(row), gt)  # type: ignore[arg-type]
            if dist <= strict_tol:
                y = 1
            elif dist >= negative_min_dist:
                y = 0
            else:
                continue
            examples.append({"clip": clip, "frame": frame, "row": row, "label": lab, "y": y, "dist_px": dist})
    return examples, feature_rows


def selected_baseline_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    selected = [r for r in rows if str(r.get("selected", "")).strip() in {"1", "True", "true"}]
    if selected:
        return selected[0]
    # If the runtime selected row is absent from top-N, this is no selection.
    return None


def best_verified_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: float_or_default(r.get("verified_score"), -1e9))


def eval_choice(
    lab: dict[str, str],
    row: dict[str, str] | None,
    strict_tol: float,
    loose_tol: float,
) -> dict[str, Any]:
    is_visible = visible(lab)
    selected = row is not None
    dist: float | None = None
    strict = False
    loose = False
    if is_visible and row is not None:
        dist = center_dist(bbox(row), bbox(lab, label=True))
        strict = dist <= strict_tol
        loose = dist <= loose_tol
    correct = (strict if is_visible else not selected)
    return {
        "visible": is_visible,
        "selected": selected,
        "strict_hit": strict,
        "loose_hit": loose,
        "no_target_correct": (not selected) if not is_visible else "",
        "all_correct": correct,
        "dist_px": "" if dist is None else round(dist, 3),
    }


def summarize(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    visible_rows = [r for r in rows if r["visible"]]
    invisible_rows = [r for r in rows if not r["visible"]]
    return {
        "model": model,
        "frames": len(rows),
        "all_correct": sum(bool(r["all_correct"]) for r in rows),
        "all_accuracy": round(sum(bool(r["all_correct"]) for r in rows) / max(1, len(rows)), 4),
        "visible_frames": len(visible_rows),
        "visible_strict_hits": sum(bool(r["strict_hit"]) for r in visible_rows),
        "visible_strict_recall": round(sum(bool(r["strict_hit"]) for r in visible_rows) / max(1, len(visible_rows)), 4),
        "visible_loose_hits": sum(bool(r["loose_hit"]) for r in visible_rows),
        "visible_loose_recall": round(sum(bool(r["loose_hit"]) for r in visible_rows) / max(1, len(visible_rows)), 4),
        "invisible_frames": len(invisible_rows),
        "invisible_no_select": sum(bool(r["no_target_correct"]) for r in invisible_rows),
        "invisible_no_select_rate": round(sum(bool(r["no_target_correct"]) for r in invisible_rows) / max(1, len(invisible_rows)), 4),
        "selected_frames": sum(bool(r["selected"]) for r in rows),
        "selected_rate": round(sum(bool(r["selected"]) for r in rows) / max(1, len(rows)), 4),
    }


def score_frames(
    labels: list[dict[str, str]],
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    model_name: str,
    model: Pipeline | None,
    numeric: list[str],
    sources: list[str],
    threshold: float,
    strict_tol: float,
    loose_tol: float,
    decision_mode: str = "select_best",
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for lab in labels:
        clip = lab["clip"]
        frame = int_or_default(lab.get("frame"), -1)
        rows = top_by_clip.get(clip, {}).get(frame, [])
        chosen: dict[str, str] | None = None
        score = ""
        if model_name == "baseline_selected":
            chosen = selected_baseline_row(rows)
        elif model_name == "baseline_verified":
            chosen = best_verified_row(rows)
        elif rows and model is not None:
            if decision_mode == "gate_selected":
                selected = selected_baseline_row(rows)
                if selected is not None:
                    selected_score = predict_score(model, vectorize([selected], numeric, sources))[0]
                    score = round(float(selected_score), 6)
                    if float(selected_score) >= threshold:
                        chosen = selected
            else:
                scores = predict_score(model, vectorize(rows, numeric, sources))
                best_i = int(np.argmax(scores))
                score = round(float(scores[best_i]), 6)
                if float(scores[best_i]) >= threshold:
                    chosen = rows[best_i]
        rec = eval_choice(lab, chosen, strict_tol, loose_tol)
        rec.update(
            {
                "model": model_name,
                "clip": clip,
                "frame": frame,
                "threshold": threshold if model is not None else "",
                "score": score,
                "rank": "" if chosen is None else chosen.get("rank", ""),
                "source": "" if chosen is None else chosen.get("cand_source", ""),
                "x": "" if chosen is None else chosen.get("x", ""),
                "y": "" if chosen is None else chosen.get("y", ""),
                "w": "" if chosen is None else chosen.get("w", ""),
                "h": "" if chosen is None else chosen.get("h", ""),
                "verified_score": "" if chosen is None else chosen.get("verified_score", ""),
                "tube_verifier_score": "" if chosen is None else chosen.get("tube_verifier_score", ""),
                "confidence": lab.get("confidence", ""),
            }
        )
        rows_out.append(rec)
    return rows_out


def apply_score_threshold(
    scored_rows: list[dict[str, Any]],
    model_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Apply a select/no-select threshold to rows scored once by a model.

    `score_frames(..., threshold=-inf)` already finds the best candidate and
    records its score. Threshold sweeps should not recompute those scores for
    every threshold; they only need to flip low-score frames to no-selection.
    """
    out: list[dict[str, Any]] = []
    for row in scored_rows:
        rec = dict(row)
        score = safe_float(rec.get("score"), -math.inf)
        keep = score is not None and score >= threshold and bool(rec.get("selected"))
        rec["model"] = model_name
        rec["threshold"] = round(threshold, 6)
        if not keep:
            is_visible = bool(rec.get("visible"))
            rec.update(
                {
                    "selected": False,
                    "strict_hit": False,
                    "loose_hit": False,
                    "no_target_correct": (not is_visible) if not is_visible else "",
                    "all_correct": not is_visible,
                    "dist_px": "",
                    "rank": "",
                    "source": "",
                    "x": "",
                    "y": "",
                    "w": "",
                    "h": "",
                }
            )
        out.append(rec)
    return out


def by_clip_summary(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["clip"]].append(row)
    out: list[dict[str, Any]] = []
    for clip, group in sorted(grouped.items()):
        rec = summarize(group, model)
        rec["clip"] = clip
        out.append(rec)
    return out


def choose_threshold(
    train_rows_by_thr: dict[float, list[dict[str, Any]]],
) -> tuple[float, dict[str, Any]]:
    best_thr = 0.0
    best_sum: dict[str, Any] | None = None
    for thr, rows in train_rows_by_thr.items():
        rec = summarize(rows, f"train_thr{thr:.2f}")
        # Full-video correctness first; visible loose recall second so a
        # degenerate "select nothing" threshold cannot win when visible frames
        # are common.
        key = (rec["all_accuracy"], rec["visible_loose_recall"], rec["visible_strict_recall"], -rec["selected_rate"])
        if best_sum is None or key > (
            best_sum["all_accuracy"],
            best_sum["visible_loose_recall"],
            best_sum["visible_strict_recall"],
            -best_sum["selected_rate"],
        ):
            best_thr = thr
            best_sum = rec
    assert best_sum is not None
    return best_thr, best_sum


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    clips = sorted({r["clip"] for r in labels})
    top_by_clip = {clip: load_top_tubes(Path(args.results_dir), clip, args.max_rank) for clip in clips}
    examples, feature_rows = build_examples(labels, top_by_clip, args.strict_tol_px, args.negative_min_dist_px, args.decision_mode)
    if not examples:
        raise SystemExit("no training examples")
    numeric, sources = infer_features(feature_rows)
    x = vectorize([e["row"] for e in examples], numeric, sources)
    y = np.asarray([int(e["y"]) for e in examples], dtype=np.int32)
    ex_clips = np.asarray([e["clip"] for e in examples], dtype=object)
    thresholds = parse_thresholds(args.thresholds)
    models = make_models(args.random_state)

    all_predictions: list[dict[str, Any]] = []
    baseline_selected = score_frames(labels, top_by_clip, "baseline_selected", None, numeric, sources, 0.0, args.strict_tol_px, args.loose_tol_px)
    baseline_verified = score_frames(labels, top_by_clip, "baseline_verified", None, numeric, sources, 0.0, args.strict_tol_px, args.loose_tol_px)
    all_predictions.extend(baseline_selected)
    all_predictions.extend(baseline_verified)

    summary_rows = [summarize(baseline_selected, "baseline_selected"), summarize(baseline_verified, "baseline_verified")]
    nested_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    nested_selection: list[dict[str, Any]] = []
    oof_scored_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in args.models}

    selected_models = {name: models[name] for name in args.models}

    for model_name, template in selected_models.items():
        scored_by_thr: dict[float, list[dict[str, Any]]] = {thr: [] for thr in thresholds}
        for held_clip in clips:
            train_idx = np.asarray([i for i, clip in enumerate(ex_clips) if clip != held_clip], dtype=int)
            if len(train_idx) == 0 or len(set(y[train_idx])) < 2:
                continue
            model = Pipeline(template.steps)
            model.fit(x[train_idx], y[train_idx])
            held_labels = [r for r in labels if r["clip"] == held_clip]
            held_scored = score_frames(
                held_labels,
                top_by_clip,
                model_name,
                model,
                numeric,
                sources,
                -math.inf,
                args.strict_tol_px,
                args.loose_tol_px,
                args.decision_mode,
            )
            oof_scored_by_model[model_name].extend(
                {**row, "model": f"oof_{model_name}", "threshold": ""} for row in held_scored
            )
            for thr in thresholds:
                scored_by_thr[thr].extend(apply_score_threshold(held_scored, model_name, thr))
        for thr, rows in scored_by_thr.items():
            rec = summarize(rows, f"{model_name}_thr{thr:.2f}")
            rec["base_model"] = model_name
            rec["threshold"] = round(thr, 6)
            threshold_rows.append(rec)
        # Nested threshold per held-out clip.
        for held_clip in clips:
            train_idx = np.asarray([i for i, clip in enumerate(ex_clips) if clip != held_clip], dtype=int)
            if len(train_idx) == 0 or len(set(y[train_idx])) < 2:
                continue
            model = Pipeline(template.steps)
            model.fit(x[train_idx], y[train_idx])
            train_labels = [r for r in labels if r["clip"] != held_clip]
            held_labels = [r for r in labels if r["clip"] == held_clip]
            train_scored = score_frames(
                train_labels,
                top_by_clip,
                model_name,
                model,
                numeric,
                sources,
                -math.inf,
                args.strict_tol_px,
                args.loose_tol_px,
                args.decision_mode,
            )
            train_rows_by_thr = {thr: apply_score_threshold(train_scored, model_name, thr) for thr in thresholds}
            thr, train_sum = choose_threshold(train_rows_by_thr)
            held_scored = score_frames(
                held_labels,
                top_by_clip,
                model_name,
                model,
                numeric,
                sources,
                -math.inf,
                args.strict_tol_px,
                args.loose_tol_px,
                args.decision_mode,
            )
            held_rows = apply_score_threshold(held_scored, f"nested_{model_name}", thr)
            nested_rows.extend(held_rows)
            test_sum = summarize(held_rows, f"nested_{model_name}_{held_clip}")
            nested_selection.append(
                {
                    "heldout_clip": held_clip,
                    "model": model_name,
                    "selected_threshold": round(thr, 6),
                    "train_all_accuracy": train_sum["all_accuracy"],
                    "train_visible_loose_recall": train_sum["visible_loose_recall"],
                    "test_all_accuracy": test_sum["all_accuracy"],
                    "test_visible_strict_recall": test_sum["visible_strict_recall"],
                    "test_visible_loose_recall": test_sum["visible_loose_recall"],
                    "test_invisible_no_select_rate": test_sum["invisible_no_select_rate"],
                }
            )

    for model_name in selected_models:
        rows = [r for r in nested_rows if r["model"] == f"nested_{model_name}"]
        if rows:
            summary_rows.append(summarize(rows, f"nested_{model_name}"))

    best = max(summary_rows[2:], key=lambda r: (r["all_accuracy"], r["visible_loose_recall"], r["visible_strict_recall"])) if len(summary_rows) > 2 else None
    if best is not None:
        best_model_name = best["model"].replace("nested_", "")
        final_model = make_models(args.random_state)[best_model_name]
        final_model.fit(x, y)
        model_path = out_dir / f"{best_model_name}_acquisition_null_ranker.joblib"
        joblib.dump(
            {
                "model": final_model,
                "numeric_features": numeric,
                "source_features": sources,
                "strict_tol_px": args.strict_tol_px,
                "loose_tol_px": args.loose_tol_px,
                "negative_min_dist_px": args.negative_min_dist_px,
                "best_nested_summary": best,
            },
            model_path,
        )
    else:
        model_path = out_dir / "no_model.joblib"

    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "threshold_sweep.csv", threshold_rows)
    write_csv(out_dir / "nested_predictions.csv", nested_rows)
    write_csv(out_dir / "nested_selection.csv", nested_selection)
    for model_name, rows in oof_scored_by_model.items():
        write_csv(out_dir / f"oof_{args.decision_mode}_scores_{model_name}.csv", rows)
    write_csv(out_dir / "baseline_selected_predictions.csv", baseline_selected)
    write_csv(out_dir / "baseline_verified_predictions.csv", baseline_verified)
    write_csv(out_dir / "nested_by_clip.csv", [row for model in selected_models for row in by_clip_summary([r for r in nested_rows if r["model"] == f"nested_{model}"], f"nested_{model}")])
    write_csv(
        out_dir / "training_examples.csv",
        [
            {
                "clip": e["clip"],
                "frame": e["frame"],
                "y": e["y"],
                "dist_px": "" if e["dist_px"] == "" else round(float(e["dist_px"]), 3),
                "rank": e["row"].get("rank", ""),
                "source": e["row"].get("cand_source", ""),
                "visible": int(visible(e["label"])),
                "confidence": e["label"].get("confidence", ""),
            }
            for e in examples
        ],
    )
    metadata = {
        "labels": args.labels,
        "results_dir": args.results_dir,
        "clips": clips,
        "label_frames": len(labels),
        "examples": len(examples),
        "positive_examples": int(np.sum(y == 1)),
        "negative_examples": int(np.sum(y == 0)),
        "numeric_features": numeric,
        "source_features": sources,
        "best_model_path": str(model_path),
        "decision_mode": args.decision_mode,
        "best_nested_summary": best,
        "caveat": "Only d129 currently contributes full invisible/no-target labels; null generalization needs more no-target clips.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "README.md").write_text(
        "# Acquisition / Null Ranker\n\n"
        "Candidate-level select-vs-no-select evaluation over current top-tube exports.\n\n"
        f"Decision mode: `{args.decision_mode}`\n\n"
        f"Best nested summary: `{best}`\n\n"
        "Caveat: no-target labels are still dominated by d129, so null suppression\n"
        "is not validated as a general default yet.\n"
    )
    print(out_dir / "summary.csv")
    print(model_path)


if __name__ == "__main__":
    main()
