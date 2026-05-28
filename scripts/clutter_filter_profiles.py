#!/usr/bin/env python3
"""Build interpretable clutter-filter profiles from labeled tube alternatives."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


POSITIVE_LABELS = {"target"}
WEAK_CONF_RE = re.compile(r"vision_conf=([0-9.]+)")

PROFILE_FEATURES = [
    "cand_mean_residual",
    "cand_mean_appearance",
    "cand_line_context",
    "cand_attached_support",
    "cand_texture",
    "cand_map_score",
    "cand_native_dark_score",
    "cand_sky_like",
    "tube_hit_rate",
    "tube_map_hit_rate",
    "tube_appearance_only_rate",
    "tube_mean_line_context",
    "tube_max_line_context",
    "tube_mean_attached_support",
    "tube_max_attached_support",
    "tube_mean_native_dark_score",
    "tube_mean_sky_like",
    "tube_mean_texture",
    "tube_mean_residual",
    "tube_mean_appearance",
    "tube_mean_pair_score",
    "tube_positive_pair_rate",
    "tube_mean_align_gain",
    "tube_positive_align_rate",
    "tube_mean_speed",
    "tube_mean_accel",
    "tube_log_cand_density",
]

EXCLUDE_FEATURES = {
    "frame",
    "rank",
    "track_id",
    "x",
    "y",
    "eligible",
    "passes_floor",
    "selected",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--label_csv",
        action="append",
        required=True,
        help="Labeled review CSV. May be repeated.",
    )
    p.add_argument("--max_rank", type=int, default=8)
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


def top_tube_cache(results_dir: Path, clip: str, cache: dict[str, dict[tuple[int, int], dict[str, str]]]) -> dict[tuple[int, int], dict[str, str]]:
    if clip in cache:
        return cache[clip]
    path = results_dir / clip / "top_tubes.csv"
    rows: dict[tuple[int, int], dict[str, str]] = {}
    if path.exists():
        for row in read_csv(path):
            frame = int(float(row.get("frame", "0") or 0))
            rank = int(float(row.get("rank", "999") or 999))
            rows[(frame, rank)] = row
    cache[clip] = rows
    return rows


def source_name(path: Path) -> str:
    text = str(path)
    if "tube_alternative_review_packet_top16" in text:
        return "human_top16"
    if "round2" in text:
        return "vision_round2"
    if "hard_negative" in text:
        return "vision_round1"
    return path.parent.name


def label_weight(review: dict[str, str], source: str) -> float:
    if source.startswith("human"):
        return 1.0
    notes = review.get("human_notes", "")
    match = WEAK_CONF_RE.search(notes)
    conf = float(match.group(1)) if match else 0.75
    return max(0.25, min(0.85, conf))


def load_examples(label_csvs: list[Path], results_dir: Path, max_rank: int) -> list[dict[str, Any]]:
    cache: dict[str, dict[tuple[int, int], dict[str, str]]] = {}
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for path in label_csvs:
        source = source_name(path)
        for review in read_csv(path):
            label = review.get("human_label", "").strip().lower()
            if not label:
                continue
            clip = review.get("clip", "")
            frame = int(float(review.get("frame", "0") or 0))
            rank = int(float(review.get("rank", "999") or 999))
            if rank > max_rank:
                continue
            key = (clip, frame, rank, source)
            if key in seen:
                continue
            seen.add(key)
            top = top_tube_cache(results_dir, clip, cache).get((frame, rank))
            if not top:
                continue
            examples.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "rank": rank,
                    "human_label": label,
                    "source": source,
                    "weight": label_weight(review, source),
                    "y": 1 if label in POSITIVE_LABELS else 0,
                    "review": review,
                    "row": top,
                }
            )
    return examples


def infer_numeric_features(examples: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for ex in examples:
        for key, value in ex["row"].items():
            if key in EXCLUDE_FEATURES or key == "cand_source":
                continue
            if safe_float(value) is not None and key not in out:
                out.append(key)
    return out


def vectorize(examples: list[dict[str, Any]], numeric: list[str], sources: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for ex in examples:
        row = ex["row"]
        vals = [safe_float(row.get(name)) if safe_float(row.get(name)) is not None else np.nan for name in numeric]
        cand_source = row.get("cand_source", "")
        vals.extend(1.0 if src == f"src_{cand_source}" else 0.0 for src in sources)
        rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def percentile(vals: list[float], q: float) -> float:
    clean = [v for v in vals if math.isfinite(v)]
    return float(np.percentile(clean, q)) if clean else float("nan")


def profile_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_label[ex["human_label"]].append(ex)
    for label, exs in sorted(by_label.items()):
        row: dict[str, Any] = {
            "label": label,
            "n": len(exs),
            "targets": sum(ex["y"] for ex in exs),
            "sources": ";".join(f"{k}:{v}" for k, v in Counter(ex["source"] for ex in exs).most_common()),
        }
        for feat in PROFILE_FEATURES:
            vals = [safe_float(ex["row"].get(feat)) for ex in exs]
            nums = [v for v in vals if v is not None]
            row[f"{feat}_p25"] = round(percentile(nums, 25), 4)
            row[f"{feat}_median"] = round(percentile(nums, 50), 4)
            row[f"{feat}_p75"] = round(percentile(nums, 75), 4)
        rows.append(row)
    return rows


def target_delta_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = [ex for ex in examples if ex["y"] == 1]
    out: list[dict[str, Any]] = []
    for label in sorted({ex["human_label"] for ex in examples if ex["y"] == 0}):
        neg = [ex for ex in examples if ex["human_label"] == label]
        if len(neg) < 3 or len(target) < 3:
            continue
        for feat in PROFILE_FEATURES:
            t_vals = [safe_float(ex["row"].get(feat)) for ex in target]
            n_vals = [safe_float(ex["row"].get(feat)) for ex in neg]
            t = [v for v in t_vals if v is not None]
            n = [v for v in n_vals if v is not None]
            if len(t) < 3 or len(n) < 3:
                continue
            t_med = percentile(t, 50)
            n_med = percentile(n, 50)
            pooled_iqr = max(1e-6, 0.5 * ((percentile(t, 75) - percentile(t, 25)) + (percentile(n, 75) - percentile(n, 25))))
            out.append(
                {
                    "negative_label": label,
                    "feature": feat,
                    "target_median": round(t_med, 4),
                    "negative_median": round(n_med, 4),
                    "delta_neg_minus_target": round(n_med - t_med, 4),
                    "robust_effect": round((n_med - t_med) / pooled_iqr, 4),
                }
            )
    return sorted(out, key=lambda r: abs(float(r["robust_effect"])), reverse=True)


def row_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, weights: np.ndarray | None = None) -> dict[str, Any]:
    pred = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0, sample_weight=weights)
    out = {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "selected": int(pred.sum()),
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, scores, sample_weight=weights)), 3)
    except ValueError:
        out["roc_auc"] = ""
    try:
        out["avg_precision"] = round(float(average_precision_score(y_true, scores, sample_weight=weights)), 3)
    except ValueError:
        out["avg_precision"] = ""
    return out


def train_models(examples: list[dict[str, Any]], out_dir: Path, random_state: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    numeric = infer_numeric_features(examples)
    cand_sources = sorted({ex["row"].get("cand_source", "") for ex in examples if ex["row"].get("cand_source", "")})
    source_features = [f"src_{src}" for src in cand_sources]
    feature_names = numeric + source_features
    x = vectorize(examples, numeric, source_features)
    y = np.asarray([ex["y"] for ex in examples], dtype=np.int32)
    weights = np.asarray([ex["weight"] for ex in examples], dtype=np.float64)

    models: dict[str, Any] = {
        "weighted_logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.45, max_iter=2000, solver="liblinear", class_weight="balanced", random_state=random_state)),
            ]
        ),
        "weighted_extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=400, max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=random_state)),
            ]
        ),
    }

    summary: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        fit_kwargs = {"model__sample_weight": weights}
        model.fit(x, y, **fit_kwargs)
        scores = model.predict_proba(x)[:, 1]
        # Pick a conservative threshold: no more than 10% weighted false-positive rate if possible.
        thresholds = sorted(set(float(s) for s in scores), reverse=True)
        best_thr = 0.5
        best_key = (-1.0, -1.0)
        for thr in thresholds:
            pred = scores >= thr
            fp_w = weights[(pred == 1) & (y == 0)].sum()
            neg_w = weights[y == 0].sum()
            fpr = fp_w / max(1e-9, neg_w)
            tp_w = weights[(pred == 1) & (y == 1)].sum()
            pos_w = weights[y == 1].sum()
            rec = tp_w / max(1e-9, pos_w)
            precision_w = tp_w / max(1e-9, weights[pred == 1].sum())
            if fpr <= 0.10 and (rec, precision_w) > best_key:
                best_key = (rec, precision_w)
                best_thr = thr
        metric_row = row_metrics(y, scores, best_thr, weights)
        metric_row.update({"model": name, "examples": len(examples), "positives": int(y.sum()), "negatives": int(len(y) - y.sum())})
        summary.append(metric_row)
        joblib.dump(
            {
                "model": model,
                "threshold": best_thr,
                "numeric_features": numeric,
                "source_features": source_features,
                "feature_names": feature_names,
            },
            out_dir / f"{name}.joblib",
        )

        fitted = model.named_steps["model"]
        if hasattr(fitted, "feature_importances_"):
            importances = fitted.feature_importances_
        else:
            coefs = getattr(fitted, "coef_", np.zeros((1, len(feature_names))))[0]
            importances = np.abs(coefs)
        for feat, val in sorted(zip(feature_names, importances), key=lambda x: abs(float(x[1])), reverse=True)[:30]:
            feature_rows.append({"model": name, "feature": feat, "importance": round(float(val), 6)})

    return summary, feature_rows, numeric, source_features


def markdown_report(out_dir: Path, examples: list[dict[str, Any]], model_summary: list[dict[str, Any]], delta_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> str:
    label_counts = Counter(ex["human_label"] for ex in examples)
    source_counts = Counter(ex["source"] for ex in examples)
    lines = [
        "# Clutter Filter Profile Report",
        "",
        "This report mixes human labels and weak vision labels. Treat it as a filter-design aid, not final accuracy proof.",
        "",
        "## Dataset",
        "",
        f"- Examples matched to `top_tubes.csv`: {len(examples)}",
        f"- Target examples: {sum(ex['y'] for ex in examples)}",
        f"- Negative examples: {sum(1 - ex['y'] for ex in examples)}",
        f"- Sources: {', '.join(f'{k}={v}' for k, v in source_counts.most_common())}",
        "",
        "Labels:",
        "",
    ]
    for label, count in label_counts.most_common():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Conservative Binary Filters", ""])
    for row in model_summary:
        lines.append(
            f"- `{row['model']}` threshold `{row['threshold']}`: precision {row['precision']}, "
            f"recall {row['recall']}, FP {row['fp']}, FN {row['fn']} on the mixed label set."
        )
    lines.extend(["", "## Strongest Target-vs-Clutter Feature Differences", ""])
    for row in delta_rows[:18]:
        lines.append(
            f"- `{row['negative_label']}`: `{row['feature']}` target median {row['target_median']} vs "
            f"negative median {row['negative_median']} (effect {row['robust_effect']})."
        )
    lines.extend(["", "## Highest-Use Model Features", ""])
    for model in sorted({r["model"] for r in feature_rows}):
        lines.append(f"`{model}`:")
        for row in [r for r in feature_rows if r["model"] == model][:12]:
            lines.append(f"- `{row['feature']}` importance {row['importance']}")
        lines.append("")
    lines.extend(
        [
            "## Practical Filter Profiles",
            "",
            "- `line_attached`: prioritize attached-support and line-context rejection. This is the cleanest filter family because poles/branches score high on structure while true sky drones usually do not.",
            "- `terrain_texture`: use texture, low sky-likeness, low native-dark compactness, and weak pair/alignment evidence. Avoid a single texture threshold because dark drones against trees can collide with it.",
            "- `boundary_artifact`: use low pair evidence plus sky/cloud boundary context. Keep this as a soft penalty; skyline positives exist.",
            "- `near_target_wrong_center`: do not train it as pure negative forever. Use it to improve box-centering/ranking, not just suppression.",
            "",
            "Recommended next integration: add the weighted ExtraTrees score as a candidate/tube `clutter_filter_score`, then suppress only when the learned target probability is below threshold and the old detector score is not extreme.",
            "",
        ]
    )
    text = "\n".join(lines)
    (out_dir / "clutter_filter_profiles.md").write_text(text)
    return text


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples([Path(p) for p in args.label_csv], Path(args.results_dir), args.max_rank)
    if not examples:
        raise SystemExit("no matched examples")

    data_rows = []
    for ex in examples:
        row = {
            "clip": ex["clip"],
            "frame": ex["frame"],
            "rank": ex["rank"],
            "human_label": ex["human_label"],
            "y": ex["y"],
            "source": ex["source"],
            "weight": round(float(ex["weight"]), 4),
            "bbox": ex["review"].get("bbox", ""),
            "human_notes": ex["review"].get("human_notes", ""),
        }
        for feat in PROFILE_FEATURES:
            row[feat] = ex["row"].get(feat, "")
        data_rows.append(row)

    write_csv(out_dir / "matched_training_examples.csv", data_rows)
    profile = profile_rows(examples)
    deltas = target_delta_rows(examples)
    model_summary, feature_importance, _, _ = train_models(examples, out_dir, args.random_state)
    write_csv(out_dir / "profile_summary.csv", profile)
    write_csv(out_dir / "target_vs_clutter_deltas.csv", deltas)
    write_csv(out_dir / "model_summary.csv", model_summary)
    write_csv(out_dir / "feature_importance.csv", feature_importance)
    markdown_report(out_dir, examples, model_summary, deltas, feature_importance)
    print(out_dir / "clutter_filter_profiles.md")


if __name__ == "__main__":
    main()
