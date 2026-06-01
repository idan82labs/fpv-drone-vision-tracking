#!/usr/bin/env python3
"""Train a weakly supervised multi-class crop-stack verifier.

This is the CS-JS2 observation probe after the binary crop-stack score failed
honest LOCO replay. It keeps the same target/background crop features, but
learns separate classes for:

    T: true drone candidate
    S: static/background hot spot
    E: attached tree/branch/terrain/line clutter
    H: skyline/boundary/parallax clutter
    G: generic or uncertain clutter

The labels for negative classes are weak taxonomy labels inferred from existing
router/CLBA/context fields. This is still offline research code, not runtime.
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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import train_crop_stack_verifier as crop
except ModuleNotFoundError:  # pragma: no cover
    from scripts import train_crop_stack_verifier as crop


CLASS_NAMES = ["T", "S", "E", "H", "G"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument(
        "--taxonomy_labels",
        default="",
        help="Optional review-packet CSV with human taxonomy labels keyed by clip/frame/rank.",
    )
    p.add_argument(
        "--taxonomy_label_column",
        default="taxonomy_label",
        help="Column to read from --taxonomy_labels; falls back to human_label when empty.",
    )
    p.add_argument(
        "--append_taxonomy_examples",
        action="store_true",
        help="Append labeled review-packet rows that were not selected by the default hard-example sampler.",
    )
    p.add_argument("--results_dir", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--center_tol_px", type=float, default=8.0)
    p.add_argument("--negative_min_dist_px", type=float, default=24.0)
    p.add_argument("--negatives_per_frame", type=int, default=4)
    p.add_argument("--null_negatives_per_frame", type=int, default=5)
    p.add_argument("--confidence", nargs="*", default=["high", "medium_high"])
    p.add_argument("--null_confidence", nargs="*", default=["high_not_visible"])
    p.add_argument("--window_radius", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=17)
    p.add_argument("--patch_size", type=int, default=9)
    p.add_argument("--source_geometry_features", action="store_true")
    p.add_argument("--detector_scale", type=float, default=0.5)
    p.add_argument("--orb_features", type=int, default=900)
    p.add_argument("--min_matches", type=int, default=18)
    p.add_argument("--models", nargs="+", choices=("logistic", "hist_gbdt"), default=["hist_gbdt"])
    p.add_argument("--save_loco_models", action="store_true")
    p.add_argument("--random_state", type=int, default=23)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    crop.write_csv(path, rows)


def sigmoid_logit(prob: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, prob))
    return math.log(p / (1.0 - p))


def row_score(row: dict[str, Any], *names: str) -> float:
    return max(crop.safe_float(row.get(name)) for name in names)


def weak_clutter_class(row: dict[str, Any]) -> str:
    if crop.safe_int(row.get("hard_label"), 0) == 1:
        return "T"

    router = str(row.get("cand_router_state", "")).strip()
    boundary_rate = crop.safe_float(row.get("tube_router_boundary_rate"))
    line_rate = crop.safe_float(row.get("tube_router_line_attached_rate"))
    surface_rate = crop.safe_float(row.get("tube_router_surface_backed_rate"))
    line = row_score(row, "cand_line_context", "tube_mean_line_context")
    support = row_score(row, "cand_attached_support", "tube_mean_attached_support") / 12.0
    texture = row_score(row, "cand_texture", "tube_mean_texture") / 80.0
    sky_texture = row_score(row, "cand_sky_like", "tube_mean_sky_like") * max(0.0, texture)
    bg_static = crop.safe_float(row.get("clba_bg_static_likelihood"))
    bg_q = crop.safe_float(row.get("clba_bg_q"))
    path_dist = crop.safe_float(row.get("clba_path_bg_dist_mean"), crop.safe_float(row.get("tube_mean_bg_dist")))
    static_near = max(0.0, 1.0 - min(path_dist, 8.0) / 8.0)

    boundary_signal = 1.25 * boundary_rate + 0.55 * sky_texture + 0.35 * row_score(row, "clba_bg_anisotropy")
    attached_signal = (
        0.95 * line_rate
        + 0.55 * line
        + 0.50 * support
        + 0.35 * surface_rate
        + 0.25 * texture
        + 0.35 * crop.safe_float(row.get("clba_attached_likelihood"))
    )
    static_signal = 0.80 * bg_static + 0.45 * bg_q + 0.50 * static_near

    if router in {"boundary_mixed", "sky_target_near_surface"} or boundary_signal >= max(attached_signal, static_signal, 0.9):
        return "H"
    if router == "line_attached" or attached_signal >= max(static_signal, 0.85):
        return "E"
    if static_signal >= 0.9:
        return "S"
    return "G"


def normalize_taxonomy_label(raw: Any) -> str | None:
    """Map review-packet taxonomy labels into the T/S/E/H/G training classes.

    ``None`` means "ignore this candidate as supervised data". Near-target and
    uncertain rows should not become hard negatives.
    """

    label = str(raw or "").strip().lower()
    label = label.replace("-", "_").replace(" ", "_")
    if not label:
        return ""
    mapping: dict[str, str | None] = {
        "target": "T",
        "drone_target": "T",
        "true_target": "T",
        "static_hotspot": "S",
        "static_background": "S",
        "background_hotspot": "S",
        "line_attached": "E",
        "attached_tree_branch_terrain": "E",
        "attached_branch_tree_pole": "E",
        "attached_branch": "E",
        "terrain_texture": "E",
        "tree_branch": "E",
        "grass_terrain": "E",
        "parallax_edge": "H",
        "boundary_artifact": "H",
        "skyline_boundary_parallax": "H",
        "skyline_boundary": "H",
        "horizon_boundary": "H",
        "cloud_boundary": "H",
        "appearance_blob": "G",
        "generic_clutter": "G",
        "generic": "G",
        "noise": "G",
        "not_target": "G",
        "near_target_wrong_center": None,
        "close_but_wrong": None,
        "uncertain": None,
        "unsure": None,
        "unknown": None,
        "unk": None,
        "ignore": None,
        "skip": None,
    }
    if label not in mapping:
        raise ValueError(f"unknown taxonomy label: {raw!r}")
    return mapping[label]


def taxonomy_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("clip", "")).strip(),
        crop.safe_int(row.get("frame"), -1),
        crop.safe_int(row.get("rank"), -1),
    )


def load_taxonomy_labels(path: Path, label_column: str) -> tuple[dict[tuple[str, int, int], str | None], Counter]:
    labels: dict[tuple[str, int, int], str | None] = {}
    counts: Counter = Counter()
    if not path:
        return labels, counts
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get(label_column, "")
            if not str(raw or "").strip():
                raw = row.get("human_label", "")
            if not str(raw or "").strip():
                continue
            norm = normalize_taxonomy_label(raw)
            key = taxonomy_key(row)
            if key[0] and key[1] >= 0 and key[2] >= 0:
                labels[key] = norm
                counts["ignored" if norm is None else norm] += 1
    return labels, counts


def load_top_tube_lookup(results_dir: Path, clip: str, max_rank: int) -> dict[tuple[int, int], dict[str, str]]:
    path = crop.top_tubes_path(results_dir, clip)
    out: dict[tuple[int, int], dict[str, str]] = {}
    if path is None or not path.exists():
        return out
    for row in crop.read_csv(path):
        frame = crop.safe_int(row.get("frame"), -1)
        rank = crop.safe_int(row.get("rank"), 999999)
        if frame >= 0 and rank <= max_rank:
            out[(frame, rank)] = row
    return out


def append_taxonomy_examples(
    examples: list[dict[str, str]],
    taxonomy_path: Path,
    taxonomy_labels: dict[tuple[str, int, int], str | None],
    results_dir: Path,
    max_rank: int,
) -> int:
    existing = {taxonomy_key(row) for row in examples}
    lookups: dict[str, dict[tuple[int, int], dict[str, str]]] = {}
    added = 0
    with taxonomy_path.open(newline="") as f:
        for packet_row in csv.DictReader(f):
            key = taxonomy_key(packet_row)
            cls = taxonomy_labels.get(key, "")
            if not cls or cls is None or key in existing:
                continue
            clip, frame, rank = key
            if rank > max_rank:
                continue
            if clip not in lookups:
                lookups[clip] = load_top_tube_lookup(results_dir, clip, max_rank)
            source = lookups[clip].get((frame, rank))
            if source is None:
                continue
            out = dict(source)
            out.update(
                {
                    "clip": clip,
                    "hard_label": "1" if cls == "T" else "0",
                    "hard_kind": "taxonomy_packet_appended",
                    "taxonomy_packet_label": cls,
                    "taxonomy_packet_rank": str(rank),
                }
            )
            examples.append(out)
            existing.add(key)
            added += 1
    return added


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.18, class_weight="balanced", max_iter=2500, solver="lbfgs", random_state=seed)),
            ]
        ),
        "hist_gbdt": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=160,
                        learning_rate=0.035,
                        max_leaf_nodes=9,
                        min_samples_leaf=12,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def bundle(model: Pipeline, args: argparse.Namespace, model_name: str, heldout_clip: str | None = None) -> dict[str, Any]:
    out = {
        "model": model,
        "best_model_loco": model_name,
        "class_names": CLASS_NAMES,
        "window_radius": args.window_radius,
        "crop_size": args.crop_size,
        "patch_size": args.patch_size,
        "detector_scale": args.detector_scale,
        "scalar_columns": crop.SCALAR_COLUMNS,
        "score_mode": "multiclass_proba",
        "source_geometry_features": bool(args.source_geometry_features),
        "source_categories": crop.SOURCE_CATEGORIES,
        "weak_label_taxonomy": "heuristic_router_clba_context_v2_with_generic",
    }
    if heldout_clip is not None:
        out["heldout_clip"] = heldout_clip
        out["training_mode"] = "leave_one_clip_out"
    return out


def class_probs(model: Pipeline, x: np.ndarray) -> dict[str, np.ndarray]:
    probs = model.predict_proba(x)
    model_classes = [str(c) for c in model.classes_]
    return {
        cls: probs[:, model_classes.index(cls)] if cls in model_classes else np.zeros((x.shape[0],), dtype=np.float32)
        for cls in CLASS_NAMES
    }


def pairwise_t_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[(str(row.get("clip", "")), crop.safe_int(row.get("frame"), -1))].append(row)
    wins = ties = total = pos_frames = 0
    for group in by_frame.values():
        pos = [r for r in group if r.get("weak_class") == "T"]
        neg = [r for r in group if r.get("weak_class") != "T"]
        if pos and neg:
            pos_frames += 1
        for p in pos:
            ps = crop.safe_float(p.get("crop_t_logit"))
            for n in neg:
                ns = crop.safe_float(n.get("crop_t_logit"))
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
        "positive_frames_with_negatives": pos_frames,
    }


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    counts = Counter(str(r.get("weak_class", "")) for r in rows)
    correct = sum(str(r.get("weak_class", "")) == str(r.get("crop_pred_class", "")) for r in rows)
    return {
        "model": model_name,
        "rows": len(rows),
        "accuracy": round(correct / max(1, len(rows)), 6),
        **{f"class_{cls}": counts.get(cls, 0) for cls in CLASS_NAMES},
        **pairwise_t_summary(rows),
    }


def summarize_by_clip(rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[str(row.get("clip", ""))].append(row)
    out = []
    for clip, group in sorted(by_clip.items()):
        rec = summarize(group, model_name)
        rec["clip"] = clip
        out.append(rec)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = crop.build_examples_from_labels(
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
    if not examples:
        raise SystemExit("no hard examples")
    taxonomy_labels: dict[tuple[str, int, int], str | None] = {}
    taxonomy_counts: Counter = Counter()
    taxonomy_appended_examples = 0
    if args.taxonomy_labels:
        taxonomy_labels, taxonomy_counts = load_taxonomy_labels(Path(args.taxonomy_labels), args.taxonomy_label_column)
        if args.append_taxonomy_examples:
            taxonomy_appended_examples = append_taxonomy_examples(
                examples,
                Path(args.taxonomy_labels),
                taxonomy_labels,
                Path(args.results_dir),
                args.max_rank,
            )
    clips = sorted({str(r.get("clip", "")) for r in examples if r.get("clip")})
    tube_paths = [p for clip in clips if (p := crop.top_tubes_path(Path(args.results_dir), clip)) is not None]
    tube_rows = crop.align.load_tube_rows(tube_paths)
    frame_cache = crop.align.FrameCache(Path(args.video_dir), args.detector_scale)
    transform_cache = crop.align.TransformCache(frame_cache, args.orb_features, args.min_matches)
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    try:
        for row in examples:
            vector, meta = crop.extract_stack_features(
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
            key = taxonomy_key(out)
            taxonomy_class = taxonomy_labels.get(key, "")
            if taxonomy_class is None:
                continue
            if taxonomy_class:
                out["weak_class"] = taxonomy_class
                out["weak_class_source"] = "taxonomy"
            else:
                out["weak_class"] = weak_clutter_class(out)
                out["weak_class_source"] = "weak_heuristic"
            vectors.append(vector)
            rows.append(out)
    finally:
        frame_cache.close()

    x = np.vstack(vectors).astype(np.float32)
    y = np.asarray([r["weak_class"] for r in rows])
    models = make_models(args.random_state)
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        model_rows: list[dict[str, Any]] = []
        for held_clip in clips:
            train_idx = np.asarray([i for i, row in enumerate(rows) if row.get("clip") != held_clip], dtype=int)
            test_idx = np.asarray([i for i, row in enumerate(rows) if row.get("clip") == held_clip], dtype=int)
            if train_idx.size == 0 or test_idx.size == 0 or len(set(y[train_idx].tolist())) < 2:
                continue
            model = clone(models[model_name])
            model.fit(x[train_idx], y[train_idx])
            if args.save_loco_models:
                loco_dir = out_dir / "loco_models" / model_name
                loco_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(bundle(model, args, model_name, held_clip), loco_dir / f"{held_clip}.joblib")
            probs = class_probs(model, x[test_idx])
            for idx_num, idx in enumerate(test_idx):
                out = dict(rows[int(idx)])
                for cls in CLASS_NAMES:
                    p_cls = float(probs[cls][idx_num])
                    out[f"crop_{cls.lower()}_prob"] = round(p_cls, 6)
                    out[f"crop_{cls.lower()}_logit"] = round(sigmoid_logit(p_cls), 6)
                out["crop_stack_score"] = out["crop_t_prob"]
                out["crop_pred_class"] = max(CLASS_NAMES, key=lambda cls: float(probs[cls][idx_num]))
                out["model"] = model_name
                model_rows.append(out)
                prediction_rows.append(out)
        summary_rows.append(summarize(model_rows, model_name))
        write_csv(out_dir / f"{model_name}_by_clip.csv", summarize_by_clip(model_rows, model_name))

    best_model_name = max(summary_rows, key=lambda r: (r["pairwise_win_rate"], r["accuracy"]))["model"]
    final_model = clone(models[best_model_name])
    final_model.fit(x, y)
    model_path = out_dir / f"{best_model_name}_multiclass_crop_stack_verifier.joblib"
    joblib.dump(bundle(final_model, args, best_model_name), model_path)

    write_csv(out_dir / "hard_examples_multiclass_features.csv", rows)
    write_csv(out_dir / "loco_predictions.csv", prediction_rows)
    write_csv(out_dir / "loco_summary.csv", summary_rows)
    metadata = {
        "examples": len(rows),
        "class_counts": dict(Counter(y.tolist())),
        "clips": clips,
        "feature_dim": int(x.shape[1]),
        "model_path": str(model_path),
        "best_model_loco": best_model_name,
        "class_names": CLASS_NAMES,
        "weak_label_taxonomy": "heuristic_router_clba_context_v2_with_generic",
        "taxonomy_labels": args.taxonomy_labels,
        "taxonomy_label_column": args.taxonomy_label_column,
        "taxonomy_label_counts": dict(taxonomy_counts),
        "taxonomy_labels_used": sum(1 for row in rows if row.get("weak_class_source") == "taxonomy"),
        "taxonomy_appended_examples": taxonomy_appended_examples,
        "metric_caveat": "weak negative taxonomy; LOCO replay is the gating metric",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# Multi-Class Crop-Stack Verifier\n\n"
        "Weakly supervised T/S/E/H/G crop-stack observation probe for JS1.\n\n"
        f"Examples: `{len(rows)}`\n\n"
        f"Best LOCO model: `{best_model_name}`\n\n"
        "See `loco_summary.csv`, `*_by_clip.csv`, and `metadata.json`.\n"
    )
    print(out_dir / "loco_summary.csv")
    print(model_path)


if __name__ == "__main__":
    main()
