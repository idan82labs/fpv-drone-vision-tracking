#!/usr/bin/env python3
"""Train/evaluate an offline mode supervisor for Viterbi vs HMM routing.

The supervisor is trained on selector-disagreement rows:

* Viterbi visible hit, HMM miss => choose permissive continuous tracking.
* Viterbi false box, HMM suppressed => choose conservative null/HMM behavior.

It then applies the predicted mode to every labeled frame in a held-out clip and
evaluates the resulting selected boxes. This is deliberately an offline harness;
it answers whether current engineered features contain enough signal to route
between selector families before that logic is moved into the runtime detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import apply_surface_sequence_selector as selector
    import evaluate_tracking_run as tracking_eval
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import apply_surface_sequence_selector as selector
    from scripts import evaluate_tracking_run as tracking_eval


POSITIVE_HMM_CATEGORY = "a_false_b_suppressed"
NEGATIVE_VITERBI_CATEGORY = "a_visible_hit_b_miss"

BASE_FEATURES = (
    "score",
    "verified_score",
    "tube_verifier_score",
    "competitor_margin",
    "cand_area",
    "cand_fill",
    "cand_aspect",
    "cand_mean_residual",
    "cand_mean_appearance",
    "cand_local_contrast",
    "cand_texture",
    "cand_line_context",
    "cand_isolation",
    "cand_score",
    "cand_map_score",
    "cand_attached_support",
    "cand_native_dark_score",
    "cand_sky_like",
    "tube_hit_rate",
    "tube_miss_rate",
    "tube_mean_score",
    "tube_score_std",
    "tube_mean_map_score",
    "tube_map_hit_rate",
    "tube_appearance_only_rate",
    "tube_surface_source_rate",
    "tube_router_surface_backed_rate",
    "tube_router_clean_sky_rate",
    "tube_router_boundary_rate",
    "tube_router_line_attached_rate",
    "tube_router_unknown_rate",
    "tube_mean_line_context",
    "tube_max_line_context",
    "tube_mean_attached_support",
    "tube_max_attached_support",
    "tube_mean_native_dark_score",
    "tube_max_native_dark_score",
    "tube_mean_sky_like",
    "tube_mean_texture",
    "tube_mean_residual",
    "tube_mean_appearance",
    "tube_mean_pair_score",
    "tube_positive_pair_rate",
    "tube_mean_align_gain",
    "tube_positive_align_rate",
    "tube_mean_bg_dist",
    "tube_mean_cv_resid",
    "tube_mean_bg_minus_cv",
    "tube_mean_cand_density",
    "tube_log_cand_density",
    "tube_mean_speed",
    "tube_max_speed",
    "tube_mean_accel",
    "tube_max_accel",
    "clba_target_q",
    "clba_bg_q",
    "clba_gain",
    "clba_gain_norm",
    "clba_control_median",
    "clba_control_sigma",
    "clba_path_bg_dist_mean",
    "clba_path_bg_dist_max",
    "clba_target_stack_dark_z",
    "clba_bg_stack_dark_z",
    "clba_target_anisotropy",
    "clba_bg_anisotropy",
    "clba_bg_static_likelihood",
    "clba_attached_likelihood",
    "clba_target_likelihood",
    "clba_used_frames",
    "clba_transform_failures",
)

CROP_FEATURES = (
    "crop_target_q",
    "crop_bg_q",
    "crop_gain",
    "crop_target_stack_dark_z",
    "crop_bg_stack_dark_z",
    "crop_path_bg_dist_mean",
    "crop_used_frames",
    "crop_transform_failures",
    "crop_stack_score",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--disagreements", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--viterbi_dir", required=True)
    p.add_argument("--hmm_dir", required=True)
    p.add_argument("--model", required=True, help="Surface ranker .joblib used to add learned_score to top tubes.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--classifier", choices=("logistic", "hgbdt"), default="logistic")
    p.add_argument("--include_crop_features", action="store_true")
    p.add_argument("--include_branch_context", action="store_true")
    p.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--max_rank", type=int, default=20)
    p.add_argument("--rolling_window", type=int, default=9)
    p.add_argument(
        "--viterbi_protect_streak",
        type=int,
        default=0,
        help=(
            "Force Viterbi when its selected-track streak is at least this many frames. "
            "This probes a production guardrail for continuous-visible shots."
        ),
    )
    p.add_argument(
        "--viterbi_protect_min_score",
        type=float,
        default=0.0,
        help="Minimum Viterbi learned_score required for the continuous-streak protection.",
    )
    p.add_argument(
        "--viterbi_protect_max_bg_risk",
        type=float,
        default=1.0e9,
        help="Maximum top CLBA background risk allowed for continuous-streak protection.",
    )
    p.add_argument(
        "--hmm_enter_count",
        type=int,
        default=1,
        help="Consecutive frames above the threshold required before entering HMM mode.",
    )
    p.add_argument(
        "--hmm_exit_count",
        type=int,
        default=1,
        help="Consecutive frames below the exit threshold, or protected by Viterbi, required before leaving HMM mode.",
    )
    p.add_argument(
        "--hmm_exit_threshold",
        type=float,
        default=None,
        help="Probability threshold for exiting HMM mode. Defaults to the active enter threshold.",
    )
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--random_state", type=int, default=11)
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


def parse_thresholds(raw: str) -> list[float]:
    out = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not out:
        raise SystemExit("no thresholds supplied")
    return out


def active_base_features(include_crop_features: bool) -> tuple[str, ...]:
    return BASE_FEATURES + (CROP_FEATURES if include_crop_features else ())


def fnum(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def mean(values: list[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def labels_by_clip(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        clip = row.get("clip", "")
        frame = tracking_eval.fnum(row.get("frame"))
        if clip and frame is not None:
            grouped[clip].append(row)
    return dict(sorted(grouped.items()))


def selected_by_frame(selector_dir: Path, clip: str) -> dict[int, dict[str, str]]:
    path = selector_dir / clip / "selected_tracks.csv"
    if not path.exists():
        path = selector_dir / clip / "sequence_selected_tracks.csv"
    if not path.exists():
        return {}
    selected: dict[int, dict[str, str]] = {}
    for row in read_csv(path):
        if not tracking_eval.row_is_selected(row):
            continue
        frame = tracking_eval.fnum(row.get("frame"))
        if frame is not None:
            selected[int(frame)] = row
    return selected


def top_tubes_path(results_dir: Path, clip: str) -> Path:
    direct = results_dir / clip / "top_tubes.csv"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"missing top_tubes.csv for {clip}: {direct}")


def bg_risk(row: dict[str, Any]) -> float:
    target = fnum(row.get("clba_target_likelihood"))
    static = fnum(row.get("clba_bg_static_likelihood"))
    attached = fnum(row.get("clba_attached_likelihood"))
    return max(0.0, static - target, attached - target)


def row_feature_summary(rows: list[dict[str, Any]], max_rank: int, base_features: tuple[str, ...]) -> dict[str, float]:
    rows = [row for row in rows if int(fnum(row.get("rank"), 999999)) <= max_rank]
    rows.sort(key=lambda r: int(fnum(r.get("rank"), 999999)))
    features: dict[str, float] = {"cand_count": float(len(rows))}
    if not rows:
        for name in base_features:
            features[f"top_{name}"] = 0.0
        for key in ("learned_score", "bg_risk", "static_minus_target", "attached_minus_target"):
            features[f"top_{key}"] = 0.0
            features[f"mean5_{key}"] = 0.0
            features[f"max5_{key}"] = 0.0
        features["score_margin"] = 0.0
        return features

    top = rows[0]
    top5 = rows[:5]
    for name in base_features:
        features[f"top_{name}"] = fnum(top.get(name))
    learned = [fnum(row.get("learned_score")) for row in top5]
    risks = [bg_risk(row) for row in top5]
    static_minus = [fnum(row.get("clba_bg_static_likelihood")) - fnum(row.get("clba_target_likelihood")) for row in top5]
    attached_minus = [fnum(row.get("clba_attached_likelihood")) - fnum(row.get("clba_target_likelihood")) for row in top5]
    for key, values in (
        ("learned_score", learned),
        ("bg_risk", risks),
        ("static_minus_target", static_minus),
        ("attached_minus_target", attached_minus),
    ):
        features[f"top_{key}"] = values[0] if values else 0.0
        features[f"mean5_{key}"] = mean(values)
        features[f"max5_{key}"] = max(values) if values else 0.0
        features[f"median5_{key}"] = median(values)
    features["score_margin"] = learned[0] - learned[1] if len(learned) > 1 else learned[0]
    return features


def add_rolling_features(per_frame: dict[int, dict[str, float]], window: int) -> dict[int, dict[str, float]]:
    frames = sorted(per_frame)
    history: list[dict[str, float]] = []
    rolling_keys = (
        "top_learned_score",
        "score_margin",
        "top_bg_risk",
        "mean5_bg_risk",
        "top_static_minus_target",
        "top_clba_gain_norm",
        "top_clba_path_bg_dist_mean",
        "top_tube_mean_bg_dist",
        "top_tube_mean_cv_resid",
        "top_tube_mean_cand_density",
    )
    out: dict[int, dict[str, float]] = {}
    for frame in frames:
        current = dict(per_frame[frame])
        history.append(current)
        if len(history) > max(1, window):
            history = history[-window:]
        for key in rolling_keys:
            values = [h.get(key, 0.0) for h in history]
            current[f"roll_mean_{key}"] = mean(values)
            current[f"roll_med_{key}"] = median(values)
            current[f"roll_max_{key}"] = max(values) if values else 0.0
        out[frame] = current
    return out


def matching_ranked_row(
    selected_row: dict[str, str],
    ranked_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected_rank = fnum(selected_row.get("rank"), -1.0)
    if selected_rank >= 0:
        for row in ranked_rows:
            if int(fnum(row.get("rank"), -999.0)) == int(selected_rank):
                return row
    try:
        selected_box = selector.bbox(selected_row)
    except Exception:
        return None
    best_row: dict[str, Any] | None = None
    best_dist = 1.0e9
    for row in ranked_rows:
        try:
            dist = selector.center_dist(selected_box, selector.bbox(row))
        except Exception:
            continue
        if dist < best_dist:
            best_dist = dist
            best_row = row
    return best_row if best_dist <= 2.0 else None


def selected_row_feature_summary(row: dict[str, Any] | None, prefix: str, base_features: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in base_features:
        out[f"{prefix}_sel_{name}"] = fnum(row.get(name)) if row is not None else 0.0
    if row is None:
        out[f"{prefix}_sel_learned_score"] = 0.0
        out[f"{prefix}_sel_bg_risk"] = 0.0
        out[f"{prefix}_sel_static_minus_target"] = 0.0
        out[f"{prefix}_sel_attached_minus_target"] = 0.0
        return out
    out[f"{prefix}_sel_learned_score"] = fnum(row.get("learned_score"))
    out[f"{prefix}_sel_bg_risk"] = bg_risk(row)
    out[f"{prefix}_sel_static_minus_target"] = fnum(row.get("clba_bg_static_likelihood")) - fnum(
        row.get("clba_target_likelihood")
    )
    out[f"{prefix}_sel_attached_minus_target"] = fnum(row.get("clba_attached_likelihood")) - fnum(
        row.get("clba_target_likelihood")
    )
    return out


def branch_features(
    selected: dict[int, dict[str, str]],
    ranked_by_frame: dict[int, list[dict[str, Any]]],
    frames: list[int],
    prefix: str,
    base_features: tuple[str, ...],
    include_context: bool,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    last_row: dict[str, str] | None = None
    last_frame: int | None = None
    streak = 0
    for frame in sorted(frames):
        row = selected.get(frame)
        values: dict[str, float] = {
            f"{prefix}_selected": 0.0,
            f"{prefix}_score": 0.0,
            f"{prefix}_rank": 999.0,
            f"{prefix}_jump_px": 999.0,
            f"{prefix}_gap": 999.0,
            f"{prefix}_streak": float(streak),
        }
        if row is not None:
            values[f"{prefix}_selected"] = 1.0
            values[f"{prefix}_score"] = fnum(row.get("learned_score"))
            values[f"{prefix}_rank"] = fnum(row.get("rank"), 999.0)
            if include_context:
                values.update(
                    selected_row_feature_summary(
                        matching_ranked_row(row, ranked_by_frame.get(frame, [])),
                        prefix,
                        base_features,
                    )
                )
            if last_row is not None and last_frame is not None:
                values[f"{prefix}_jump_px"] = selector.center_dist(selector.bbox(last_row), selector.bbox(row))
                values[f"{prefix}_gap"] = max(1.0, frame - last_frame)
            streak += 1
            last_row = row
            last_frame = frame
        else:
            if include_context:
                values.update(selected_row_feature_summary(None, prefix, base_features))
            streak = 0
        values[f"{prefix}_streak"] = float(streak)
        out[frame] = values
    return out


def build_features(
    clips: list[str],
    labels: dict[str, list[dict[str, str]]],
    results_dir: Path,
    model_path: Path,
    viterbi_dir: Path,
    hmm_dir: Path,
    max_rank: int,
    rolling_window: int,
    base_features: tuple[str, ...],
    include_branch_context: bool,
) -> tuple[dict[tuple[str, int], dict[str, float]], list[str]]:
    all_features: dict[tuple[str, int], dict[str, float]] = {}
    feature_names: list[str] = []
    for clip in clips:
        rows = selector.load_ranked_rows(top_tubes_path(results_dir, clip), max_rank)
        scored, _ = selector.score_rows(rows, model_path)
        by_frame = selector.group_by_frame(scored)
        frame_features = {
            frame: row_feature_summary(frame_rows, max_rank, base_features) for frame, frame_rows in by_frame.items()
        }
        frame_features = add_rolling_features(frame_features, rolling_window)
        label_frames = sorted({int(float(row["frame"])) for row in labels.get(clip, [])})
        frames = sorted(set(frame_features) | set(label_frames))
        vit = branch_features(
            selected_by_frame(viterbi_dir, clip),
            by_frame,
            frames,
            "viterbi",
            base_features,
            include_branch_context,
        )
        hmm = branch_features(
            selected_by_frame(hmm_dir, clip),
            by_frame,
            frames,
            "hmm",
            base_features,
            include_branch_context,
        )
        for frame in frames:
            features = dict(frame_features.get(frame, {}))
            features.update(vit.get(frame, {}))
            features.update(hmm.get(frame, {}))
            features["branch_score_delta_vit_minus_hmm"] = features.get("viterbi_score", 0.0) - features.get("hmm_score", 0.0)
            features["branch_rank_delta_vit_minus_hmm"] = features.get("viterbi_rank", 999.0) - features.get("hmm_rank", 999.0)
            all_features[(clip, frame)] = features
            for key in features:
                if key not in feature_names:
                    feature_names.append(key)
    return all_features, feature_names


def disagreement_examples(
    path: Path,
    features: dict[tuple[str, int], dict[str, float]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    examples: list[dict[str, Any]] = []
    for row in read_csv(path):
        category = row.get("category")
        if category == POSITIVE_HMM_CATEGORY:
            y = 1
        elif category == NEGATIVE_VITERBI_CATEGORY:
            y = 0
        else:
            continue
        clip = row["clip"]
        frame = int(float(row["frame"]))
        if (clip, frame) not in features:
            continue
        examples.append({"clip": clip, "frame": frame, "category": category, "y": y})
    y_arr = np.asarray([int(row["y"]) for row in examples], dtype=np.int32)
    clips = np.asarray([str(row["clip"]) for row in examples])
    frames = np.asarray([int(row["frame"]) for row in examples], dtype=np.int32)
    return examples, y_arr, clips, frames


def vectorize(keys: list[tuple[str, int]], features: dict[tuple[str, int], dict[str, float]], feature_names: list[str]) -> np.ndarray:
    x = np.zeros((len(keys), len(feature_names)), dtype=np.float32)
    for i, key in enumerate(keys):
        row = features.get(key, {})
        for j, name in enumerate(feature_names):
            x[i, j] = float(row.get(name, 0.0) or 0.0)
    return x


def make_model(kind: str, random_state: int):
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        )
    if kind == "hgbdt":
        return HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.06,
            max_leaf_nodes=8,
            l2_regularization=0.1,
            random_state=random_state,
        )
    raise ValueError(kind)


def selector_row(selected: dict[int, dict[str, str]], frame: int) -> dict[str, str] | None:
    return selected.get(frame)


def protected_viterbi(features: dict[str, float], args: argparse.Namespace) -> bool:
    if args.viterbi_protect_streak <= 0:
        return False
    if features.get("viterbi_selected", 0.0) < 0.5:
        return False
    if features.get("viterbi_streak", 0.0) < args.viterbi_protect_streak:
        return False
    if features.get("viterbi_score", 0.0) < args.viterbi_protect_min_score:
        return False
    if features.get("top_bg_risk", 0.0) > args.viterbi_protect_max_bg_risk:
        return False
    return True


def evaluate_routed_frame(
    label: dict[str, str],
    row: dict[str, str] | None,
    strict_tol: float,
    loose_tol: float,
) -> dict[str, Any]:
    visible = tracking_eval.visible(label)
    label_box = tracking_eval.label_bbox(label) if visible else None
    selected_box = tracking_eval.row_bbox(row) if row is not None else None
    dist = None
    strict = False
    loose = False
    if visible and label_box is not None and selected_box is not None:
        dist = tracking_eval.center_dist(label_box, selected_box)
        strict = dist <= strict_tol
        loose = dist <= loose_tol
    return {
        "visible": int(visible),
        "selected": int(row is not None),
        "strict_hit": int(strict),
        "loose_hit": int(loose),
        "dist_px": "" if dist is None else round(dist, 3),
    }


def aggregate_eval(rows: list[dict[str, Any]], mode: str, threshold: float) -> dict[str, Any]:
    visible = sum(int(row["visible"]) for row in rows)
    strict = sum(int(row["strict_hit"]) for row in rows)
    loose = sum(int(row["loose_hit"]) for row in rows)
    invisible = sum(1 for row in rows if not int(row["visible"]))
    invisible_no_box = sum(1 for row in rows if not int(row["visible"]) and not int(row["selected"]))
    return {
        "mode": mode,
        "threshold": threshold,
        "frames": len(rows),
        "visible_frames": visible,
        "strict_hits": strict,
        "strict_recall": round(strict / max(1, visible), 4),
        "loose_hits": loose,
        "loose_recall": round(loose / max(1, visible), 4),
        "invisible_frames": invisible,
        "invisible_no_box": invisible_no_box,
        "invisible_no_box_rate": round(invisible_no_box / max(1, invisible), 4),
        "selected_frames_total": sum(int(row["selected"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = labels_by_clip(Path(args.labels))
    clips = sorted(labels)
    base_features = active_base_features(args.include_crop_features)
    features, feature_names = build_features(
        clips=clips,
        labels=labels,
        results_dir=Path(args.results_dir),
        model_path=Path(args.model),
        viterbi_dir=Path(args.viterbi_dir),
        hmm_dir=Path(args.hmm_dir),
        max_rank=args.max_rank,
        rolling_window=args.rolling_window,
        base_features=base_features,
        include_branch_context=args.include_branch_context,
    )
    examples, y, example_clips, _ = disagreement_examples(Path(args.disagreements), features)
    if len(set(y.tolist())) < 2:
        raise SystemExit("need both HMM-positive and Viterbi-positive disagreement examples")

    thresholds = parse_thresholds(args.thresholds)
    example_rows: list[dict[str, Any]] = []
    frame_eval_rows: list[dict[str, Any]] = []
    by_clip_summary: list[dict[str, Any]] = []
    threshold_eval: dict[float, list[dict[str, Any]]] = {thr: [] for thr in thresholds}

    example_clip_set = set(example_clips.tolist())
    for holdout_clip in clips:
        if holdout_clip in example_clip_set:
            train_rows = [row for row in examples if row["clip"] != holdout_clip]
        else:
            train_rows = list(examples)
        train_keys = [(row["clip"], int(row["frame"])) for row in train_rows]
        train_y = np.asarray([int(row["y"]) for row in train_rows], dtype=np.int32)
        if len(set(train_y.tolist())) < 2:
            continue
        model = make_model(args.classifier, args.random_state)
        model.fit(vectorize(train_keys, features, feature_names), train_y)

        # OOF probabilities for disagreement examples.
        if holdout_clip in example_clip_set:
            holdout_examples = [row for row in examples if row["clip"] == holdout_clip]
            holdout_keys = [(row["clip"], int(row["frame"])) for row in holdout_examples]
            holdout_probs = model.predict_proba(vectorize(holdout_keys, features, feature_names))[:, 1]
            for row, prob in zip(holdout_examples, holdout_probs):
                example_rows.append({**row, "hmm_probability": round(float(prob), 6)})

        # Apply the same held-out model to every labeled frame in this clip.
        label_keys = [(holdout_clip, int(float(row["frame"]))) for row in labels[holdout_clip]]
        label_probs = model.predict_proba(vectorize(label_keys, features, feature_names))[:, 1]
        prob_map = {frame: float(prob) for (_, frame), prob in zip(label_keys, label_probs)}
        viterbi_selected = selected_by_frame(Path(args.viterbi_dir), holdout_clip)
        hmm_selected = selected_by_frame(Path(args.hmm_dir), holdout_clip)
        for threshold in thresholds:
            clip_rows: list[dict[str, Any]] = []
            hmm_mode = False
            enter_run = 0
            exit_run = 0
            exit_threshold = threshold if args.hmm_exit_threshold is None else args.hmm_exit_threshold
            for label in sorted(labels[holdout_clip], key=lambda r: int(float(r["frame"]))):
                frame = int(float(label["frame"]))
                prob = prob_map.get(frame, 0.0)
                frame_features = features.get((holdout_clip, frame), {})
                protect_viterbi = protected_viterbi(frame_features, args)
                wants_hmm = prob >= threshold and not protect_viterbi
                wants_viterbi = protect_viterbi or prob < exit_threshold
                if hmm_mode:
                    if wants_viterbi:
                        exit_run += 1
                        enter_run = 0
                    else:
                        exit_run = 0
                    if exit_run >= max(1, args.hmm_exit_count):
                        hmm_mode = False
                        exit_run = 0
                else:
                    if wants_hmm:
                        enter_run += 1
                        exit_run = 0
                    else:
                        enter_run = 0
                    if enter_run >= max(1, args.hmm_enter_count):
                        hmm_mode = True
                        enter_run = 0
                use_hmm = hmm_mode
                selected = selector_row(hmm_selected if use_hmm else viterbi_selected, frame)
                ev = evaluate_routed_frame(label, selected, args.strict_tol_px, args.loose_tol_px)
                row = {
                    "clip": holdout_clip,
                    "frame": frame,
                    "threshold": threshold,
                    "hmm_probability": round(prob, 6),
                    "chosen_selector": "hmm" if use_hmm else "viterbi",
                    "protected_viterbi": int(protect_viterbi),
                    "hmm_enter_count": args.hmm_enter_count,
                    "hmm_exit_count": args.hmm_exit_count,
                    **ev,
                }
                clip_rows.append(row)
                threshold_eval[threshold].append(row)
            by_clip_summary.append({"clip": holdout_clip, **aggregate_eval(clip_rows, "supervisor", threshold)})
            frame_eval_rows.extend(clip_rows)

    summary_rows = [aggregate_eval(rows, "supervisor", threshold) for threshold, rows in threshold_eval.items()]
    summary_rows.sort(key=lambda r: (r["threshold"]))

    y_true = [int(row["y"]) for row in example_rows]
    probs = [float(row["hmm_probability"]) for row in example_rows]
    pred = [int(p >= 0.5) for p in probs]
    example_metrics = {
        "examples": len(example_rows),
        "hmm_positive_examples": int(sum(y_true)),
        "viterbi_positive_examples": int(len(y_true) - sum(y_true)),
        "oof_auc": round(roc_auc_score(y_true, probs), 4) if len(set(y_true)) > 1 else "",
        "oof_accuracy_at_0p5": round(accuracy_score(y_true, pred), 4) if y_true else "",
        "oof_balanced_accuracy_at_0p5": round(balanced_accuracy_score(y_true, pred), 4) if y_true else "",
    }

    write_csv(out_dir / "mode_supervisor_summary.csv", summary_rows)
    write_csv(out_dir / "mode_supervisor_by_clip.csv", by_clip_summary)
    write_csv(out_dir / "mode_supervisor_example_oof.csv", example_rows)
    write_csv(out_dir / "mode_supervisor_frame_eval.csv", frame_eval_rows)
    (out_dir / "mode_supervisor_metadata.json").write_text(
        json.dumps(
            {
                "labels": args.labels,
                "disagreements": args.disagreements,
                "results_dir": args.results_dir,
                "viterbi_dir": args.viterbi_dir,
                "hmm_dir": args.hmm_dir,
                "model": args.model,
                "classifier": args.classifier,
                "max_rank": args.max_rank,
                "rolling_window": args.rolling_window,
                "include_crop_features": args.include_crop_features,
                "include_branch_context": args.include_branch_context,
                "thresholds": thresholds,
                "feature_names": feature_names,
                "example_metrics": example_metrics,
            },
            indent=2,
        )
    )
    print(json.dumps({"example_metrics": example_metrics, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
