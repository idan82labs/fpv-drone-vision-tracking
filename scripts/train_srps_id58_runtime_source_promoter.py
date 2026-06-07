#!/usr/bin/env python3
"""Train/evaluate SRPS-ID58 runtime source-promoter models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROWS = "artifacts/srps_id58_runtime_source_promoter_dataset_v1/source_promoter_rows.csv"
DEFAULT_FEATURES = "artifacts/srps_id58_runtime_source_promoter_dataset_v1/trainable_feature_columns.json"
DEFAULT_DATASET_SUMMARY = "artifacts/srps_id58_runtime_source_promoter_dataset_v1/summary.json"
DEFAULT_OUT = "artifacts/srps_id58_runtime_source_promoter_loose_v1"

TARGETS = {"loose": "promote_y_loose", "strict": "promote_y_strict"}
STRICT_COL = "label_strict_8px"
LOOSE_COL = "label_loose_16px"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_rows", default=DEFAULT_ROWS)
    p.add_argument("--feature_columns", default=DEFAULT_FEATURES)
    p.add_argument("--dataset_summary", default=DEFAULT_DATASET_SUMMARY)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--target", choices=sorted(TARGETS), default="loose")
    p.add_argument("--random_state", type=int, default=58)
    p.add_argument("--top_k_eval", type=int, default=3)
    p.add_argument("--focus_dataset", default="e271")
    p.add_argument("--focus_frame_min", type=int, default=654)
    p.add_argument("--focus_frame_max", type=int, default=698)
    p.add_argument("--min_train_rows", type=int, default=20)
    return p.parse_args(argv)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def load_feature_schema(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text())
    numeric = [str(c) for c in payload.get("numeric", [])]
    categorical = [str(c) for c in payload.get("categorical", [])]
    return numeric, categorical


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.source_rows)
    if df.empty:
        raise SystemExit(f"no rows in {args.source_rows}")
    for col in [TARGETS[args.target], STRICT_COL, LOOSE_COL, "frame", "rank"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Ambiguous near rows remain in prediction exports but are not train targets.
    trainable = pd.to_numeric(df.get("is_ambiguous_near", 0), errors="coerce").fillna(0).astype(int) == 0
    df["is_trainable_source_promoter_row"] = trainable.astype(int)
    return df


def make_preprocessor(numeric: list[str], categorical: list[str], *, scale_numeric: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_models(numeric: list[str], categorical: list[str], random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("features", make_preprocessor(numeric, categorical, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                ("features", make_preprocessor(numeric, categorical, scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.055,
                        l2_regularization=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extratrees": Pipeline(
            [
                ("features", make_preprocessor(numeric, categorical, scale_numeric=False)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=320,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def score_model(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(df)
    classes = [str(c) for c in model.classes_]
    if "1" not in classes:
        return np.zeros(len(df), dtype=float)
    return proba[:, classes.index("1")]


def baseline_score(df: pd.DataFrame) -> np.ndarray:
    score = pd.to_numeric(df.get("score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    verified = pd.to_numeric(df.get("verified_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rank = pd.to_numeric(df.get("rank", 999.0), errors="coerce").fillna(999.0).to_numpy(dtype=float)
    return score + 0.18 * verified - 0.12 * np.log1p(rank)


def longest_bad_run(frame_rows: list[dict[str, Any]]) -> int:
    run = best = 0
    for row in sorted(frame_rows, key=lambda r: int(r["frame"])):
        if int(row["top1_loose_hit"]) == 1:
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


def frame_rank_metrics(df: pd.DataFrame, scores: np.ndarray, top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    work = df.copy()
    work["_source_promoter_score"] = scores
    rows: list[dict[str, Any]] = []
    frames = top1_strict = top1_loose = topk_strict = topk_loose = 0
    for (dataset, clip, frame), group in work.groupby(["dataset", "clip", "frame"], sort=True):
        frames += 1
        ranked = group.sort_values("_source_promoter_score", ascending=False)
        top1 = ranked.iloc[0]
        topn = ranked.head(top_k)
        strict = int(safe_int(top1.get(STRICT_COL)) == 1)
        loose = int(safe_int(top1.get(LOOSE_COL)) == 1)
        top1_strict += strict
        top1_loose += loose
        topk_strict += int((topn[STRICT_COL].astype(int) == 1).any())
        topk_loose += int((topn[LOOSE_COL].astype(int) == 1).any())
        rows.append(
            {
                "dataset": dataset,
                "clip": clip,
                "frame": int(frame),
                "top1_candidate_id": top1.get("candidate_id", ""),
                "top1_score": round(float(top1["_source_promoter_score"]), 8),
                "top1_rank": top1.get("rank", ""),
                "top1_source_family": top1.get("source_family", ""),
                "top1_source": top1.get("source", ""),
                "top1_distance": top1.get("distance_to_reviewed_center", ""),
                "top1_strict_hit": strict,
                "top1_loose_hit": loose,
                f"top{top_k}_strict_available": int((topn[STRICT_COL].astype(int) == 1).any()),
                f"top{top_k}_loose_available": int((topn[LOOSE_COL].astype(int) == 1).any()),
            }
        )
    metrics = {
        "frames": frames,
        "top1_strict": top1_strict,
        "top1_loose": top1_loose,
        f"top{top_k}_strict": topk_strict,
        f"top{top_k}_loose": topk_loose,
        "longest_wrong_loose_run": longest_bad_run(rows),
    }
    return metrics, rows


def quality(df: pd.DataFrame, scores: np.ndarray, target_col: str) -> dict[str, Any]:
    y = df[target_col].astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        return {"auc": "", "ap": ""}
    return {
        "auc": round(float(roc_auc_score(y, scores)), 5),
        "ap": round(float(average_precision_score(y, scores)), 5),
    }


def split_masks(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, tuple[pd.Series, pd.Series]]:
    is_focus = (
        df["dataset"].eq(args.focus_dataset)
        & (df["frame"].astype(int) >= args.focus_frame_min)
        & (df["frame"].astype(int) <= args.focus_frame_max)
    )
    focus_a = is_focus & (df["frame"].astype(int) <= args.focus_frame_min + 14)
    focus_b = is_focus & (df["frame"].astype(int) > args.focus_frame_min + 14) & (df["frame"].astype(int) <= args.focus_frame_min + 29)
    focus_c = is_focus & (df["frame"].astype(int) > args.focus_frame_min + 29)
    return {
        "all_fit": (df["is_trainable_source_promoter_row"].eq(1), pd.Series(True, index=df.index)),
        "focus_all_fit": (df["is_trainable_source_promoter_row"].eq(1), is_focus),
        "focus_holdout": (df["is_trainable_source_promoter_row"].eq(1) & ~is_focus, is_focus),
        "focus_block_a_to_bc": (df["is_trainable_source_promoter_row"].eq(1) & ~focus_a, focus_a),
        "focus_block_b_to_ac": (df["is_trainable_source_promoter_row"].eq(1) & ~focus_b, focus_b),
        "focus_block_c_to_ab": (df["is_trainable_source_promoter_row"].eq(1) & ~focus_c, focus_c),
    }


def train_eval(
    model_name: str,
    model: Pipeline | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, Pipeline | None]:
    if model_name == "baseline_score":
        scores = baseline_score(test)
        fitted = None
    else:
        y = train[target_col].astype(int)
        if len(set(y.tolist())) < 2:
            raise ValueError("training split has only one target class")
        fitted = model.fit(train, y)
        scores = score_model(fitted, test)
    metrics, frame_rows = frame_rank_metrics(test, scores, top_k)
    metrics.update(quality(test, scores, target_col))
    metrics.update(
        {
            "model": model_name,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_positives": int(train[target_col].astype(int).sum()) if target_col in train else "",
            "test_positives": int(test[target_col].astype(int).sum()) if target_col in test else "",
        }
    )
    return metrics, frame_rows, scores, fitted


def gate_summary(summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    focus_rows = [r for r in summary_rows if r.get("split") == "focus_all_fit" and r.get("model") != "baseline_score"]
    if not focus_rows:
        return {"phase_c_source_promoter_gate_pass": 0, "phase_c_failures": "missing focus_all_fit models"}
    best = max(
        focus_rows,
        key=lambda r: (
            int(r.get("top1_strict", 0)),
            int(r.get("top1_loose", 0)),
            int(r.get(f"top{args.top_k_eval}_strict", 0)),
            int(r.get(f"top{args.top_k_eval}_loose", 0)),
        ),
    )
    failures: list[str] = []
    if args.target == "loose":
        if int(best["top1_loose"]) < 38:
            failures.append(f"all-fit top1 loose {best['top1_loose']} < 38")
        if int(best[f"top{args.top_k_eval}_loose"]) < 42:
            failures.append(f"all-fit top{args.top_k_eval} loose {best[f'top{args.top_k_eval}_loose']} < 42")
    else:
        if int(best["top1_strict"]) < 25:
            failures.append(f"all-fit top1 strict {best['top1_strict']} < 25")
        if int(best[f"top{args.top_k_eval}_strict"]) < 35:
            failures.append(f"all-fit top{args.top_k_eval} strict {best[f'top{args.top_k_eval}_strict']} < 35")
    if int(best["longest_wrong_loose_run"]) > 1:
        failures.append(f"longest wrong loose run {best['longest_wrong_loose_run']} > 1")
    return {
        "phase_c_source_promoter_gate_pass": int(not failures),
        "phase_c_failures": "; ".join(failures),
        "best_policy_model": best["model"],
        "best_policy_split": best["split"],
        "best_policy_top1_strict": int(best["top1_strict"]),
        "best_policy_top1_loose": int(best["top1_loose"]),
        f"best_policy_top{args.top_k_eval}_strict": int(best[f"top{args.top_k_eval}_strict"]),
        f"best_policy_top{args.top_k_eval}_loose": int(best[f"top{args.top_k_eval}_loose"]),
        "best_policy_longest_wrong_loose_run": int(best["longest_wrong_loose_run"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(args)
    numeric, categorical = load_feature_schema(Path(args.feature_columns))
    for col in numeric + categorical:
        if col not in df:
            df[col] = np.nan
    target_col = TARGETS[args.target]
    models = make_models(numeric, categorical, args.random_state)
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    frame_rows_out: list[dict[str, Any]] = []
    saved_policies: dict[str, str] = {}

    for split, (train_mask, test_mask) in split_masks(df, args).items():
        train = df[train_mask].copy()
        test = df[test_mask].copy()
        if train.empty or test.empty or len(train) < args.min_train_rows:
            continue
        for model_name in ["baseline_score", "logistic", "hgb", "extratrees"]:
            try:
                metrics, frame_rows, scores, fitted = train_eval(
                    model_name,
                    None if model_name == "baseline_score" else models[model_name],
                    train,
                    test,
                    target_col,
                    args.top_k_eval,
                )
            except ValueError as exc:
                summary_rows.append({"split": split, "model": model_name, "error": str(exc), "train_rows": len(train), "test_rows": len(test)})
                continue
            metrics["split"] = split
            metrics["target"] = args.target
            summary_rows.append(metrics)
            for row in frame_rows:
                row.update({"split": split, "model": model_name, "target": args.target})
                frame_rows_out.append(row)
            for i, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "split": split,
                        "model": model_name,
                        "target": args.target,
                        "dataset": row.get("dataset", ""),
                        "clip": row.get("clip", ""),
                        "frame": safe_int(row.get("frame")),
                        "candidate_id": row.get("candidate_id", ""),
                        "rank": row.get("rank", ""),
                        "source_family": row.get("source_family", ""),
                        "source": row.get("source", ""),
                        "source_promoter_score": round(float(scores[i]), 8),
                        "promote_y_strict": safe_int(row.get("promote_y_strict")),
                        "promote_y_loose": safe_int(row.get("promote_y_loose")),
                        "label_strict_8px": safe_int(row.get(STRICT_COL)),
                        "label_loose_16px": safe_int(row.get(LOOSE_COL)),
                        "distance_to_reviewed_center": row.get("distance_to_reviewed_center", ""),
                    }
                )
            if fitted is not None and split in {"all_fit", "focus_all_fit"}:
                model_path = model_dir / f"{model_name}_{args.target}_{split}.joblib"
                joblib.dump(fitted, model_path)
                policy_path = model_dir / f"{model_name}_{args.target}_{split}_policy.json"
                policy = {
                    "model_path": str(model_path),
                    "threshold": 0.0,
                    "feature_columns": numeric + categorical,
                    "numeric_feature_columns": numeric,
                    "categorical_feature_columns": categorical,
                    "score_column": f"srps_id58_{args.target}_source_promoter_score",
                    "target": args.target,
                }
                write_json(policy_path, policy)
                if split == "focus_all_fit":
                    saved_policies[model_name] = str(policy_path)

    gate = gate_summary(summary_rows, args)
    if gate.get("best_policy_model") in saved_policies:
        gate["best_policy_path"] = saved_policies[str(gate["best_policy_model"])]

    dataset_summary: dict[str, Any] = {}
    dataset_summary_path = Path(args.dataset_summary)
    if dataset_summary_path.exists():
        dataset_summary = json.loads(dataset_summary_path.read_text())

    payload = {
        "artifact": Path(args.out_dir).name,
        "target": args.target,
        "rows": len(df),
        "feature_columns": len(numeric) + len(categorical),
        "dataset_source_gate_pass": dataset_summary.get("phase_b_source_gate_pass", ""),
        "gate": gate,
    }
    write_csv(out_dir / "model_summary.csv", summary_rows)
    write_csv(out_dir / "prediction_rows.csv", prediction_rows)
    write_csv(out_dir / "frame_rank_metrics.csv", frame_rows_out)
    write_json(out_dir / "summary.json", payload)
    readme = f"""# SRPS-ID58 Runtime Source Promoter ({args.target})

Trains source-promotion models over the ID58 Pi-computable candidate table.

- phase C gate pass: `{gate.get('phase_c_source_promoter_gate_pass', 0)}`
- failures: `{gate.get('phase_c_failures', '')}`
- best policy: `{gate.get('best_policy_model', '')}`
- best policy path: `{gate.get('best_policy_path', '')}`

Files:

- `model_summary.csv`
- `prediction_rows.csv`
- `frame_rank_metrics.csv`
- `models/*_policy.json`
- `summary.json`
"""
    (out_dir / "README.md").write_text(readme)
    return payload


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
