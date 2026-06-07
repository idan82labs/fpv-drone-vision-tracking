#!/usr/bin/env python3
"""Build the SRPS-ID58 runtime source-promoter training table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = "artifacts/srps_id50_pi_source_distillation_rows_topk_export80_v1/candidate_rows.csv"
DEFAULT_PROVENANCE = "artifacts/srps_id58_source_promoter_provenance_v1/provenance_matches.csv"
DEFAULT_OUT = "artifacts/srps_id58_runtime_source_promoter_dataset_v1"


STRICT_COL = "label_strict_8px"
LOOSE_COL = "label_loose_16px"

FORBIDDEN_FEATURES = {
    "candidate_id",
    "clip",
    "dataset",
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "cx",
    "cy",
    "det_x",
    "det_y",
    "det_w",
    "det_h",
    "label_cx",
    "label_cy",
    "distance_to_reviewed_center",
    "label_strict_8px",
    "label_loose_16px",
    "teacher_distance",
    "teacher_selected",
    "teacher_source_family",
    "teacher_source",
    "pi_distance",
    "pi_selected",
    "pi_selected_source_family",
    "hard_negative",
    "promote_y_strict",
    "promote_y_loose",
    "promote_label",
    "is_ambiguous_near",
    "is_external_positive_source",
}

CATEGORICAL_HINTS = {
    "source_family",
    "source",
    "background_bucket",
    "label_confidence",
    "split_group",
    "cand_source",
    "cand_router_state",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate_rows", default=DEFAULT_CANDIDATES)
    p.add_argument("--provenance_matches", default=DEFAULT_PROVENANCE)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--reject_min_px", type=float, default=24.0)
    p.add_argument("--focus_dataset", default="e271")
    p.add_argument("--focus_frame_min", type=int, default=654)
    p.add_argument("--focus_frame_max", type=int, default=698)
    return p.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def promote_label(row: dict[str, str], args: argparse.Namespace) -> str:
    dist = safe_float(row.get("distance_to_reviewed_center"), 1.0e9)
    if dist <= args.strict_tol_px:
        return "PROMOTE_STRICT"
    if dist <= args.loose_tol_px:
        return "PROMOTE_LOOSE"
    if dist < args.reject_min_px:
        return "AMBIGUOUS_NEAR"
    return "REJECT_WRONG"


def feature_columns(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    if not rows:
        return {"numeric": [], "categorical": [], "columns": []}
    for col in rows[0].keys():
        if col in FORBIDDEN_FEATURES:
            continue
        if col in CATEGORICAL_HINTS:
            categorical.append(col)
            continue
        values = [row.get(col, "") for row in rows]
        convertible = 0
        for value in values:
            if str(value).strip() == "":
                continue
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(out):
                convertible += 1
        if convertible >= max(2, int(0.25 * len(rows))):
            numeric.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical, "columns": numeric + categorical}


def frame_availability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (dataset, frame), group in sorted(group_by(rows, ["dataset", "frame"]).items()):
        strict = any(safe_int(r.get(STRICT_COL)) == 1 for r in group)
        loose = any(safe_int(r.get(LOOSE_COL)) == 1 for r in group)
        out.append(
            {
                "dataset": dataset,
                "frame": frame,
                "candidate_rows": len(group),
                "strict_available": int(strict),
                "loose_available": int(loose),
                "best_distance": round(min(safe_float(r.get("distance_to_reviewed_center"), 1.0e9) for r in group), 4),
            }
        )
    return out


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        out.setdefault(key, []).append(row)
    return out


def split_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counter[(str(row.get("dataset", "")), str(row.get("promote_label", "")))] += 1
    return [
        {"dataset": dataset, "promote_label": label, "rows": count}
        for (dataset, label), count in sorted(counter.items())
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_rows = read_csv(Path(args.candidate_rows))
    if not candidate_rows:
        raise SystemExit(f"no candidate rows in {args.candidate_rows}")
    provenance_rows = read_csv(Path(args.provenance_matches))
    provenance_ids = {
        str(r.get("candidate_id", ""))
        for r in provenance_rows
        if safe_int(r.get("in_pi_candidate_rows")) == 1 and (safe_int(r.get("label_strict_8px")) == 1 or safe_int(r.get("label_loose_16px")) == 1)
    }

    out_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        label = promote_label(row, args)
        strict = int(label == "PROMOTE_STRICT")
        loose = int(label in {"PROMOTE_STRICT", "PROMOTE_LOOSE"})
        out = dict(row)
        out["promote_label"] = label
        out["promote_y_strict"] = strict
        out["promote_y_loose"] = loose
        out["is_ambiguous_near"] = int(label == "AMBIGUOUS_NEAR")
        out["is_external_positive_source"] = int(str(row.get("candidate_id", "")) in provenance_ids)
        out_rows.append(out)

    feature_payload = feature_columns(out_rows)
    leakage_columns = sorted(set(feature_payload["columns"]) & FORBIDDEN_FEATURES)
    availability = frame_availability(out_rows)
    focus_rows = [
        r
        for r in out_rows
        if r.get("dataset") == args.focus_dataset and args.focus_frame_min <= safe_int(r.get("frame")) <= args.focus_frame_max
    ]
    focus_avail = frame_availability(focus_rows)
    focus_strict = sum(safe_int(r.get("strict_available")) for r in focus_avail)
    focus_loose = sum(safe_int(r.get("loose_available")) for r in focus_avail)
    focus_frames = len(focus_avail)
    regression_datasets = sorted({str(r.get("dataset", "")) for r in out_rows if str(r.get("dataset", "")) != args.focus_dataset})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "source_promoter_rows.csv", out_rows)
    write_csv(out_dir / "frame_candidate_availability.csv", availability)
    write_csv(out_dir / "label_distribution.csv", split_counts(out_rows))
    write_json(out_dir / "trainable_feature_columns.json", feature_payload)
    summary = {
        "artifact": "srps_id58_runtime_source_promoter_dataset_v1",
        "source_rows": len(out_rows),
        "feature_columns": len(feature_payload["columns"]),
        "leakage_columns": leakage_columns,
        "leakage_column_count": len(leakage_columns),
        "focus_dataset": args.focus_dataset,
        "focus_frame_min": args.focus_frame_min,
        "focus_frame_max": args.focus_frame_max,
        "focus_frames": focus_frames,
        "focus_strict_available_frames": focus_strict,
        "focus_loose_available_frames": focus_loose,
        "external_positive_ids_joined": len(provenance_ids),
        "regression_datasets": regression_datasets,
        "phase_b_source_gate_pass": int(focus_strict >= 35 and focus_loose >= 42 and not leakage_columns and bool(regression_datasets)),
    }
    write_json(out_dir / "summary.json", summary)
    readme = f"""# SRPS-ID58 Runtime Source-Promoter Dataset

This table keeps only Pi-computable source candidates as training rows. Label
columns and true-box geometry are retained for evaluation but excluded from
`trainable_feature_columns.json`.

- phase B source gate pass: `{summary['phase_b_source_gate_pass']}`
- focus strict/loose availability: `{focus_strict}` / `{focus_loose}` of `{focus_frames}`
- trainable feature columns: `{summary['feature_columns']}`
- leakage columns in feature schema: `{summary['leakage_column_count']}`

Files:

- `source_promoter_rows.csv`
- `frame_candidate_availability.csv`
- `label_distribution.csv`
- `trainable_feature_columns.json`
- `summary.json`
"""
    (out_dir / "README.md").write_text(readme)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
