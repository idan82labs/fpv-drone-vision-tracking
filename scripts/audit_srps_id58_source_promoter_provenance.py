#!/usr/bin/env python3
"""Audit whether ID57 external positives exist in Pi-computable source rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXTERNAL = "artifacts/srps_id51_balanced_source_label_packet_full_aaf1_v1/review_candidates.csv"
DEFAULT_CANDIDATES = "artifacts/srps_id50_pi_source_distillation_rows_topk_export80_v1/candidate_rows.csv"
DEFAULT_PREDICTIONS = "artifacts/srps_id52_multiclass_source_promoter_bootstrap_full_aaf1_v1/prediction_rows.csv"
DEFAULT_RUNTIME = "artifacts/srps_id57_runtime_reacquisition_rows_f654_698_v1/runtime_reacquisition_rows.csv"
DEFAULT_OUT = "artifacts/srps_id58_source_promoter_provenance_v1"


LEAKAGE_COLS = {
    "label_cx",
    "label_cy",
    "distance_to_reviewed_center",
    "label_strict_8px",
    "label_loose_16px",
    "teacher_selected",
    "teacher_distance",
    "pi_selected",
    "pi_distance",
    "review_reason",
    "review_role",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external_candidates", default=DEFAULT_EXTERNAL)
    p.add_argument("--pi_candidate_rows", default=DEFAULT_CANDIDATES)
    p.add_argument("--prediction_rows", default=DEFAULT_PREDICTIONS)
    p.add_argument("--runtime_rows", default=DEFAULT_RUNTIME)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--dataset", default="e271")
    p.add_argument("--frame_min", type=int, default=654)
    p.add_argument("--frame_max", type=int, default=698)
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


def in_focus(row: dict[str, str], args: argparse.Namespace) -> bool:
    if str(row.get("dataset", args.dataset)) != args.dataset:
        return False
    frame = safe_int(row.get("frame"), -1)
    return args.frame_min <= frame <= args.frame_max


def is_positive(row: dict[str, str]) -> bool:
    return safe_int(row.get("label_strict_8px")) == 1 or safe_int(row.get("label_loose_16px")) == 1


def leakage_required(row: dict[str, str]) -> int:
    """Rows may contain leakage labels, but the runtime source must not need them."""

    non_leak_values = [
        value
        for key, value in row.items()
        if key not in LEAKAGE_COLS and str(value).strip() not in {"", "nan", "None"}
    ]
    return int(len(non_leak_values) == 0)


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    counter: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        counter[tuple(row.get(key, "") for key in keys)] += 1
    return [{**{key: value for key, value in zip(keys, group)}, "count": count} for group, count in counter.most_common()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    external_rows = [r for r in read_csv(Path(args.external_candidates)) if in_focus(r, args)]
    external_positive = [r for r in external_rows if is_positive(r)]
    pi_rows = [r for r in read_csv(Path(args.pi_candidate_rows)) if in_focus(r, args)]
    prediction_rows = [r for r in read_csv(Path(args.prediction_rows)) if in_focus(r, args)]
    runtime_rows = [r for r in read_csv(Path(args.runtime_rows)) if in_focus(r, args)]

    pi_by_id = {str(r.get("candidate_id", "")): r for r in pi_rows}
    pred_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        pred_by_id[str(row.get("candidate_id", ""))].append(row)
    runtime_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in runtime_rows:
        runtime_by_id[str(row.get("candidate_id", ""))].append(row)

    matches: list[dict[str, Any]] = []
    strict_frames: set[int] = set()
    loose_frames: set[int] = set()
    strict_frames_pi: set[int] = set()
    loose_frames_pi: set[int] = set()
    strict_frames_runtime: set[int] = set()
    loose_frames_runtime: set[int] = set()
    leakage_missing = 0
    for row in external_positive:
        candidate_id = str(row.get("candidate_id", ""))
        frame = safe_int(row.get("frame"))
        strict = safe_int(row.get("label_strict_8px"))
        loose = safe_int(row.get("label_loose_16px"))
        if strict:
            strict_frames.add(frame)
        if loose:
            loose_frames.add(frame)
        pi = pi_by_id.get(candidate_id)
        runtime = runtime_by_id.get(candidate_id, [])
        pred = pred_by_id.get(candidate_id, [])
        has_pi = int(pi is not None)
        has_runtime = int(bool(runtime))
        if has_pi and strict:
            strict_frames_pi.add(frame)
        if has_pi and loose:
            loose_frames_pi.add(frame)
        if has_runtime and strict:
            strict_frames_runtime.add(frame)
        if has_runtime and loose:
            loose_frames_runtime.add(frame)
        leak = leakage_required(pi or {})
        leakage_missing += int(leak)
        matches.append(
            {
                "dataset": row.get("dataset", args.dataset),
                "clip": row.get("clip", ""),
                "frame": frame,
                "candidate_id": candidate_id,
                "rank": row.get("rank", ""),
                "source_family": row.get("source_family", ""),
                "source": row.get("source", ""),
                "x": row.get("x", ""),
                "y": row.get("y", ""),
                "w": row.get("w", ""),
                "h": row.get("h", ""),
                "label_strict_8px": strict,
                "label_loose_16px": loose,
                "in_pi_candidate_rows": has_pi,
                "in_prediction_rows": int(bool(pred)),
                "in_runtime_rows": has_runtime,
                "runtime_row_count": len(runtime),
                "pi_source_family": "" if pi is None else pi.get("source_family", ""),
                "pi_source": "" if pi is None else pi.get("source", ""),
                "pi_rank": "" if pi is None else pi.get("rank", ""),
                "pi_computable_nonleak_features_available": int(not leak),
                "best_prediction_model_score": "" if not pred else max(safe_float(p.get("model_score")) for p in pred),
            }
        )

    total = len(external_positive)
    matched = sum(int(r["in_pi_candidate_rows"]) for r in matches)
    runtime_matched = sum(int(r["in_runtime_rows"]) for r in matches)
    matched_rate = matched / max(1, total)
    nonleak_rate = (total - leakage_missing) / max(1, total)
    summary = {
        "artifact": "srps_id58_source_promoter_provenance_v1",
        "external_positive_rows": total,
        "matched_pi_positive_rows": matched,
        "matched_pi_positive_rate": round(matched_rate, 6),
        "matched_runtime_positive_rows": runtime_matched,
        "matched_runtime_positive_rate": round(runtime_matched / max(1, total), 6),
        "external_strict_positive_frames": len(strict_frames),
        "external_loose_positive_frames": len(loose_frames),
        "matched_pi_strict_positive_frames": len(strict_frames_pi),
        "matched_pi_loose_positive_frames": len(loose_frames_pi),
        "matched_runtime_strict_positive_frames": len(strict_frames_runtime),
        "matched_runtime_loose_positive_frames": len(loose_frames_runtime),
        "nonleak_reproducible_positive_rows": total - leakage_missing,
        "nonleak_reproducible_rate": round(nonleak_rate, 6),
        "phase_a_gate_pass": int(matched_rate >= 0.9 and nonleak_rate >= 0.9),
    }

    missing_runtime = [r for r in matches if int(r["in_runtime_rows"]) == 0]
    write_csv(out_dir / "provenance_matches.csv", matches)
    write_csv(out_dir / "positive_source_summary.csv", summarize_by(matches, ["source_family", "source", "in_pi_candidate_rows", "in_runtime_rows"]))
    write_csv(out_dir / "missing_runtime_source_summary.csv", summarize_by(missing_runtime, ["source_family", "source", "in_pi_candidate_rows"]))
    write_json(out_dir / "summary.json", summary)
    readme = f"""# SRPS-ID58 Source-Promoter Provenance

External positive rows traced to Pi-computable candidate rows.

- phase A gate pass: `{summary['phase_a_gate_pass']}`
- external positive rows: `{total}`
- matched Pi rows: `{matched}` ({summary['matched_pi_positive_rate']})
- matched runtime rows: `{runtime_matched}` ({summary['matched_runtime_positive_rate']})
- Pi strict/loose frame availability from positives: `{len(strict_frames_pi)}` / `{len(loose_frames_pi)}`

Files:

- `provenance_matches.csv`
- `positive_source_summary.csv`
- `missing_runtime_source_summary.csv`
- `summary.json`
"""
    (out_dir / "README.md").write_text(readme)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
