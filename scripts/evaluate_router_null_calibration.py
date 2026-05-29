#!/usr/bin/env python3
"""Evaluate router-specific max-null thresholds on OOF candidate scores.

This tests the professor's calibration recommendation without retraining a
ranker. Given out-of-fold candidate scores, choose per-router thresholds from
the *training clips'* invisible/no-target frames, then apply those thresholds to
the held-out clip.

The policy is intentionally simple:

- group candidate scores by candidate-local router bucket;
- for each train null frame, record the maximum score per bucket;
- threshold each bucket at a requested null quantile;
- in the held-out clip, emit the highest-scoring candidate whose own bucket
  score clears that bucket's threshold.

If this cannot beat a global threshold, router-specific max-null calibration is
not the missing piece for the current feature set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--candidate_scores", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--score_column", default="score")
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--quantiles", default="0.80,0.90,0.95,0.98,0.99")
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--min_bucket_null_samples", type=int, default=3)
    p.add_argument("--min_threshold", type=float, default=0.0)
    p.add_argument("--threshold_margin", type=float, default=0.0)
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


def fnum(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def visible(row: dict[str, str]) -> bool:
    raw = str(row.get("visible", "")).strip().lower()
    return raw in {"1", "true", "yes", "visible", "target"}


def load_labels(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(path):
        clip = row.get("clip", "")
        frame = int(fnum(row.get("frame"), -1))
        if not clip or frame < 0:
            continue
        is_visible = visible(row)
        bbox = None
        if is_visible:
            bbox = (
                fnum(row.get("det_x", row.get("x"))),
                fnum(row.get("det_y", row.get("y"))),
                max(1.0, fnum(row.get("det_w", row.get("w")), 1.0)),
                max(1.0, fnum(row.get("det_h", row.get("h")), 1.0)),
            )
        out[(clip, frame)] = {"visible": is_visible and bbox is not None, "bbox": bbox, "row": row}
    return out


def row_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        fnum(row.get("x")),
        fnum(row.get("y")),
        max(1.0, fnum(row.get("w"), 1.0)),
        max(1.0, fnum(row.get("h"), 1.0)),
    )


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return float(math.hypot(ax - bx, ay - by))


def parse_quantiles(raw: str) -> list[float]:
    vals = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise SystemExit("no quantiles supplied")
    for q in vals:
        if q < 0.0 or q > 1.0:
            raise SystemExit(f"quantile out of range: {q}")
    return vals


def bucket_for_row(row: dict[str, str]) -> str:
    state = str(row.get("cand_router_state", "")).strip()
    if state == "surface_backed":
        return "surface"
    if state in {"boundary_mixed", "sky_target_near_surface"}:
        return "boundary"
    if state == "line_attached":
        return "line"
    if state == "clean_sky":
        return "clean_sky"
    rates = {
        "surface": fnum(row.get("tube_router_surface_backed_rate")),
        "clean_sky": fnum(row.get("tube_router_clean_sky_rate")),
        "boundary": fnum(row.get("tube_router_boundary_rate")),
        "line": fnum(row.get("tube_router_line_attached_rate")),
    }
    best, val = max(rates.items(), key=lambda item: item[1])
    if val >= 0.35:
        return best
    return "unknown"


def load_candidates(
    path: Path,
    label_keys: set[tuple[str, int]],
    score_column: str,
    models: list[str],
) -> dict[str, dict[tuple[str, int], list[dict[str, Any]]]]:
    rows = read_csv(path)
    available_models = sorted({row.get("model", "") for row in rows if row.get("model", "")})
    selected_models = models or available_models
    out: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = {
        model: defaultdict(list) for model in selected_models
    }
    for row in rows:
        model = row.get("model", "")
        if model not in out:
            continue
        clip = row.get("clip", "")
        frame = int(fnum(row.get("frame"), -1))
        key = (clip, frame)
        if key not in label_keys:
            continue
        score = fnum(row.get(score_column), -math.inf)
        if not math.isfinite(score):
            continue
        rec: dict[str, Any] = dict(row)
        rec["_score"] = score
        rec["_bucket"] = bucket_for_row(row)
        out[model][key].append(rec)
    return out


def max_null_scores_by_bucket(
    labels: dict[tuple[str, int], dict[str, Any]],
    cands: dict[tuple[str, int], list[dict[str, Any]]],
    train_clips: set[str],
) -> dict[str, list[float]]:
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for (clip, frame), lab in labels.items():
        if clip not in train_clips or lab["visible"]:
            continue
        rows = cands.get((clip, frame), [])
        if not rows:
            continue
        frame_best: dict[str, float] = defaultdict(lambda: -math.inf)
        for row in rows:
            bucket = str(row["_bucket"])
            frame_best[bucket] = max(frame_best[bucket], float(row["_score"]))
            frame_best["global"] = max(frame_best["global"], float(row["_score"]))
        for bucket, score in frame_best.items():
            if math.isfinite(score):
                by_bucket[bucket].append(score)
    return by_bucket


def thresholds_from_nulls(
    nulls: dict[str, list[float]],
    quantile: float,
    min_bucket_null_samples: int,
    min_threshold: float,
    margin: float,
) -> dict[str, float]:
    global_vals = nulls.get("global", [])
    global_thr = float(np.quantile(global_vals, quantile)) if global_vals else 1.0
    thresholds = {"global": max(min_threshold, global_thr + margin)}
    for bucket, vals in nulls.items():
        if bucket == "global":
            continue
        if len(vals) < min_bucket_null_samples:
            thresholds[bucket] = max(min_threshold, global_thr + margin)
        else:
            thresholds[bucket] = max(min_threshold, float(np.quantile(vals, quantile)) + margin)
    return thresholds


def threshold_for(row: dict[str, Any], thresholds: dict[str, float]) -> float:
    return thresholds.get(str(row["_bucket"]), thresholds.get("global", 1.0))


def evaluate_clip(
    labels: dict[tuple[str, int], dict[str, Any]],
    cands: dict[tuple[str, int], list[dict[str, Any]]],
    clip: str,
    thresholds: dict[str, float],
    strict_tol: float,
    loose_tol: float,
    model: str,
    quantile: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, lab in sorted(labels.items(), key=lambda item: item[0][1]):
        if key[0] != clip:
            continue
        rows = cands.get(key, [])
        passing = [row for row in rows if float(row["_score"]) >= threshold_for(row, thresholds)]
        chosen = max(passing, key=lambda row: float(row["_score"])) if passing else None
        selected = chosen is not None
        strict = False
        loose = False
        dist = ""
        if lab["visible"] and chosen is not None and lab["bbox"] is not None:
            d = center_dist(row_bbox(chosen), lab["bbox"])
            dist = round(d, 3)
            strict = d <= strict_tol
            loose = d <= loose_tol
        out.append(
            {
                "clip": key[0],
                "frame": key[1],
                "model": model,
                "quantile": quantile,
                "visible": int(bool(lab["visible"])),
                "selected": int(selected),
                "strict_hit": int(strict),
                "loose_hit": int(loose),
                "dist_px": dist,
                "score": "" if chosen is None else round(float(chosen["_score"]), 9),
                "bucket": "" if chosen is None else chosen["_bucket"],
                "threshold": "" if chosen is None else round(threshold_for(chosen, thresholds), 9),
                "rank": "" if chosen is None else chosen.get("rank", ""),
                "x": "" if chosen is None else chosen.get("x", ""),
                "y": "" if chosen is None else chosen.get("y", ""),
                "w": "" if chosen is None else chosen.get("w", ""),
                "h": "" if chosen is None else chosen.get("h", ""),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], model: str, quantile: float, clip: str = "ALL") -> dict[str, Any]:
    visible_rows = [row for row in rows if int(row["visible"])]
    invisible_rows = [row for row in rows if not int(row["visible"])]
    return {
        "clip": clip,
        "model": model,
        "quantile": quantile,
        "frames": len(rows),
        "visible_frames": len(visible_rows),
        "strict_hits": sum(int(row["strict_hit"]) for row in visible_rows),
        "strict_recall": round(sum(int(row["strict_hit"]) for row in visible_rows) / max(1, len(visible_rows)), 4),
        "loose_hits": sum(int(row["loose_hit"]) for row in visible_rows),
        "loose_recall": round(sum(int(row["loose_hit"]) for row in visible_rows) / max(1, len(visible_rows)), 4),
        "invisible_frames": len(invisible_rows),
        "invisible_no_box": sum(1 for row in invisible_rows if not int(row["selected"])),
        "invisible_no_box_rate": round(
            sum(1 for row in invisible_rows if not int(row["selected"])) / max(1, len(invisible_rows)),
            4,
        ),
        "selected_frames_total": sum(int(row["selected"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels))
    clips = sorted({clip for clip, _frame in labels})
    candidates = load_candidates(
        Path(args.candidate_scores),
        set(labels),
        args.score_column,
        args.models,
    )
    quantiles = parse_quantiles(args.quantiles)

    frame_rows: list[dict[str, Any]] = []
    by_clip_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for model, cands in candidates.items():
        if not cands:
            continue
        for q in quantiles:
            all_model_rows: list[dict[str, Any]] = []
            for heldout in clips:
                train_clips = set(clips) - {heldout}
                nulls = max_null_scores_by_bucket(labels, cands, train_clips)
                thresholds = thresholds_from_nulls(
                    nulls,
                    q,
                    args.min_bucket_null_samples,
                    args.min_threshold,
                    args.threshold_margin,
                )
                threshold_rows.extend(
                    {
                        "heldout_clip": heldout,
                        "model": model,
                        "quantile": q,
                        "bucket": bucket,
                        "threshold": round(thr, 9),
                        "null_samples": len(nulls.get(bucket, [])),
                    }
                    for bucket, thr in sorted(thresholds.items())
                )
                rows = evaluate_clip(
                    labels,
                    cands,
                    heldout,
                    thresholds,
                    args.strict_tol_px,
                    args.loose_tol_px,
                    model,
                    q,
                )
                all_model_rows.extend(rows)
                by_clip_rows.append(summarize(rows, model, q, heldout))
            frame_rows.extend(all_model_rows)
            by_clip_rows.append(summarize(all_model_rows, model, q))

    summary_rows = [row for row in by_clip_rows if row["clip"] == "ALL"]
    summary_rows.sort(key=lambda r: (r["model"], r["quantile"]))
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "by_clip_summary.csv", by_clip_rows)
    write_csv(out_dir / "frame_eval.csv", frame_rows)
    write_csv(out_dir / "thresholds.csv", threshold_rows)
    metadata = {
        "labels": args.labels,
        "candidate_scores": args.candidate_scores,
        "score_column": args.score_column,
        "models": list(candidates),
        "quantiles": quantiles,
        "min_threshold": args.min_threshold,
        "clips": clips,
        "label_frames": len(labels),
        "caveat": "Uses existing out-of-fold candidate scores; tests calibration only, not feature learning.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "README.md").write_text(
        "# Router Null Calibration\n\n"
        "Leave-one-clip-out router-specific max-null threshold calibration over existing OOF candidate scores.\n"
    )
    print(out_dir / "summary.csv")


if __name__ == "__main__":
    main()
