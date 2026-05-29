#!/usr/bin/env python3
"""Evaluate whether partial labels from one clip improve held-out frames.

This answers a practical active-learning question:

    If we label more frames from a hard clip, does the ranker learn a reusable
    surface/background distinction inside that clip, or does it only memorize
    individual frames?

The script trains on all other clips plus a subset of the target clip, then
tests on the remaining target-clip frames. It uses the same candidate examples
and feature vectorization as ``train_surface_xy_ranker.py`` so the result is
comparable to the normal LOCO ranker reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import train_surface_xy_ranker as ranker
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import train_surface_xy_ranker as ranker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--center_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--negative_min_dist_px", type=float, default=24.0)
    p.add_argument("--confidence", nargs="*", default=["high", "medium_high"])
    p.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "hist_gbdt", "extra_trees"),
        default=["logistic", "hist_gbdt", "extra_trees"],
    )
    p.add_argument("--random_state", type=int, default=17)
    return p.parse_args()


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


def frame_no(row: dict[str, str]) -> int:
    return ranker.int_or_default(row.get("frame"), 0)


def examples_for_labels(
    labels: list[dict[str, str]],
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    center_tol_px: float,
    negative_min_dist_px: float,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for lab in labels:
        rows = top_by_clip.get(lab["clip"], {}).get(frame_no(lab), [])
        for row in rows:
            dist_px = ranker.dist_to_label(row, lab)
            if dist_px <= center_tol_px:
                y = 1
            elif dist_px >= negative_min_dist_px:
                y = 0
            else:
                continue
            examples.append(
                {
                    "clip": lab["clip"],
                    "frame": frame_no(lab),
                    "row": row,
                    "y": y,
                    "dist_px": dist_px,
                }
            )
    return examples


def evaluate_model(
    model: Any,
    test_labels: list[dict[str, str]],
    top_by_clip: dict[str, dict[int, list[dict[str, str]]]],
    numeric: list[str],
    sources: list[str],
    split_name: str,
    model_name: str,
    center_tol_px: float,
    loose_tol_px: float,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for lab in test_labels:
        rows = top_by_clip.get(lab["clip"], {}).get(frame_no(lab), [])
        if not rows:
            continue
        scores = ranker.predict_score(model, ranker.vectorize(rows, numeric, sources))
        best_i = int(np.argmax(scores))
        best = rows[best_i]
        dist_px = ranker.dist_to_label(best, lab)
        oracle_dist = min((ranker.dist_to_label(row, lab) for row in rows), default=float("inf"))
        rows_out.append(
            {
                "split": split_name,
                "model": model_name,
                "clip": lab["clip"],
                "frame": frame_no(lab),
                "rank": best.get("rank", ""),
                "score": round(float(scores[best_i]), 6),
                "source": best.get("cand_source", ""),
                "dist_px": round(dist_px, 3),
                "strict_hit": dist_px <= center_tol_px,
                "loose_hit": dist_px <= loose_tol_px,
                "oracle_hit": oracle_dist <= center_tol_px,
            }
        )
    return rows_out


def summarize(rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    return {
        **extra,
        "test_frames": n,
        "oracle_recall": round(sum(bool(r["oracle_hit"]) for r in rows) / max(1, n), 4),
        "strict_recall": round(sum(bool(r["strict_hit"]) for r in rows) / max(1, n), 4),
        "loose_recall": round(sum(bool(r["loose_hit"]) for r in rows) / max(1, n), 4),
    }


def make_splits(target_labels: list[dict[str, str]]) -> dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]]:
    ordered = sorted(target_labels, key=frame_no)
    if not ordered:
        return {}
    min_f = frame_no(ordered[0])
    max_f = frame_no(ordered[-1])
    midpoint = 0.5 * (min_f + max_f)
    early = [r for r in ordered if frame_no(r) <= midpoint]
    late = [r for r in ordered if frame_no(r) > midpoint]
    return {
        "train_early_test_late": (early, late),
        "train_late_test_early": (late, early),
        "train_even_test_odd": (ordered[::2], ordered[1::2]),
        "train_odd_test_even": (ordered[1::2], ordered[::2]),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [row for row in ranker.read_csv(Path(args.labels)) if ranker.label_visible(row)]
    if args.confidence:
        allowed = set(args.confidence)
        labels = [row for row in labels if row.get("confidence") in allowed]
    labels.sort(key=lambda row: (row.get("clip", ""), frame_no(row)))

    clips = sorted({row["clip"] for row in labels})
    if args.clip not in clips:
        raise SystemExit(f"clip {args.clip!r} not present in labels")

    top_by_clip = {
        clip: ranker.load_top_tubes(Path(args.results_dir), clip, args.max_rank)
        for clip in clips
    }
    all_rows: list[dict[str, str]] = []
    for by_frame in top_by_clip.values():
        for rows in by_frame.values():
            all_rows.extend(rows)
    numeric, sources = ranker.infer_features(all_rows)

    target_labels = [row for row in labels if row["clip"] == args.clip]
    other_labels = [row for row in labels if row["clip"] != args.clip]
    splits = make_splits(target_labels)
    models = ranker.make_models(args.random_state)

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for split_name, (target_train, target_test) in splits.items():
        if not target_train or not target_test:
            continue
        train_labels = other_labels + target_train
        examples = examples_for_labels(
            train_labels,
            top_by_clip,
            args.center_tol_px,
            args.negative_min_dist_px,
        )
        if not examples:
            continue
        x = ranker.vectorize([ex["row"] for ex in examples], numeric, sources)
        y = np.asarray([int(ex["y"]) for ex in examples], dtype=np.int32)
        if len(set(y.tolist())) < 2:
            continue
        for model_name in args.models:
            model = models[model_name]
            model.fit(x, y)
            rows = evaluate_model(
                model,
                target_test,
                top_by_clip,
                numeric,
                sources,
                split_name,
                model_name,
                args.center_tol_px,
                args.loose_tol_px,
            )
            prediction_rows.extend(rows)
            summary_rows.append(
                summarize(
                    rows,
                    {
                        "split": split_name,
                        "model": model_name,
                        "train_target_frames": len(target_train),
                        "test_target_frames": len(target_test),
                        "train_examples": len(examples),
                        "positive_examples": int(np.sum(y == 1)),
                        "negative_examples": int(np.sum(y == 0)),
                    },
                )
            )

    write_csv(out_dir / "partial_clip_summary.csv", summary_rows)
    write_csv(out_dir / "partial_clip_predictions.csv", prediction_rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "labels": str(args.labels),
                "results_dir": str(args.results_dir),
                "clip": args.clip,
                "max_rank": args.max_rank,
                "confidence": args.confidence,
                "numeric_features": numeric,
                "source_features": sources,
            },
            indent=2,
        )
        + "\n"
    )
    print(out_dir / "partial_clip_summary.csv")


if __name__ == "__main__":
    main()
