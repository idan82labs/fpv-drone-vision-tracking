#!/usr/bin/env python3
"""Evaluate leave-one-clip-out candidate re-ranking over exported top tubes.

This is a selector/ranker probe, not a runtime path. It labels top-tube
candidates from dense frame labels, trains on all but one clip, then applies the
learned candidate score to the held-out clip. A candidate is emitted only when
the best score clears a threshold; otherwise the frame is no-box.

The goal is to separate two failure modes that branch routing cannot fix:

- visible frames where the target is in top alternatives but the selected branch
  suppresses or picks clutter;
- no-target frames where both branch families still emit a plausible speck/edge.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone

try:
    import train_full_video_state_ranker as ranker
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import train_full_video_state_ranker as ranker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True, help="Directory containing <clip>/top_tubes.csv files.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=20)
    p.add_argument("--positive_tol_px", type=float, default=8.0)
    p.add_argument("--negative_min_dist_px", type=float, default=16.0)
    p.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "hist_gbdt", "extra_trees"),
        default=("logistic", "hist_gbdt", "extra_trees"),
    )
    p.add_argument("--random_state", type=int, default=23)
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


def clips_in_labels(path: Path) -> list[str]:
    clips = sorted({row.get("clip", "") for row in read_csv(path) if row.get("clip", "")})
    if not clips:
        raise SystemExit("labels do not contain clip ids")
    return clips


def load_clip_examples(
    labels_path: Path,
    results_dir: Path,
    clip: str,
    max_rank: int,
    positive_tol_px: float,
    negative_min_dist_px: float,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], int]:
    labels = ranker.load_labels(labels_path, clip)
    top_path = results_dir / clip / "top_tubes.csv"
    if not top_path.exists():
        raise FileNotFoundError(top_path)
    top_by_frame = ranker.load_top_tubes(top_path, max_rank)
    examples, ignored_near = ranker.make_examples(labels, top_by_frame, positive_tol_px, negative_min_dist_px)
    return labels, examples, ignored_near


def evaluate_selected_rows(
    labels: dict[int, dict[str, Any]],
    selected_by_frame: dict[int, dict[str, Any]],
    threshold: float,
    model: str,
    clip: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for frame, lab in sorted(labels.items()):
        selected = selected_by_frame.get(frame)
        visible = bool(lab["visible"])
        selected_box = None
        if selected is not None:
            selected_box = ranker.row_bbox(selected)
        label_box = lab["bbox"] if visible else None
        dist = None
        strict = False
        loose = False
        if visible and label_box is not None and selected_box is not None:
            dist = ranker.center_dist(selected_box, label_box)
            strict = dist <= 8.0
            loose = dist <= 16.0
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "model": model,
                "threshold": threshold,
                "visible": int(visible),
                "selected": int(selected is not None),
                "strict_hit": int(strict),
                "loose_hit": int(loose),
                "dist_px": "" if dist is None else round(float(dist), 3),
                "score": "" if selected is None else round(float(selected.get("score", 0.0)), 9),
                "rank": "" if selected is None else selected.get("rank", ""),
                "x": "" if selected is None else selected.get("x", ""),
                "y": "" if selected is None else selected.get("y", ""),
                "w": "" if selected is None else selected.get("w", ""),
                "h": "" if selected is None else selected.get("h", ""),
            }
        )
    summary = aggregate(rows, model, threshold, clip=clip)
    return summary, rows


def aggregate(rows: list[dict[str, Any]], model: str, threshold: float, clip: str = "ALL") -> dict[str, Any]:
    visible = sum(int(row["visible"]) for row in rows)
    strict = sum(int(row["strict_hit"]) for row in rows)
    loose = sum(int(row["loose_hit"]) for row in rows)
    invisible = sum(1 for row in rows if not int(row["visible"]))
    invisible_no_box = sum(1 for row in rows if not int(row["visible"]) and not int(row["selected"]))
    return {
        "clip": clip,
        "model": model,
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


def oracle_summary(labels: dict[int, dict[str, Any]], examples: list[dict[str, Any]], clip: str) -> dict[str, Any]:
    positive_frames = {int(ex["frame"]) for ex in examples if int(ex["y"]) == 1}
    visible_frames = [frame for frame, lab in labels.items() if lab["visible"]]
    invisible_frames = [frame for frame, lab in labels.items() if not lab["visible"]]
    candidate_frames = {int(ex["frame"]) for ex in examples}
    return {
        "clip": clip,
        "visible_frames": len(visible_frames),
        "invisible_frames": len(invisible_frames),
        "visible_oracle_strict_frames": len(positive_frames),
        "visible_oracle_strict_rate": round(len(positive_frames) / max(1, len(visible_frames)), 4),
        "candidate_label_frames": len(candidate_frames),
    }


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)

    clips = clips_in_labels(labels_path)
    labels_by_clip: dict[str, dict[int, dict[str, Any]]] = {}
    examples_by_clip: dict[str, list[dict[str, Any]]] = {}
    ignored_near_by_clip: dict[str, int] = {}
    all_examples: list[dict[str, Any]] = []
    for clip in clips:
        labels, examples, ignored_near = load_clip_examples(
            labels_path,
            results_dir,
            clip,
            args.max_rank,
            args.positive_tol_px,
            args.negative_min_dist_px,
        )
        labels_by_clip[clip] = labels
        examples_by_clip[clip] = examples
        ignored_near_by_clip[clip] = ignored_near
        all_examples.extend(examples)

    if not all_examples:
        raise SystemExit("no candidate examples")
    numeric, sources = ranker.infer_features([ex["row"] for ex in all_examples])
    models = ranker.make_models(args.random_state)

    oracle_rows = [oracle_summary(labels_by_clip[clip], examples_by_clip[clip], clip) for clip in clips]
    for row in oracle_rows:
        row["ignored_near_examples"] = ignored_near_by_clip[row["clip"]]
    write_csv(out_dir / "oracle_summary.csv", oracle_rows)

    candidate_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    by_clip_rows: list[dict[str, Any]] = []
    summary_buckets: dict[tuple[str, float], list[dict[str, Any]]] = {}

    for model_name in args.models:
        for holdout_clip in clips:
            train_examples = [ex for clip in clips if clip != holdout_clip for ex in examples_by_clip[clip]]
            holdout_examples = examples_by_clip[holdout_clip]
            if not train_examples or not holdout_examples:
                continue
            y_train = np.asarray([int(ex["y"]) for ex in train_examples], dtype=np.int32)
            if len(set(y_train.tolist())) < 2:
                continue
            model = clone(models[model_name])
            x_train = ranker.vectorize_rows([ex["row"] for ex in train_examples], numeric, sources)
            model.fit(x_train, y_train)
            x_holdout = ranker.vectorize_rows([ex["row"] for ex in holdout_examples], numeric, sources)
            scores = ranker.predict_score(model, x_holdout)

            scored_by_frame: dict[int, list[dict[str, Any]]] = {}
            for ex, score in zip(holdout_examples, scores):
                row = dict(ex["row"])
                frame = int(ex["frame"])
                row.update(
                    {
                        "clip": holdout_clip,
                        "frame": frame,
                        "model": model_name,
                        "score": round(float(score), 9),
                        "target_candidate": int(ex["y"]),
                        "dist_px": "" if ex["dist_px"] is None else round(float(ex["dist_px"]), 3),
                        "visible": int(ex["label"]["visible"]),
                        "reason": ex["reason"],
                    }
                )
                candidate_rows.append(row)
                scored_by_frame.setdefault(frame, []).append(row)

            best_by_frame = {
                frame: max(rows, key=lambda r: float(r["score"])) for frame, rows in scored_by_frame.items()
            }
            for threshold in thresholds:
                selected = {
                    frame: row for frame, row in best_by_frame.items() if float(row["score"]) >= threshold
                }
                clip_summary, clip_rows = evaluate_selected_rows(
                    labels_by_clip[holdout_clip],
                    selected,
                    threshold,
                    model_name,
                    holdout_clip,
                )
                by_clip_rows.append(clip_summary)
                frame_rows.extend(clip_rows)
                summary_buckets.setdefault((model_name, threshold), []).extend(clip_rows)

    summary_rows = [
        aggregate(rows, model_name, threshold) for (model_name, threshold), rows in sorted(summary_buckets.items())
    ]
    summary_rows.sort(
        key=lambda r: (
            r["model"],
            r["threshold"],
        )
    )

    write_csv(out_dir / "oof_candidate_scores.csv", candidate_rows)
    write_csv(out_dir / "frame_eval.csv", frame_rows)
    write_csv(out_dir / "by_clip_summary.csv", by_clip_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    metadata = {
        "labels": args.labels,
        "results_dir": args.results_dir,
        "max_rank": args.max_rank,
        "positive_tol_px": args.positive_tol_px,
        "negative_min_dist_px": args.negative_min_dist_px,
        "thresholds": thresholds,
        "models": list(args.models),
        "clips": clips,
        "examples": len(all_examples),
        "positive_examples": int(sum(int(ex["y"]) for ex in all_examples)),
        "negative_examples": int(sum(1 - int(ex["y"]) for ex in all_examples)),
        "numeric_features": numeric,
        "source_features": sources,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "README.md").write_text(
        "# Multiclip Candidate Ranker\n\n"
        "Leave-one-clip-out candidate re-ranking over exported top tubes. "
        "This is an offline selector probe for reselect/no-box behavior, not a "
        "runtime detector path.\n"
    )
    print(out_dir / "summary.csv")


if __name__ == "__main__":
    main()
