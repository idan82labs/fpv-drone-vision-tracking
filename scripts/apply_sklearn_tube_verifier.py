#!/usr/bin/env python3
"""Apply a trained sklearn tube verifier to exported top_tubes.csv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import train_tube_verifier_sklearn as train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tube_labels", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--max_rank", type=int, default=None)
    return p.parse_args()


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


def row_vector(row: dict[str, str], numeric: list[str], sources: list[str]) -> np.ndarray:
    vals = []
    for name in numeric:
        val = train.safe_float(row.get(name))
        vals.append(np.nan if val is None else val)
    source = row.get("cand_source", "")
    vals.extend(1.0 if name == f"src_{source}" else 0.0 for name in sources)
    return np.asarray([vals], dtype=np.float64)


def rows_matrix(rows: list[dict[str, str]], numeric: list[str], sources: list[str]) -> np.ndarray:
    vals_all: list[list[float]] = []
    for row in rows:
        vals = []
        for name in numeric:
            val = train.safe_float(row.get(name))
            vals.append(np.nan if val is None else val)
        source = row.get("cand_source", "")
        vals.extend(1.0 if name == f"src_{source}" else 0.0 for name in sources)
        vals_all.append(vals)
    return np.asarray(vals_all, dtype=np.float64)


def load_label_lookup(path: Path | None) -> dict[tuple[str, int, int], str]:
    if path is None:
        return {}
    lookup: dict[tuple[str, int, int], str] = {}
    for row in read_csv(path):
        clip = row.get("clip", "")
        frame = int(float(row.get("frame", "0") or 0))
        rank = int(float(row.get("rank", "999") or 999))
        lookup[(clip, frame, rank)] = row.get("human_label", "")
    return lookup


def main() -> None:
    args = parse_args()
    model_obj = joblib.load(args.model)
    model = model_obj["model"]
    threshold = float(args.threshold if args.threshold is not None else model_obj["threshold"])
    max_rank = int(args.max_rank if args.max_rank is not None else model_obj.get("max_rank", 16))
    numeric = list(model_obj["numeric_features"])
    sources = list(model_obj["source_features"])
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_lookup = load_label_lookup(Path(args.tube_labels) if args.tube_labels else None)

    selections: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/top_tubes.csv")):
        clip = path.parent.name
        by_frame: dict[int, list[dict[str, str]]] = {}
        for row in read_csv(path):
            rank = int(float(row.get("rank", "999") or 999))
            if rank > max_rank:
                continue
            frame = int(float(row.get("frame", "0") or 0))
            by_frame.setdefault(frame, []).append(row)

        clip_rows = [row for frame in sorted(by_frame) for row in by_frame[frame]]
        clip_scores = train.predict_score(model, rows_matrix(clip_rows, numeric, sources))
        score_by_id = {id(row): float(score) for row, score in zip(clip_rows, clip_scores)}

        for frame, rows in sorted(by_frame.items()):
            scored: list[tuple[float, dict[str, str]]] = []
            for row in rows:
                score = score_by_id[id(row)]
                rank = int(float(row.get("rank", "999") or 999))
                scored.append((score, row))
                scored_rows.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "rank": rank,
                        "learned_score": round(score, 6),
                        "threshold": round(threshold, 6),
                        "human_label": label_lookup.get((clip, frame, rank), ""),
                        "x": row.get("x", ""),
                        "y": row.get("y", ""),
                        "w": row.get("w", ""),
                        "h": row.get("h", ""),
                        "verified_score": row.get("verified_score", ""),
                        "source": row.get("cand_source", ""),
                    }
                )
            if not scored:
                continue
            best_score, best = max(scored, key=lambda item: item[0])
            selected = best_score >= threshold
            rank = int(float(best.get("rank", "999") or 999))
            selections.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "selected": int(selected),
                    "rank": rank if selected else "",
                    "learned_score": round(best_score, 6),
                    "threshold": round(threshold, 6),
                    "human_label": label_lookup.get((clip, frame, rank), "") if selected else "",
                    "x": best.get("x", "") if selected else "",
                    "y": best.get("y", "") if selected else "",
                    "w": best.get("w", "") if selected else "",
                    "h": best.get("h", "") if selected else "",
                    "verified_score": best.get("verified_score", "") if selected else "",
                    "source": best.get("cand_source", "") if selected else "",
                }
            )

    write_csv(out_dir / "learned_tube_scores.csv", scored_rows)
    write_csv(out_dir / "learned_selections.csv", selections)
    print(out_dir / "learned_selections.csv")


if __name__ == "__main__":
    main()
