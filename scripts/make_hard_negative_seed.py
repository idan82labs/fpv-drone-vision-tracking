#!/usr/bin/env python3
"""Select high-score full-video frames for the next active-labeling round."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", required=True, help="learned_selections.csv from apply_sklearn_tube_verifier.py")
    p.add_argument("--existing_labels", required=True, help="previous tube_alternatives_to_label.csv")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--max_frames", type=int, default=36)
    p.add_argument("--per_clip", type=int, default=5)
    p.add_argument("--min_gap", type=int, default=25)
    p.add_argument("--max_rank", type=int, default=8)
    return p.parse_args()


def far_enough(frame: int, selected: list[int], min_gap: int) -> bool:
    return all(abs(frame - other) >= min_gap for other in selected)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit("no rows selected")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selections = pd.read_csv(args.selections)
    existing = pd.read_csv(args.existing_labels)
    existing_frames = set((str(r.clip), int(r.frame)) for r in existing.itertuples(index=False))

    selected = selections[
        selections["selected"].eq(1)
        & selections["rank"].notna()
        & selections["x"].notna()
        & (selections["rank"].astype(float) <= args.max_rank)
    ].copy()
    selected["rank"] = selected["rank"].astype(int)
    selected["frame"] = selected["frame"].astype(int)
    selected["learned_score"] = selected["learned_score"].astype(float)
    selected = selected[
        ~selected.apply(lambda r: (str(r["clip"]), int(r["frame"])) in existing_frames, axis=1)
    ]

    chosen: list[dict[str, Any]] = []
    for clip, group in selected.sort_values("learned_score", ascending=False).groupby("clip"):
        frames: list[int] = []
        kept = 0
        for row in group.itertuples(index=False):
            frame = int(row.frame)
            if not far_enough(frame, frames, args.min_gap):
                continue
            frames.append(frame)
            kept += 1
            bbox = [
                int(round(float(row.x))),
                int(round(float(row.y))),
                int(round(float(row.w))),
                int(round(float(row.h))),
            ]
            chosen.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "label": "active_mine_high_score",
                    "selected_bbox": str(bbox),
                    "notes": (
                        f"learned_score={float(row.learned_score):.3f}; "
                        f"selected_rank={int(row.rank)}; "
                        f"source={row.source}; "
                        "label whether this high-score candidate is target or clutter"
                    ),
                    "selected_rank": int(row.rank),
                    "learned_score": round(float(row.learned_score), 6),
                    "source": row.source,
                }
            )
            if kept >= args.per_clip:
                break

    chosen = sorted(chosen, key=lambda r: float(r["learned_score"]), reverse=True)[: args.max_frames]
    write_csv(Path(args.out_csv), chosen)
    print(args.out_csv)
    print(f"selected_frames={len(chosen)}")


if __name__ == "__main__":
    main()
