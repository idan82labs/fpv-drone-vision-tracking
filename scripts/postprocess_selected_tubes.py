#!/usr/bin/env python3
"""Snap and fill long selected tube tracks from top_tubes.csv.

The detector can keep the right tube alive while selecting a weaker branch
inside that same tube on some frames. This post-processor promotes the best
candidate within each long selected tube id and fills short no-selection gaps
where that tube already has candidates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--min_selected_frames", type=int, default=50)
    p.add_argument("--min_mean_selected_score", type=float, default=30.0)
    p.add_argument("--min_fill_score", type=float, default=5.0)
    p.add_argument("--merge_gap", type=int, default=60)
    p.add_argument("--threshold", type=float, default=5.0)
    return p.parse_args()


def merged_spans(frames: list[int], merge_gap: int) -> list[tuple[int, int, int]]:
    if not frames:
        return []
    spans: list[tuple[int, int, int]] = []
    start = prev = frames[0]
    count = 1
    for frame in frames[1:]:
        if frame - prev > merge_gap:
            spans.append((start, prev, count))
            start = frame
            count = 0
        prev = frame
        count += 1
    spans.append((start, prev, count))
    return spans


def main() -> None:
    args = parse_args()
    top = pd.read_csv(args.top_tubes)
    selected = top[top["selected"].eq(1)].copy()
    chosen: dict[int, pd.Series] = {}
    accepted: list[dict[str, object]] = []

    for track_id, group in selected.groupby("track_id"):
        frames = sorted(int(f) for f in group["frame"].unique())
        for start, end, selected_count in merged_spans(frames, args.merge_gap):
            selected_span = group[group["frame"].between(start, end)]
            if selected_count < args.min_selected_frames:
                continue
            mean_score = float(selected_span["verified_score"].mean())
            if mean_score < args.min_mean_selected_score:
                continue
            fill = top[
                top["track_id"].eq(track_id)
                & top["frame"].between(start, end)
                & top["verified_score"].ge(args.min_fill_score)
            ].copy()
            if fill.empty:
                continue
            fill = fill.sort_values(["frame", "verified_score", "rank"], ascending=[True, False, True])
            for frame, candidates in fill.groupby("frame", sort=True):
                best = candidates.iloc[0]
                existing = chosen.get(int(frame))
                if existing is None or float(best["verified_score"]) > float(existing["verified_score"]):
                    chosen[int(frame)] = best
            accepted.append(
                {
                    "track_id": int(track_id),
                    "start": start,
                    "end": end,
                    "selected_frames": selected_count,
                    "filled_frames": int(fill["frame"].nunique()),
                    "mean_selected_score": round(mean_score, 3),
                }
            )

    rows: list[dict[str, object]] = []
    for frame in sorted(int(f) for f in top["frame"].unique()):
        row = chosen.get(frame)
        if row is None:
            rows.append(
                {
                    "clip": args.clip,
                    "frame": frame,
                    "selected": 0,
                    "rank": "",
                    "learned_score": "",
                    "threshold": args.threshold,
                    "human_label": "",
                    "x": "",
                    "y": "",
                    "w": "",
                    "h": "",
                    "verified_score": "",
                    "source": "",
                    "track_id": "",
                }
            )
        else:
            rows.append(
                {
                    "clip": args.clip,
                    "frame": frame,
                    "selected": 1,
                    "rank": int(row["rank"]),
                    "learned_score": round(float(row["verified_score"]), 6),
                    "threshold": args.threshold,
                    "human_label": "",
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "w": int(row["w"]),
                    "h": int(row["h"]),
                    "verified_score": round(float(row["verified_score"]), 6),
                    "source": str(row["cand_source"]),
                    "track_id": int(row["track_id"]),
                }
            )

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out_path)
    print(pd.DataFrame(accepted).to_string(index=False))
    print(f"selected_frames={int(out['selected'].sum())}")


if __name__ == "__main__":
    main()
