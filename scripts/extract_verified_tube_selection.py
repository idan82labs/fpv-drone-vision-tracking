#!/usr/bin/env python3
"""Extract a demo selection from verified target tube ids.

This is a post-run selector for analysis/demo work. It does not create new
proposals; it chooses the best candidate already present inside hand-reviewed
target tube ids from a top_tubes.csv export.
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
    p.add_argument(
        "--segment",
        action="append",
        required=True,
        help="Segment as start:end:track_id, frame numbers inclusive.",
    )
    p.add_argument("--min_score", type=float, default=None)
    p.add_argument("--threshold", type=float, default=0.0)
    return p.parse_args()


def parse_segment(raw: str) -> tuple[int, int, int]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"bad segment {raw!r}; expected start:end:track_id")
    start, end, track_id = (int(x) for x in parts)
    if end < start:
        raise argparse.ArgumentTypeError(f"bad segment {raw!r}; end before start")
    return start, end, track_id


def main() -> None:
    args = parse_args()
    top = pd.read_csv(args.top_tubes)
    frames = sorted(int(f) for f in top["frame"].unique())
    segments = [parse_segment(s) for s in args.segment]

    chosen: dict[int, pd.Series] = {}
    for start, end, track_id in segments:
        seg = top[
            top["frame"].between(start, end)
            & top["track_id"].eq(track_id)
        ].copy()
        if args.min_score is not None:
            seg = seg[seg["verified_score"].ge(args.min_score)]
        if seg.empty:
            continue
        seg = seg.sort_values(["frame", "verified_score", "rank"], ascending=[True, False, True])
        for frame, group in seg.groupby("frame", sort=True):
            chosen[int(frame)] = group.iloc[0]

    rows: list[dict[str, object]] = []
    for frame in frames:
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
    print(f"selected_frames={int(out['selected'].sum())}")


if __name__ == "__main__":
    main()
