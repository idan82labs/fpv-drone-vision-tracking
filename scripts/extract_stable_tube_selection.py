#!/usr/bin/env python3
"""Extract the strongest stable selected tube from a top_tubes.csv export."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--min_run", type=int, default=30)
    p.add_argument("--max_jump", type=float, default=12.0)
    return p.parse_args()


def center(row: Any) -> tuple[float, float]:
    return float(row.x) + float(row.w) / 2.0, float(row.y) + float(row.h) / 2.0


def selected_runs(df: pd.DataFrame, max_jump: float) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for track_id, group in df[df["selected"].eq(1)].sort_values("frame").groupby("track_id"):
        current: list[Any] = []
        prev = None
        for row in group.itertuples(index=False):
            starts_new = False
            if prev is not None:
                frame_gap = int(row.frame) - int(prev.frame)
                cx, cy = center(row)
                px, py = center(prev)
                jump = math.hypot(cx - px, cy - py)
                starts_new = frame_gap != 1 or jump > max_jump
            if starts_new and current:
                runs.append({"track_id": track_id, "rows": current})
                current = []
            current.append(row)
            prev = row
        if current:
            runs.append({"track_id": track_id, "rows": current})
    for run in runs:
        rows = run["rows"]
        scores = [float(r.verified_score) for r in rows]
        run["start"] = int(rows[0].frame)
        run["end"] = int(rows[-1].frame)
        run["length"] = len(rows)
        run["mean_score"] = sum(scores) / max(1, len(scores))
        run["tube_score"] = run["length"] * run["mean_score"]
    return runs


def output_rows(df: pd.DataFrame, clip: str, run: dict[str, Any]) -> pd.DataFrame:
    frames = sorted(int(f) for f in df["frame"].unique())
    selected_by_frame = {int(r.frame): r for r in run["rows"]}
    rows: list[dict[str, Any]] = []
    for frame in frames:
        if frame in selected_by_frame:
            r = selected_by_frame[frame]
            rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "selected": 1,
                    "rank": int(r.rank),
                    "learned_score": round(float(r.verified_score), 6),
                    "threshold": 0.0,
                    "human_label": "",
                    "x": int(r.x),
                    "y": int(r.y),
                    "w": int(r.w),
                    "h": int(r.h),
                    "verified_score": round(float(r.verified_score), 6),
                    "source": str(r.cand_source),
                    "track_id": int(r.track_id),
                    "stable_run_start": run["start"],
                    "stable_run_end": run["end"],
                    "stable_run_len": run["length"],
                }
            )
        else:
            best = df[df["frame"].eq(frame)].sort_values("verified_score", ascending=False).iloc[0]
            rows.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "selected": 0,
                    "rank": "",
                    "learned_score": round(float(best.verified_score), 6),
                    "threshold": 0.0,
                    "human_label": "",
                    "x": "",
                    "y": "",
                    "w": "",
                    "h": "",
                    "verified_score": round(float(best.verified_score), 6),
                    "source": "",
                    "track_id": "",
                    "stable_run_start": run["start"],
                    "stable_run_end": run["end"],
                    "stable_run_len": run["length"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.top_tubes)
    runs = [r for r in selected_runs(df, args.max_jump) if r["length"] >= args.min_run]
    if not runs:
        raise SystemExit("no stable selected runs found")
    best = max(runs, key=lambda r: (r["tube_score"], r["length"]))
    out = output_rows(df, args.clip, best)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out_path)
    print(
        f"track_id={best['track_id']} start={best['start']} end={best['end']} "
        f"length={best['length']} mean_score={best['mean_score']:.3f} tube_score={best['tube_score']:.1f}"
    )


if __name__ == "__main__":
    main()
