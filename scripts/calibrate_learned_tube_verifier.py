#!/usr/bin/env python3
"""Calibrate learned tube-verifier scores against local review labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.44,
    0.47,
    0.50,
    0.5261650677544744,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", required=True, help="learned_tube_scores.csv from apply_sklearn_tube_verifier.py")
    p.add_argument("--labels", required=True, help="local tube_alternatives_to_label.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--thresholds", default=None, help="optional comma-separated threshold list")
    p.add_argument("--include_score_thresholds", action="store_true", help="also test every observed best-frame score")
    return p.parse_args()


def parse_thresholds(raw: str | None, scores: pd.Series, include_score_thresholds: bool) -> list[float]:
    values = list(DEFAULT_THRESHOLDS)
    if raw:
        values.extend(float(v.strip()) for v in raw.split(",") if v.strip())
    if include_score_thresholds:
        values.extend(float(v) for v in scores.dropna().unique())
    return sorted(set(round(float(v), 12) for v in values if 0.0 <= float(v) <= 1.0))


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


def best_per_frame(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.sort_values(["clip", "frame", "learned_score", "rank"], ascending=[True, True, False, True])
        .groupby(["clip", "frame"], as_index=False)
        .first()
    )


def run_stats(best: pd.DataFrame, threshold: float) -> dict[str, int]:
    run_count = 0
    run_ge3 = 0
    run_ge5 = 0
    max_run = 0
    for _, group in best.sort_values(["clip", "frame"]).groupby("clip"):
        selected = group["learned_score"].to_numpy() >= threshold
        frames = group["frame"].to_numpy()
        run = 0
        prev = None
        for frame, is_selected in zip(frames, selected):
            contiguous = prev is not None and int(frame) == int(prev) + 1
            if is_selected:
                if run and contiguous:
                    run += 1
                else:
                    if run:
                        run_count += 1
                        run_ge3 += int(run >= 3)
                        run_ge5 += int(run >= 5)
                        max_run = max(max_run, run)
                    run = 1
            elif run:
                run_count += 1
                run_ge3 += int(run >= 3)
                run_ge5 += int(run >= 5)
                max_run = max(max_run, run)
                run = 0
            prev = frame
        if run:
            run_count += 1
            run_ge3 += int(run >= 3)
            run_ge5 += int(run >= 5)
            max_run = max(max_run, run)
    return {
        "selected_runs": run_count,
        "selected_runs_ge3": run_ge3,
        "selected_runs_ge5": run_ge5,
        "selected_max_run": max_run,
    }


def frame_targets(labels: pd.DataFrame) -> pd.DataFrame:
    grouped = labels.assign(is_target=labels["human_label"].eq("target")).groupby(["clip", "frame"], as_index=False)
    return grouped.agg(has_target=("is_target", "any"), labeled_candidates=("rank", "count"))


def evaluate_checkpoint(best: pd.DataFrame, labels: pd.DataFrame, threshold: float, prefix: str) -> dict[str, Any]:
    targets = frame_targets(labels)
    labeled_lookup = labels[["clip", "frame", "rank", "human_label"]]
    merged = best.merge(targets, on=["clip", "frame"], how="inner")
    merged = merged.merge(labeled_lookup, on=["clip", "frame", "rank"], how="left", suffixes=("", "_reviewed"))
    selected = merged["learned_score"] >= threshold
    has_target = merged["has_target"].astype(bool)
    reviewed = merged["human_label"].fillna("")
    known_target = reviewed.eq("target")
    known_non_target = reviewed.ne("") & ~known_target
    unknown_selected = selected & reviewed.eq("")

    target_hit = int((has_target & selected & known_target).sum())
    target_wrong = int((has_target & selected & known_non_target).sum())
    target_unlabeled = int((has_target & unknown_selected).sum())
    target_miss = int((has_target & ~selected).sum())
    no_target_tn = int((~has_target & ~selected).sum())
    no_target_fp = int((~has_target & selected & known_non_target).sum())
    no_target_unlabeled_fp = int((~has_target & unknown_selected).sum())
    target_frames = int(has_target.sum())
    no_target_frames = int((~has_target).sum())
    selected_known = target_hit + target_wrong + no_target_fp
    selected_any = selected.sum()

    return {
        f"{prefix}_frames": int(len(merged)),
        f"{prefix}_target_frames": target_frames,
        f"{prefix}_no_target_frames": no_target_frames,
        f"{prefix}_target_hit": target_hit,
        f"{prefix}_target_wrong": target_wrong,
        f"{prefix}_target_unlabeled": target_unlabeled,
        f"{prefix}_target_miss": target_miss,
        f"{prefix}_no_target_tn": no_target_tn,
        f"{prefix}_no_target_fp": no_target_fp,
        f"{prefix}_no_target_unlabeled_fp": no_target_unlabeled_fp,
        f"{prefix}_target_recall_known": round(target_hit / max(1, target_frames), 3),
        f"{prefix}_precision_known": round(target_hit / max(1, selected_known), 3),
        f"{prefix}_precision_conservative": round(target_hit / max(1, int(selected_any)), 3),
        f"{prefix}_no_target_suppression_known": round(no_target_tn / max(1, no_target_frames), 3),
    }


def labeled_only_scores(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keys = labels[["clip", "frame", "rank"]]
    return scores.merge(keys, on=["clip", "frame", "rank"], how="inner")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.labels)
    scores["rank"] = scores["rank"].astype(int)
    scores["frame"] = scores["frame"].astype(int)
    labels["rank"] = labels["rank"].astype(int)
    labels["frame"] = labels["frame"].astype(int)

    full_best = best_per_frame(scores)
    packet_best = best_per_frame(labeled_only_scores(scores, labels))
    thresholds = parse_thresholds(args.thresholds, full_best["learned_score"], args.include_score_thresholds)

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        full_selected = int((full_best["learned_score"] >= threshold).sum())
        row: dict[str, Any] = {
            "threshold": threshold,
            "full_frames": int(len(full_best)),
            "full_selected": full_selected,
            "full_selected_frac": round(full_selected / max(1, len(full_best)), 4),
        }
        row.update(run_stats(full_best, threshold))
        row.update(evaluate_checkpoint(packet_best, labels, threshold, "packet"))
        row.update(evaluate_checkpoint(full_best, labels, threshold, "fulltop"))
        rows.append(row)

    write_csv(out_dir / "threshold_sweep.csv", rows)

    packet_preferred = [
        r
        for r in rows
        if r["packet_target_recall_known"] >= 0.80
        and r["packet_no_target_fp"] == 0
    ]
    packet_chosen = (
        min(packet_preferred, key=lambda r: (r["full_selected"], -r["packet_target_hit"]))
        if packet_preferred
        else rows[-1]
    )
    fulltop_strict = [
        r
        for r in rows
        if r["fulltop_no_target_fp"] == 0 and r["fulltop_no_target_unlabeled_fp"] == 0
    ]
    strict_chosen = (
        max(fulltop_strict, key=lambda r: (r["packet_target_hit"], -r["full_selected"]))
        if fulltop_strict
        else rows[-1]
    )
    def describe(row: dict[str, Any]) -> str:
        return (
            f"`{row['threshold']}`: packet {row['packet_target_hit']}/{row['packet_target_frames']} target hits, "
            f"{row['packet_target_wrong']} wrong target-frame selections, "
            f"{row['packet_no_target_tn']}/{row['packet_no_target_frames']} no-target frames suppressed; "
            f"full export selects {row['full_selected']}/{row['full_frames']} frames"
        )

    summary = [
        "# Learned Verifier Calibration",
        "",
        f"Scores: `{args.scores}`",
        f"Labels: `{args.labels}`",
        f"Full scored frames: {len(full_best)}",
        f"Packet labeled rows: {len(labels)}",
        f"Packet checkpoint frames: {labels[['clip', 'frame']].drop_duplicates().shape[0]}",
        "",
        "Packet-balanced threshold from current local labels:",
        f"`{packet_chosen['threshold']}`",
        describe(packet_chosen),
        "",
        "Strict full-topK checkpoint threshold:",
        f"`{strict_chosen['threshold']}`",
        describe(strict_chosen),
        "",
        "This is a calibration report, not proof of dense full-video accuracy. Full-video rows outside the review packet remain unlabeled.",
        "",
        "See `threshold_sweep.csv` for packet-only and full-topK metrics.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(out_dir / "threshold_sweep.csv")


if __name__ == "__main__":
    main()
