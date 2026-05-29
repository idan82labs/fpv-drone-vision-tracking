#!/usr/bin/env python3
"""Run full-video OOF candidate scoring plus state-machine sweeps.

This is a thin orchestration wrapper around:

- train_full_video_state_ranker.py
- evaluate_lock_state_machine.py

It keeps the workflow reproducible for complete-video benchmarks where labels
include both visible target frames and no-target/null frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ACQUIRE_THRESHOLDS = "0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9"
DEFAULT_TRACK_THRESHOLDS = "0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8"
DEFAULT_BASELINE_ACQUIRE_THRESHOLDS = "12,14,16,18,20,22,24,26,28,30,32,34,36,40"
DEFAULT_BASELINE_TRACK_THRESHOLDS = "6,8,10,12,14,16,18,20,22,24,26,28,30"
DEFAULT_JUMPS = "12,18,24,32,48"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--top_tubes", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--positive_tol_px", type=float, default=8.0)
    p.add_argument("--negative_min_dist_px", type=float, default=16.0)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold_strategy", choices=("stratified_blocks", "frame_mod"), default="stratified_blocks")
    p.add_argument("--models", nargs="+", choices=("logistic", "hist_gbdt", "extra_trees"), default=["logistic", "hist_gbdt", "extra_trees"])
    p.add_argument("--acquire_thresholds", default=DEFAULT_ACQUIRE_THRESHOLDS)
    p.add_argument("--track_thresholds", default=DEFAULT_TRACK_THRESHOLDS)
    p.add_argument("--baseline_acquire_thresholds", default=DEFAULT_BASELINE_ACQUIRE_THRESHOLDS)
    p.add_argument("--baseline_track_thresholds", default=DEFAULT_BASELINE_TRACK_THRESHOLDS)
    p.add_argument("--acquire_hits", default="1,2,3")
    p.add_argument("--max_misses", default="0,1,2")
    p.add_argument("--max_jump_px", default=DEFAULT_JUMPS)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--train_script", default="scripts/train_full_video_state_ranker.py")
    p.add_argument("--state_script", default="scripts/evaluate_lock_state_machine.py")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def with_optional_clip(cmd: list[str], clip: str) -> list[str]:
    if clip:
        return [*cmd, "--clip", clip]
    return cmd


def model_summary_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["model"]: row for row in read_csv(path)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        args.python,
        args.train_script,
        "--labels",
        args.labels,
        "--top_tubes",
        args.top_tubes,
        "--clip",
        args.clip,
        "--out_dir",
        str(out_dir),
        "--max_rank",
        str(args.max_rank),
        "--positive_tol_px",
        str(args.positive_tol_px),
        "--negative_min_dist_px",
        str(args.negative_min_dist_px),
        "--folds",
        str(args.folds),
        "--fold_strategy",
        args.fold_strategy,
        "--models",
        *args.models,
    ]
    run(train_cmd)

    model_summary = model_summary_lookup(out_dir / "model_summary.csv")
    comparison: list[dict[str, Any]] = []

    baseline_dir = out_dir / "state_machine_baseline_verified_score"
    run(
        with_optional_clip(
            [
            args.python,
            args.state_script,
            "--labels",
            args.labels,
            "--candidates",
            str(out_dir / "best_per_frame_baseline_verified_score.csv"),
            "--score_column",
            "learned_score",
            "--max_rank",
            str(args.max_rank),
            "--strict_tol_px",
            str(args.strict_tol_px),
            "--loose_tol_px",
            str(args.loose_tol_px),
            "--acquire_thresholds",
            args.baseline_acquire_thresholds,
            "--track_thresholds",
            args.baseline_track_thresholds,
            "--acquire_hits",
            args.acquire_hits,
            "--max_misses",
            args.max_misses,
            "--max_jump_px",
            args.max_jump_px,
            "--out_dir",
            str(baseline_dir),
            ],
            args.clip,
        )
    )
    baseline_cfg = json.loads((baseline_dir / "best_config.json").read_text())
    comparison.append({"model": "baseline_verified_score", **model_summary.get("baseline_verified_score", {}), **baseline_cfg})

    for model in args.models:
        score_col = f"oof_{model}_score"
        cand_path = out_dir / f"oof_best_per_frame_{model}.csv"
        if not cand_path.exists():
            continue
        state_dir = out_dir / f"state_machine_{model}"
        run(
            with_optional_clip(
                [
                args.python,
                args.state_script,
                "--labels",
                args.labels,
                "--candidates",
                str(cand_path),
                "--score_column",
                score_col,
                "--max_rank",
                str(args.max_rank),
                "--strict_tol_px",
                str(args.strict_tol_px),
                "--loose_tol_px",
                str(args.loose_tol_px),
                "--acquire_thresholds",
                args.acquire_thresholds,
                "--track_thresholds",
                args.track_thresholds,
                "--acquire_hits",
                args.acquire_hits,
                "--max_misses",
                args.max_misses,
                "--max_jump_px",
                args.max_jump_px,
                "--out_dir",
                str(state_dir),
                ],
                args.clip,
            )
        )
        cfg = json.loads((state_dir / "best_config.json").read_text())
        comparison.append({"model": model, **model_summary.get(model, {}), **cfg})

    comparison.sort(
        key=lambda r: (
            float(r.get("all_frame_accuracy", 0.0) or 0.0),
            float(r.get("visible_strict_recall", 0.0) or 0.0),
            float(r.get("invisible_no_box_rate", 0.0) or 0.0),
        ),
        reverse=True,
    )
    write_csv(out_dir / "state_machine_model_comparison.csv", comparison)
    (out_dir / "README.md").write_text(
        "# Full-Video OOF State Evaluation\n\n"
        "Generated by `scripts/run_full_video_oof_state_eval.py`.\n\n"
        "This artifact contains out-of-fold candidate scores for complete-video "
        "state-machine evaluation. Treat one-clip OOF as architecture evidence, "
        "not deployment proof.\n\n"
        "Key files:\n\n"
        "- `oof_candidate_scores_<model>.csv`: OOF scores for every labeled candidate.\n"
        "- `oof_best_per_frame_<model>.csv`: best learned candidate per frame.\n"
        "- `state_machine_<model>/`: acquire/track/null sweep for that model.\n"
        "- `state_machine_model_comparison.csv`: compact comparison across models.\n"
    )
    print(out_dir / "state_machine_model_comparison.csv")


if __name__ == "__main__":
    main()
