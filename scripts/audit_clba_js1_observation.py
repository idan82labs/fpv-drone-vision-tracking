#!/usr/bin/env python3
"""Audit CLBA-JS1 observation ranking against labeled boxes.

This separates proposal recall from observation quality. For each visible label,
it finds the nearest candidate in exported top tubes and asks whether the
CLBA-JS1 target log-likelihood ranks that candidate above false competitors.
For invisible labels, it records the strongest target log-likelihood so null
calibration can be inspected separately from tracking state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import evaluate_explicit_state_selector as explicit
except ModuleNotFoundError:  # pragma: no cover
    from scripts import evaluate_explicit_state_selector as explicit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--candidates", required=True, help="top_tubes.csv or scored candidates CSV.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--score_column", default="score")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--score_weight", type=float, default=0.30)
    p.add_argument("--proposal_clip", type=float, default=2.0)
    p.add_argument("--proposal_mode", choices=("shared", "target_only"), default="shared")
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--null_priors", default="")
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


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return float(math.hypot(ax - bx, ay - by))


def load_labels(path: Path, clip: str) -> dict[int, explicit.Label]:
    return explicit.load_labels(path, clip)


def load_candidates(
    path: Path,
    clip: str,
    max_rank: int,
    score_column: str,
    null_priors: str,
    score_weight: float,
    proposal_clip: float,
    proposal_mode: str,
) -> dict[int, list[explicit.Candidate]]:
    class Args:
        pass

    args = Args()
    args.score_weight = score_weight
    args.clba_weight = 0.55
    args.path_weight = 0.25
    args.static_weight = 0.70
    args.attached_weight = 0.70
    args.rank_weight = 0.10
    args.proposal_clip = proposal_clip
    args.proposal_mode = proposal_mode
    args.null_priors = null_priors

    return explicit.load_candidates(path, clip, max_rank, score_column, args)


def audit_frame(
    clip: str,
    frame: int,
    label: explicit.Label,
    candidates: list[explicit.Candidate],
    strict_tol: float,
    loose_tol: float,
) -> dict[str, Any]:
    if not candidates:
        return {
            "clip": clip,
            "frame": frame,
            "visible": int(label.visible),
            "status": "no_candidates",
        }
    best_obs = max(candidates, key=lambda c: c.target_llr)
    row: dict[str, Any] = {
        "clip": clip,
        "frame": frame,
        "visible": int(label.visible),
        "status": "audited",
        "best_obs_rank": best_obs.rank,
        "best_obs_target_llr": round(best_obs.target_llr, 6),
        "best_obs_router": best_obs.router_bucket,
    }
    if not label.visible or label.bbox is None:
        row.update(
            {
                "oracle_strict": "",
                "oracle_loose": "",
                "true_rank": "",
                "true_target_llr": "",
                "true_beats_best_false": "",
                "obs_selected_strict": "",
                "obs_selected_loose": "",
            }
        )
        return row

    ranked = [(center_dist(c.bbox, label.bbox), c) for c in candidates]
    ranked.sort(key=lambda item: item[0])
    true_dist, true_cand = ranked[0]
    strict = true_dist <= strict_tol
    loose = true_dist <= loose_tol
    false_competitors = [c for d, c in ranked if d > loose_tol]
    best_false = max(false_competitors, key=lambda c: c.target_llr) if false_competitors else None
    obs_dist = center_dist(best_obs.bbox, label.bbox)
    row.update(
        {
            "oracle_strict": int(strict),
            "oracle_loose": int(loose),
            "true_rank": true_cand.rank,
            "true_dist_px": round(true_dist, 3),
            "true_target_llr": round(true_cand.target_llr, 6),
            "true_static_llr": round(true_cand.static_llr, 6),
            "true_attached_llr": round(true_cand.attached_llr, 6),
            "true_boundary_llr": round(true_cand.boundary_llr, 6),
            "best_false_rank": "" if best_false is None else best_false.rank,
            "best_false_target_llr": "" if best_false is None else round(best_false.target_llr, 6),
            "true_beats_best_false": "" if best_false is None else int(true_cand.target_llr > best_false.target_llr),
            "obs_selected_strict": int(obs_dist <= strict_tol),
            "obs_selected_loose": int(obs_dist <= loose_tol),
            "obs_selected_dist_px": round(obs_dist, 3),
        }
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("clip", ""))].append(row)
    out: list[dict[str, Any]] = []
    for clip, clip_rows in sorted(groups.items()):
        visible = [r for r in clip_rows if r.get("visible") == 1 and r.get("status") == "audited"]
        invisible = [r for r in clip_rows if r.get("visible") == 0 and r.get("status") == "audited"]
        strict_oracle = sum(int(r.get("oracle_strict") or 0) for r in visible)
        loose_oracle = sum(int(r.get("oracle_loose") or 0) for r in visible)
        obs_strict = sum(int(r.get("obs_selected_strict") or 0) for r in visible)
        obs_loose = sum(int(r.get("obs_selected_loose") or 0) for r in visible)
        pair_rows = [r for r in visible if r.get("true_beats_best_false") != ""]
        pair_wins = sum(int(r.get("true_beats_best_false") or 0) for r in pair_rows)
        out.append(
            {
                "clip": clip,
                "visible_frames": len(visible),
                "invisible_frames": len(invisible),
                "oracle_strict_rate": round(strict_oracle / max(1, len(visible)), 4),
                "oracle_loose_rate": round(loose_oracle / max(1, len(visible)), 4),
                "obs_top_strict_rate": round(obs_strict / max(1, len(visible)), 4),
                "obs_top_loose_rate": round(obs_loose / max(1, len(visible)), 4),
                "pairwise_true_over_false_rate": round(pair_wins / max(1, len(pair_rows)), 4),
                "pairwise_frames": len(pair_rows),
                "null_frames": len(invisible),
                "null_max_target_llr_mean": round(
                    sum(float(r.get("best_obs_target_llr") or 0.0) for r in invisible) / max(1, len(invisible)),
                    6,
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels), args.clip)
    candidates = load_candidates(
        Path(args.candidates),
        args.clip,
        args.max_rank,
        args.score_column,
        args.null_priors,
        args.score_weight,
        args.proposal_clip,
        args.proposal_mode,
    )
    rows = [
        audit_frame(args.clip, frame, label, candidates.get(frame, []), args.strict_tol_px, args.loose_tol_px)
        for frame, label in sorted(labels.items())
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_csv(out_dir / "observation_audit.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "labels": args.labels,
                "candidates": args.candidates,
                "clip": args.clip,
                "score_column": args.score_column,
                "max_rank": args.max_rank,
                "score_weight": args.score_weight,
                "proposal_clip": args.proposal_clip,
                "proposal_mode": args.proposal_mode,
            },
            indent=2,
        )
        + "\n"
    )
    print(out_dir / "summary.csv")
    if summary:
        print(json.dumps(summary[0], indent=2))


if __name__ == "__main__":
    main()
