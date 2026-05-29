#!/usr/bin/env python3
"""Sweep CLBA score modifiers through the proven acquire/track selector.

This is a deliberately narrower follow-up to the explicit-state selector.  The
full A/P/T/S/E/C prototype underperformed, but the professor's core primitive
still has a clean test:

    learned candidate probability
    + target-aligned evidence
    - background/attached clutter evidence

The script keeps the existing lock-state-machine behavior and only changes the
per-candidate score used for acquisition/tracking.  If this fails, the CLBA
terms are not yet reliable as direct selector modifiers and should stay as
ranker features only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

try:
    import evaluate_lock_state_machine as lock
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import evaluate_lock_state_machine as lock


@dataclass(frozen=True)
class Weights:
    gain: float = 0.0
    path: float = 0.0
    target_q: float = 0.0
    bg: float = 0.0
    attached: float = 0.0
    density: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--candidates", required=True, help="OOF per-candidate CSV with CLBA columns.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--base_score_column", default="score")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--gain_weights", default="0,0.15,0.3")
    p.add_argument("--path_weights", default="0,0.1")
    p.add_argument("--target_q_weights", default="0,0.05")
    p.add_argument("--bg_weights", default="0,0.1,0.2")
    p.add_argument("--attached_weights", default="0,0.1,0.2")
    p.add_argument("--density_weights", default="0,0.03")
    p.add_argument("--acquire_thresholds", default="0.2,0.35,0.55,0.65,0.75,0.85,0.9")
    p.add_argument("--track_thresholds", default="0.02,0.05,0.15,0.3,0.5,0.65")
    p.add_argument("--acquire_hits", default="1,2,3")
    p.add_argument("--max_misses", default="0,1,2")
    p.add_argument("--max_jump_px", default="12,18,32")
    p.add_argument("--output_tentative", action="store_true")
    p.add_argument("--coast_output", action="store_true")
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


def fnum(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sigmoid(logit_value: float) -> float:
    if logit_value >= 0:
        z = math.exp(-logit_value)
        return 1.0 / (1.0 + z)
    z = math.exp(logit_value)
    return z / (1.0 + z)


def score_to_logit(score: float) -> float:
    """Map a model probability or raw positive score onto a bounded logit."""

    if 0.0 <= score <= 1.0:
        p = clamp(score, 1e-5, 1.0 - 1e-5)
        return math.log(p / (1.0 - p))
    # Detector scores can be larger than one.  Keep them monotone but bounded so
    # CLBA modifiers cannot be numerically drowned out.
    return clamp(math.log1p(max(0.0, score)), -6.0, 6.0)


def max_feature(row: dict[str, str], *names: str) -> float:
    return max(fnum(row.get(name)) for name in names)


def row_terms(row: dict[str, str]) -> dict[str, float]:
    """Return robustly clipped feature terms used by the adjustment."""

    path_dist = fnum(row.get("clba_path_bg_dist_mean"), fnum(row.get("tube_mean_bg_dist")))
    line = max_feature(row, "cand_line_context", "tube_mean_line_context")
    support = max_feature(row, "cand_attached_support", "tube_mean_attached_support")
    attached_likelihood = fnum(row.get("clba_attached_likelihood"))
    bg_static = fnum(row.get("clba_bg_static_likelihood"))
    bg_q = fnum(row.get("clba_bg_q"))
    density = fnum(row.get("tube_log_cand_density"), math.log1p(max_feature(row, "tube_mean_cand_density")))

    attached_context = max(0.0, attached_likelihood) + 0.35 * max(0.0, line) + 0.035 * max(0.0, support)
    bg_context = max(0.0, bg_q) + 0.55 * max(0.0, bg_static)
    return {
        "gain": clamp(fnum(row.get("clba_gain_norm"), fnum(row.get("tube_mean_align_gain"))), -4.0, 4.0),
        "path": clamp(path_dist / 8.0, 0.0, 2.0),
        "target_q": clamp(fnum(row.get("clba_target_q"), fnum(row.get("tube_mean_native_dark_score"))), -3.0, 4.0),
        "bg": clamp(bg_context, 0.0, 6.0),
        "attached": clamp(attached_context, 0.0, 6.0),
        "density": clamp(density, 0.0, 8.0),
    }


def adjusted_logit(row: dict[str, str], weights: Weights, base_score_column: str) -> float:
    base = fnum(row.get(base_score_column), fnum(row.get("score")))
    terms = row_terms(row)
    return (
        score_to_logit(base)
        + weights.gain * terms["gain"]
        + weights.path * terms["path"]
        + weights.target_q * terms["target_q"]
        - weights.bg * terms["bg"]
        - weights.attached * terms["attached"]
        - weights.density * terms["density"]
    )


def adjusted_score(row: dict[str, str], weights: Weights, base_score_column: str) -> float:
    return sigmoid(adjusted_logit(row, weights, base_score_column))


def load_candidate_rows(path: Path, clip: str, max_rank: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(path):
        if clip and row.get("clip", clip) != clip:
            continue
        frame = int(fnum(row.get("frame"), -1))
        rank = int(fnum(row.get("rank"), 999999))
        if frame < 0 or rank > max_rank:
            continue
        rows.append(row)
    rows.sort(key=lambda r: (int(fnum(r.get("frame"), 0)), int(fnum(r.get("rank"), 999999))))
    return rows


def candidate_map_from_rows(
    rows: list[dict[str, str]],
    weights: Weights,
    base_score_column: str,
) -> dict[int, lock.Candidate]:
    out: dict[int, lock.Candidate] = {}
    for row in rows:
        frame = int(fnum(row.get("frame"), -1))
        if frame < 0:
            continue
        score = adjusted_score(row, weights, base_score_column)
        rank = int(fnum(row.get("rank"), 999999))
        x = fnum(row.get("x"), math.nan)
        y = fnum(row.get("y"), math.nan)
        w = max(1.0, fnum(row.get("w"), 1.0))
        h = max(1.0, fnum(row.get("h"), 1.0))
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            continue
        cand = lock.Candidate(score=score, track_score=score, rank=rank, bbox=(x, y, w, h))
        if frame not in out or cand.score > out[frame].score:
            out[frame] = cand
    return out


def adjusted_rows(rows: list[dict[str, str]], weights: Weights, base_score_column: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        terms = row_terms(row)
        score = adjusted_score(row, weights, base_score_column)
        new = dict(row)
        new.update(
            {
                "adjusted_score": round(score, 9),
                "adjusted_logit": round(adjusted_logit(row, weights, base_score_column), 9),
                "adjust_gain_term": round(weights.gain * terms["gain"], 9),
                "adjust_path_term": round(weights.path * terms["path"], 9),
                "adjust_target_q_term": round(weights.target_q * terms["target_q"], 9),
                "adjust_bg_term": round(-weights.bg * terms["bg"], 9),
                "adjust_attached_term": round(-weights.attached * terms["attached"], 9),
                "adjust_density_term": round(-weights.density * terms["density"], 9),
            }
        )
        out.append(new)
    return out


def best_per_frame_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame = int(fnum(row.get("frame"), -1))
        if frame < 0:
            continue
        score = fnum(row.get("adjusted_score"), -1e9)
        if frame not in by_frame or score > fnum(by_frame[frame].get("adjusted_score"), -1e9):
            by_frame[frame] = row
    return [by_frame[f] for f in sorted(by_frame)]


def weight_grid(args: argparse.Namespace) -> list[Weights]:
    return [
        Weights(*vals)
        for vals in product(
            parse_float_list(args.gain_weights),
            parse_float_list(args.path_weights),
            parse_float_list(args.target_q_weights),
            parse_float_list(args.bg_weights),
            parse_float_list(args.attached_weights),
            parse_float_list(args.density_weights),
        )
    ]


def evaluate_weight_set(
    labels: dict[int, lock.Label],
    candidate_rows: list[dict[str, str]],
    weights: Weights,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = candidate_map_from_rows(candidate_rows, weights, args.base_score_column)
    best_summary: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    best_sort_key = (-1.0, -1.0, -1.0, 0)
    for aq in parse_float_list(args.acquire_thresholds):
        for tr in parse_float_list(args.track_thresholds):
            if tr > aq:
                continue
            for hits in parse_int_list(args.acquire_hits):
                for misses in parse_int_list(args.max_misses):
                    for jump in parse_float_list(args.max_jump_px):
                        summary, rows = lock.evaluate_rows(
                            labels,
                            candidates,
                            acquire_threshold=aq,
                            track_threshold=tr,
                            acquire_hits=hits,
                            max_misses=misses,
                            max_jump_px=jump,
                            strict_tol_px=args.strict_tol_px,
                            loose_tol_px=args.loose_tol_px,
                            output_tentative=args.output_tentative,
                            coast_output=args.coast_output,
                        )
                        sort_key = (
                            summary["all_frame_accuracy"],
                            summary["visible_strict_recall"],
                            summary["invisible_no_box_rate"],
                            -summary["selected_frames"],
                        )
                        if sort_key > best_sort_key:
                            best_sort_key = sort_key
                            best_summary = summary
                            best_rows = rows
    if best_summary is None:
        raise RuntimeError("no selector summaries produced")
    best_summary = dict(best_summary)
    best_summary.update(asdict(weights))
    return best_summary, best_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = lock.load_labels(Path(args.labels), args.clip)
    candidate_rows = load_candidate_rows(Path(args.candidates), args.clip, args.max_rank)
    if not labels:
        raise SystemExit("no labels loaded")
    if not candidate_rows:
        raise SystemExit("no candidates loaded")

    summaries: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    for weights in weight_grid(args):
        summary, rows = evaluate_weight_set(labels, candidate_rows, weights, args)
        summaries.append(summary)
        sort_key = (
            summary["all_frame_accuracy"],
            summary["visible_strict_recall"],
            summary["invisible_no_box_rate"],
            -summary["selected_frames"],
        )
        if best_summary is None or sort_key > (
            best_summary["all_frame_accuracy"],
            best_summary["visible_strict_recall"],
            best_summary["invisible_no_box_rate"],
            -best_summary["selected_frames"],
        ):
            best_summary = summary
            best_rows = rows

    summaries.sort(
        key=lambda r: (
            r["all_frame_accuracy"],
            r["visible_strict_recall"],
            r["invisible_no_box_rate"],
            -r["selected_frames"],
        ),
        reverse=True,
    )
    best_weights = Weights(
        gain=float(best_summary["gain"]),
        path=float(best_summary["path"]),
        target_q=float(best_summary["target_q"]),
        bg=float(best_summary["bg"]),
        attached=float(best_summary["attached"]),
        density=float(best_summary["density"]),
    )
    best_adjusted = adjusted_rows(candidate_rows, best_weights, args.base_score_column)
    write_csv(out_dir / "clba_score_adjustment_sweep.csv", summaries)
    write_csv(out_dir / "best_frame_predictions.csv", best_rows)
    write_csv(out_dir / "best_adjusted_candidates.csv", best_adjusted)
    write_csv(out_dir / "best_adjusted_per_frame_candidates.csv", best_per_frame_rows(best_adjusted))
    (out_dir / "best_config.json").write_text(json.dumps(best_summary, indent=2))
    (out_dir / "README.md").write_text(
        "# CLBA Score Adjustment Sweep\n\n"
        "This artifact keeps the existing acquire/track state machine and sweeps direct\n"
        "CLBA score modifiers over out-of-fold candidate probabilities. A gain over the\n"
        "zero-weight row means the target/background-alignment primitive is useful as a\n"
        "selector modifier. No gain means it should stay inside the learned ranker only.\n"
    )
    print(out_dir / "clba_score_adjustment_sweep.csv")
    print(json.dumps(best_summary, indent=2))


if __name__ == "__main__":
    main()
