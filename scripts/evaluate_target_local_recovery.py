#!/usr/bin/env python3
"""Evaluate target-local proposal recovery on dense visible labels.

This is an offline oracle-style harness for the hard surface failure mode. It
does not use the current-frame label as a proposal. Instead, it predicts the
current target center from prior labels, searches a small high-resolution
window around that predicted path, and reports whether the local proposal set
contains a strict candidate.

The intent is to answer a narrower question than full tracking:

    If a tracker has recently been on the true drone, can a target-local branch
    recover centered candidates against terrain/trees where generic proposals
    miss?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import recenter_top_tube_candidates as recenter  # noqa: E402


@dataclass(frozen=True)
class Label:
    clip: str
    frame: int
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + 0.5 * self.w

    @property
    def cy(self) -> float:
        return self.y + 0.5 * self.h


@dataclass(frozen=True)
class RecoveryCandidate:
    frame: int
    x: float
    y: float
    w: float
    h: float
    score: float
    source: str
    seed_frame: int
    seed_gap: int
    pred_x: float
    pred_y: float
    method: str

    @property
    def cx(self) -> float:
        return self.x + 0.5 * self.w

    @property
    def cy(self) -> float:
        return self.y + 0.5 * self.h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, help="Visible frame label CSV.")
    p.add_argument("--video", required=True, help="Source video.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="", help="Optional clip id filter.")
    p.add_argument("--frame_min", type=int, default=-1)
    p.add_argument("--frame_max", type=int, default=-1)
    p.add_argument("--detector_scale", type=float, default=0.5)
    p.add_argument("--strict_tol_px", type=float, default=8.0, help="Detector-space strict center tolerance.")
    p.add_argument("--loose_tol_px", type=float, default=16.0, help="Detector-space loose center tolerance.")
    p.add_argument("--max_seed_gap", type=int, default=5)
    p.add_argument("--search_radius_det_px", type=float, default=18.0)
    p.add_argument("--radii_orig", default="2,3,4")
    p.add_argument("--texture_weight", type=float, default=0.010)
    p.add_argument("--peaks_per_frame", type=int, default=8)
    p.add_argument("--box_size_det_px", type=float, default=4.0)
    p.add_argument("--shift_penalty", type=float, default=0.020)
    p.add_argument(
        "--predictor",
        choices=("previous", "constant_velocity"),
        default="constant_velocity",
        help="Use prior label only, or prior two labels when available.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    out = safe_float(value)
    return default if out is None else int(round(out))


def truthy(value: Any) -> bool:
    raw = str(value).strip().lower()
    return raw not in {"", "0", "false", "no", "none", "nan", "not_visible", "not visible"}


def parse_ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def label_from_row(row: dict[str, str]) -> Label | None:
    x = safe_float(row.get("det_x", row.get("label_x", row.get("x"))))
    y = safe_float(row.get("det_y", row.get("label_y", row.get("y"))))
    w = safe_float(row.get("det_w", row.get("label_w", row.get("w"))), 1.0)
    h = safe_float(row.get("det_h", row.get("label_h", row.get("h"))), 1.0)
    frame = safe_float(row.get("frame"))
    clip = str(row.get("clip", "")).strip()
    if x is None or y is None or w is None or h is None or frame is None:
        return None
    return Label(clip=clip, frame=int(round(frame)), x=x, y=y, w=max(1.0, w), h=max(1.0, h))


def load_labels(path: Path, clip: str, frame_min: int, frame_max: int) -> list[Label]:
    labels: list[Label] = []
    for row in read_csv(path):
        if clip and row.get("clip", "") != clip:
            continue
        if "visible" in row and not truthy(row.get("visible")):
            continue
        label = label_from_row(row)
        if label is None:
            continue
        if frame_min >= 0 and label.frame < frame_min:
            continue
        if frame_max >= 0 and label.frame > frame_max:
            continue
        labels.append(label)
    labels.sort(key=lambda x: (x.clip, x.frame))
    return labels


def load_gray_frames(video: Path, frame_numbers: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    frames: dict[int, np.ndarray] = {}
    for frame_no in sorted(n for n in frame_numbers if n >= 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        frames[frame_no] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return frames


def center_dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return float(math.hypot(ax - bx, ay - by))


def center_dist_label(candidate: RecoveryCandidate, label: Label) -> float:
    return center_dist_xy(candidate.cx, candidate.cy, label.cx, label.cy)


def predict_from_history(history: list[Label], current_frame: int, max_seed_gap: int, predictor: str) -> tuple[float, float, int, int] | None:
    if not history:
        return None
    prev = history[-1]
    gap = current_frame - prev.frame
    if gap <= 0 or gap > max_seed_gap:
        return None
    pred_x, pred_y = prev.cx, prev.cy
    if predictor == "constant_velocity" and len(history) >= 2:
        prev2 = history[-2]
        prev_gap = max(1, prev.frame - prev2.frame)
        vx = (prev.cx - prev2.cx) / prev_gap
        vy = (prev.cy - prev2.cy) / prev_gap
        pred_x = prev.cx + vx * gap
        pred_y = prev.cy + vy * gap
    return pred_x, pred_y, prev.frame, gap


def recover_candidates(
    label: Label,
    prediction: tuple[float, float, int, int],
    gray: np.ndarray,
    radii_orig: list[int],
    detector_scale: float,
    search_radius_det_px: float,
    texture_weight: float,
    peaks_per_frame: int,
    box_size_det_px: float,
    shift_penalty: float,
) -> list[RecoveryCandidate]:
    pred_x_det, pred_y_det, seed_frame, seed_gap = prediction
    pred_x_orig = pred_x_det / detector_scale
    pred_y_orig = pred_y_det / detector_scale
    score_maps = {r: recenter.compact_dark_map(gray, r, texture_weight) for r in radii_orig}
    peaks = recenter.find_local_peaks(
        score_maps,
        (pred_x_orig, pred_y_orig),
        search_radius_det_px / detector_scale,
        max(1, peaks_per_frame),
    )
    out: list[RecoveryCandidate] = []
    for peak in peaks:
        cx_det = peak.x_orig * detector_scale
        cy_det = peak.y_orig * detector_scale
        shift = center_dist_xy(cx_det, cy_det, pred_x_det, pred_y_det)
        score = peak.score - shift_penalty * shift
        out.append(
            RecoveryCandidate(
                frame=label.frame,
                x=cx_det - 0.5 * box_size_det_px,
                y=cy_det - 0.5 * box_size_det_px,
                w=box_size_det_px,
                h=box_size_det_px,
                score=score,
                source="target_local_highres_dark_ring",
                seed_frame=seed_frame,
                seed_gap=seed_gap,
                pred_x=pred_x_det,
                pred_y=pred_y_det,
                method=f"dark_ring_r{peak.radius_orig}",
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out[: max(1, peaks_per_frame)]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels = load_labels(Path(args.labels), args.clip, args.frame_min, args.frame_max)
    frame_set = {label.frame for label in labels}
    frames = load_gray_frames(Path(args.video), frame_set)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_clip: dict[str, list[Label]] = defaultdict(list)
    for label in labels:
        by_clip[label.clip].append(label)

    frame_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    total = seeded = pred_strict = pred_loose = top1_strict = top1_loose = topk_strict = topk_loose = 0
    pred_dists: list[float] = []
    top1_dists: list[float] = []
    best_dists: list[float] = []

    radii_orig = parse_ints(args.radii_orig)

    for clip, clip_labels in sorted(by_clip.items()):
        history: list[Label] = []
        for label in clip_labels:
            total += 1
            prediction = predict_from_history(history, label.frame, args.max_seed_gap, args.predictor)
            gray = frames.get(label.frame)
            row: dict[str, Any] = {
                "clip": clip,
                "frame": label.frame,
                "label_x": f"{label.x:.3f}",
                "label_y": f"{label.y:.3f}",
                "label_w": f"{label.w:.3f}",
                "label_h": f"{label.h:.3f}",
                "seed_available": 0,
                "top1_dist_px": "",
                "best_dist_px": "",
                "pred_dist_px": "",
                "top1_strict": 0,
                "top1_loose": 0,
                "topk_strict": 0,
                "topk_loose": 0,
            }
            if prediction is not None and gray is not None:
                seeded += 1
                pred_x, pred_y, seed_frame, seed_gap = prediction
                pdist = center_dist_xy(pred_x, pred_y, label.cx, label.cy)
                pred_dists.append(pdist)
                if pdist <= args.strict_tol_px:
                    pred_strict += 1
                if pdist <= args.loose_tol_px:
                    pred_loose += 1

                candidates = recover_candidates(
                    label,
                    prediction,
                    gray,
                    radii_orig,
                    args.detector_scale,
                    args.search_radius_det_px,
                    args.texture_weight,
                    args.peaks_per_frame,
                    args.box_size_det_px,
                    args.shift_penalty,
                )
                dists = [center_dist_label(c, label) for c in candidates]
                if dists:
                    top1_dist = dists[0]
                    best_dist = min(dists)
                    top1_dists.append(top1_dist)
                    best_dists.append(best_dist)
                    if top1_dist <= args.strict_tol_px:
                        top1_strict += 1
                    if top1_dist <= args.loose_tol_px:
                        top1_loose += 1
                    if best_dist <= args.strict_tol_px:
                        topk_strict += 1
                    if best_dist <= args.loose_tol_px:
                        topk_loose += 1
                    row.update(
                        {
                            "seed_available": 1,
                            "seed_frame": seed_frame,
                            "seed_gap": seed_gap,
                            "pred_x": f"{pred_x:.3f}",
                            "pred_y": f"{pred_y:.3f}",
                            "pred_dist_px": f"{pdist:.3f}",
                            "top1_dist_px": f"{top1_dist:.3f}",
                            "best_dist_px": f"{best_dist:.3f}",
                            "top1_strict": int(top1_dist <= args.strict_tol_px),
                            "top1_loose": int(top1_dist <= args.loose_tol_px),
                            "topk_strict": int(best_dist <= args.strict_tol_px),
                            "topk_loose": int(best_dist <= args.loose_tol_px),
                        }
                    )
                    for rank, (candidate, dist) in enumerate(zip(candidates, dists), start=1):
                        candidate_rows.append(
                            {
                                "clip": clip,
                                "frame": label.frame,
                                "rank": rank,
                                "x": f"{candidate.x:.3f}",
                                "y": f"{candidate.y:.3f}",
                                "w": f"{candidate.w:.3f}",
                                "h": f"{candidate.h:.3f}",
                                "score": f"{candidate.score:.6f}",
                                "verified_score": f"{candidate.score:.6f}",
                                "cand_source": candidate.source,
                                "proposal_variant": candidate.source,
                                "target_local_seed_frame": candidate.seed_frame,
                                "target_local_seed_gap": candidate.seed_gap,
                                "target_local_pred_x": f"{candidate.pred_x:.3f}",
                                "target_local_pred_y": f"{candidate.pred_y:.3f}",
                                "target_local_method": candidate.method,
                                "dist_to_label_px": f"{dist:.3f}",
                                "selected": 0,
                                "eligible": 1,
                            }
                        )
            frame_rows.append(row)
            history.append(label)

    summary = {
        "labels": total,
        "seeded_frames": seeded,
        "seeded_fraction": seeded / max(1, total),
        "pred_strict": pred_strict,
        "pred_strict_rate": pred_strict / max(1, seeded),
        "pred_loose": pred_loose,
        "pred_loose_rate": pred_loose / max(1, seeded),
        "top1_strict": top1_strict,
        "top1_strict_rate": top1_strict / max(1, seeded),
        "top1_loose": top1_loose,
        "top1_loose_rate": top1_loose / max(1, seeded),
        "topk_strict": topk_strict,
        "topk_strict_rate": topk_strict / max(1, seeded),
        "topk_loose": topk_loose,
        "topk_loose_rate": topk_loose / max(1, seeded),
        "pred_dist_median": percentile(pred_dists, 50),
        "top1_dist_median": percentile(top1_dists, 50),
        "best_dist_median": percentile(best_dists, 50),
        "best_dist_p90": percentile(best_dists, 90),
        "search_radius_det_px": args.search_radius_det_px,
        "peaks_per_frame": args.peaks_per_frame,
        "max_seed_gap": args.max_seed_gap,
        "predictor": args.predictor,
    }

    write_csv(out_dir / "target_local_frame_metrics.csv", frame_rows)
    write_csv(out_dir / "target_local_candidates.csv", candidate_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(out_dir / "summary.csv", [summary])
    (out_dir / "README.md").write_text(
        "# Target-Local Proposal Recovery\n\n"
        "Offline oracle-style check for ground/surface tracking. Proposals are seeded from prior labels only, "
        "then recentered with a local high-resolution dark-ring search. This should be read as a target-local "
        "proposal-recovery upper bound, not a production tracker.\n\n"
        f"- labels: {summary['labels']}\n"
        f"- seeded frames: {summary['seeded_frames']}\n"
        f"- top-k strict: {summary['topk_strict']} ({summary['topk_strict_rate']:.3f})\n"
        f"- top-1 strict: {summary['top1_strict']} ({summary['top1_strict_rate']:.3f})\n"
        f"- prediction-only strict: {summary['pred_strict']} ({summary['pred_strict_rate']:.3f})\n"
    )
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
