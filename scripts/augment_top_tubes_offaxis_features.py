#!/usr/bin/env python3
"""Add candidate-local off-axis/parallax motion features to top_tubes.csv.

This is an offline diagnostic augmenter. It asks whether sparse optical-flow
motion near a candidate moves independently from the dominant background model.
That is potentially useful for terrain/tree/grass failures, but it is not a
primary detector: if the drone is too small for LK support, these columns should
stay neutral and the selector should fall back to existing evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import profile_tube_alignment_features as align
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import profile_tube_alignment_features as align


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True, help="Input top_tubes.csv or scored_top_tubes.csv.")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--clip", default="", help="Clip id when rows do not include a clip column.")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--detector_scale", type=float, default=0.5)
    p.add_argument("--frame_min", type=int, default=-1)
    p.add_argument("--frame_max", type=int, default=-1)
    p.add_argument("--radius", type=float, default=18.0)
    p.add_argument("--move_min_px", type=float, default=0.55)
    p.add_argument("--control_radius", type=float, default=20.0)
    p.add_argument("--control_count", type=int, default=8)
    p.add_argument("--lk_max_features", type=int, default=1400)
    p.add_argument("--lk_quality", type=float, default=0.008)
    p.add_argument("--lk_min_distance", type=float, default=5.0)
    p.add_argument("--lk_fb_px", type=float, default=1.5)
    p.add_argument(
        "--frames_csv",
        default="",
        help="Optional labels/review CSV. When set, only these frame numbers are augmented.",
    )
    p.add_argument("--frames_clip", default="", help="Clip filter for --frames_csv. Defaults to --clip.")
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


def load_frame_filter(path: Path, clip: str) -> set[int]:
    frames: set[int] = set()
    for row in read_csv(path):
        if clip and row.get("clip", clip) != clip:
            continue
        frame = align.safe_int(row.get("frame"), -1)
        if frame >= 0:
            frames.add(frame)
    return frames


def load_rows(
    path: Path,
    clip: str,
    max_rank: int,
    frame_min: int,
    frame_max: int,
    frame_filter: set[int] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    inferred_clip = align.clip_id_from_path(path)
    for row in read_csv(path):
        rank = align.safe_int(row.get("rank"), 999999)
        frame = align.safe_int(row.get("frame"), -1)
        if rank > max_rank or frame < 0:
            continue
        if frame_min >= 0 and frame < frame_min:
            continue
        if frame_max >= 0 and frame > frame_max:
            continue
        if frame_filter is not None and frame not in frame_filter:
            continue
        out = dict(row)
        if clip and not out.get("clip"):
            out["clip"] = clip
        elif not out.get("clip"):
            out["clip"] = inferred_clip
        rows.append(out)
    rows.sort(key=lambda r: (align.safe_int(r.get("frame"), 0), align.safe_int(r.get("rank"), 999999)))
    return rows


def to_h3(transform: np.ndarray | None) -> np.ndarray:
    """Return a 3x3 homogeneous transform from a 2x3 or 3x3 matrix."""

    if transform is None:
        return np.eye(3, dtype=np.float32)
    mat = np.asarray(transform, dtype=np.float32)
    if mat.shape == (2, 3):
        out = np.eye(3, dtype=np.float32)
        out[:2, :] = mat
        return out
    if mat.shape == (3, 3):
        return mat
    raise ValueError(f"expected 2x3 or 3x3 transform, got {mat.shape}")


def predict_points(transform: np.ndarray | None, pts: np.ndarray) -> np.ndarray:
    """Project Nx2 points with a 2x3 affine or 3x3 homography."""

    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    h3 = to_h3(transform)
    return cv2.perspectiveTransform(arr, h3).reshape(-1, 2)


def control_offsets(radius: float, count: int) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    n = max(4, int(count))
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        r = float(radius) * (1.0 if i % 2 == 0 else 1.35)
        offsets.append((float(r * math.cos(theta)), float(r * math.sin(theta))))
    return offsets


def robust_gain(value: float, controls: list[float]) -> tuple[float, float, float]:
    arr = np.asarray([float(v) for v in controls if math.isfinite(float(v))], dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(0.05, 1.4826 * mad)
    return float((float(value) - med) / sigma), med, sigma


def edge_proximity(cx: float, cy: float, frame_wh: tuple[int, int]) -> float:
    w, h = frame_wh
    if w <= 0 or h <= 0:
        return 0.0
    d = min(float(cx), float(cy), float(w - 1 - cx), float(h - 1 - cy))
    return float(max(0.0, min(1.0, 1.0 - d / max(1.0, 0.12 * min(w, h)))))


def compute_offaxis_signals(
    prev_pts: np.ndarray,
    cur_pts: np.ndarray,
    transform: np.ndarray | None,
    cx: float,
    cy: float,
    frame_wh: tuple[int, int],
    radius: float = 18.0,
    move_min_px: float = 0.55,
) -> dict[str, float]:
    prev = np.asarray(prev_pts, dtype=np.float32).reshape(-1, 2)
    cur = np.asarray(cur_pts, dtype=np.float32).reshape(-1, 2)
    if prev.size == 0 or cur.size == 0 or prev.shape[0] != cur.shape[0]:
        return neutral_features(cx, cy, frame_wh)

    pred = predict_points(transform, prev)
    center = np.asarray([cx, cy], dtype=np.float32)
    d = np.linalg.norm(cur - center[None, :], axis=1)
    nearby = d <= float(radius)
    near_count = int(np.sum(nearby))
    if near_count == 0:
        return neutral_features(cx, cy, frame_wh, near_count=0)

    residual = cur[nearby] - pred[nearby]
    bg_flow = pred[nearby] - prev[nearby]
    residual_mag = np.linalg.norm(residual, axis=1)
    bg_mag = np.linalg.norm(bg_flow, axis=1)
    moving = residual_mag >= float(move_min_px)
    mover_count = int(np.sum(moving))

    if mover_count == 0:
        return {
            **neutral_features(cx, cy, frame_wh, near_count=near_count),
            "offaxis_bg_flow_mag": round(float(np.mean(bg_mag)) if bg_mag.size else 0.0, 6),
            "offaxis_residual_mag": round(float(np.mean(residual_mag)) if residual_mag.size else 0.0, 6),
        }

    rr = residual[moving]
    bb = bg_flow[moving]
    rr_mag = np.linalg.norm(rr, axis=1)
    bb_mag = np.linalg.norm(bb, axis=1)
    denom = np.maximum(rr_mag * np.maximum(bb_mag, 1e-3), 1e-6)
    cosang = np.sum(rr * bb, axis=1) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosang))

    unit = rr / np.maximum(rr_mag[:, None], 1e-6)
    coherence = float(np.linalg.norm(np.mean(unit, axis=0)))
    support_frac = float(np.mean(angles >= 70.0))
    # Confirmation score: independent residual, off-axis to background flow,
    # and coherent enough to avoid pure LK noise.
    angle_gain = float(np.mean(np.maximum(0.0, angles - 55.0) / 125.0))
    mag_gain = float(np.mean(np.minimum(rr_mag / 4.0, 1.5)))
    indep_score = float(angle_gain * mag_gain * (0.35 + 0.65 * coherence) * min(1.0, mover_count / 4.0))

    return {
        "offaxis_near_count": near_count,
        "offaxis_mover_count": mover_count,
        "offaxis_angle": round(float(np.mean(angles)), 6),
        "offaxis_angle_max": round(float(np.max(angles)), 6),
        "offaxis_bg_flow_mag": round(float(np.mean(bb_mag)) if bb_mag.size else 0.0, 6),
        "offaxis_residual_mag": round(float(np.mean(rr_mag)) if rr_mag.size else 0.0, 6),
        "offaxis_residual_coherence": round(coherence, 6),
        "offaxis_support_frac": round(support_frac, 6),
        "offaxis_support": float(mover_count) * support_frac,
        "offaxis_edge_proximity": round(edge_proximity(cx, cy, frame_wh), 6),
        "offaxis_indep_score": round(indep_score, 6),
    }


def neutral_features(
    cx: float,
    cy: float,
    frame_wh: tuple[int, int],
    near_count: int = 0,
) -> dict[str, float]:
    return {
        "offaxis_near_count": near_count,
        "offaxis_mover_count": 0,
        "offaxis_angle": 0.0,
        "offaxis_angle_max": 0.0,
        "offaxis_bg_flow_mag": 0.0,
        "offaxis_residual_mag": 0.0,
        "offaxis_residual_coherence": 0.0,
        "offaxis_support_frac": 0.0,
        "offaxis_support": 0.0,
        "offaxis_edge_proximity": round(edge_proximity(cx, cy, frame_wh), 6),
        "offaxis_indep_score": 0.0,
    }


def estimate_frame_flow(
    prev_gray: np.ndarray | None,
    cur_gray: np.ndarray | None,
    max_features: int = 1400,
    quality: float = 0.008,
    min_distance: float = 5.0,
    fb_px: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if prev_gray is None or cur_gray is None:
        return empty_flow("missing_frame")

    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max(32, int(max_features)),
        qualityLevel=float(quality),
        minDistance=float(min_distance),
        blockSize=5,
    )
    if pts is None or len(pts) < 8:
        return empty_flow("too_few_features")

    next_pts, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, winSize=(15, 15), maxLevel=3)
    if next_pts is None or st is None:
        return empty_flow("lk_failed")
    back_pts, st_back, _err_back = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, next_pts, None, winSize=(15, 15), maxLevel=3)
    if back_pts is None or st_back is None:
        return empty_flow("lk_back_failed")

    p0 = pts.reshape(-1, 2)
    p1 = next_pts.reshape(-1, 2)
    pback = back_pts.reshape(-1, 2)
    ok = (st.reshape(-1) > 0) & (st_back.reshape(-1) > 0)
    fb = np.linalg.norm(pback - p0, axis=1)
    ok &= fb <= float(fb_px)
    prev = p0[ok].astype(np.float32)
    cur = p1[ok].astype(np.float32)
    if prev.shape[0] < 8:
        return empty_flow("too_few_tracked")

    aff, inliers = cv2.estimateAffinePartial2D(
        prev.reshape(-1, 1, 2),
        cur.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1500,
    )
    h3 = np.eye(3, dtype=np.float32)
    inlier_count = 0
    model = "identity"
    if aff is not None and inliers is not None and int(inliers.sum()) >= 8:
        h3[:2, :] = aff.astype(np.float32)
        inlier_count = int(inliers.sum())
        model = "affine_partial"
    meta = {
        "flow_status": model,
        "flow_points": int(prev.shape[0]),
        "flow_inliers": inlier_count,
        "flow_inlier_rate": round(float(inlier_count / max(1, prev.shape[0])), 6),
    }
    return prev, cur, h3, meta


def empty_flow(status: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        {"flow_status": status, "flow_points": 0, "flow_inliers": 0, "flow_inlier_rate": 0.0},
    )


class FlowCache:
    def __init__(self, frame_cache: align.FrameCache, args: argparse.Namespace) -> None:
        self.frame_cache = frame_cache
        self.args = args
        self.cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], tuple[int, int]]] = {}

    def flow(
        self,
        clip: str,
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], tuple[int, int]]:
        key = (clip, frame)
        if key in self.cache:
            return self.cache[key]
        prev_gray = self.frame_cache.gray(clip, frame - 1)
        cur_gray = self.frame_cache.gray(clip, frame)
        if cur_gray is None and prev_gray is not None:
            frame_wh = (int(prev_gray.shape[1]), int(prev_gray.shape[0]))
        elif cur_gray is not None:
            frame_wh = (int(cur_gray.shape[1]), int(cur_gray.shape[0]))
        else:
            frame_wh = (0, 0)
        prev, cur, h3, meta = estimate_frame_flow(
            prev_gray,
            cur_gray,
            max_features=self.args.lk_max_features,
            quality=self.args.lk_quality,
            min_distance=self.args.lk_min_distance,
            fb_px=self.args.lk_fb_px,
        )
        self.cache[key] = (prev, cur, h3, meta, frame_wh)
        return self.cache[key]


def augment_row(
    row: dict[str, str],
    flow_cache: FlowCache,
    offsets: list[tuple[float, float]],
    radius: float,
    move_min_px: float,
) -> dict[str, Any]:
    clip = row.get("clip", "")
    frame = align.safe_int(row.get("frame"), -1)
    cx, cy = align.center(row)
    prev, cur, h3, flow_meta, frame_wh = flow_cache.flow(clip, frame)
    base = compute_offaxis_signals(prev, cur, h3, cx, cy, frame_wh, radius=radius, move_min_px=move_min_px)
    control_scores: list[float] = []
    control_support: list[float] = []
    for ox, oy in offsets:
        ctrl = compute_offaxis_signals(prev, cur, h3, cx + ox, cy + oy, frame_wh, radius=radius, move_min_px=move_min_px)
        control_scores.append(float(ctrl["offaxis_indep_score"]))
        control_support.append(float(ctrl["offaxis_support"]))
    score_gain, score_med, score_sigma = robust_gain(float(base["offaxis_indep_score"]), control_scores)
    support_gain, support_med, support_sigma = robust_gain(float(base["offaxis_support"]), control_support)

    out: dict[str, Any] = dict(row)
    out.update(flow_meta)
    out.update(base)
    out.update(
        {
            "offaxis_control_count": len(control_scores),
            "offaxis_control_score_median": round(score_med, 6),
            "offaxis_control_score_sigma": round(score_sigma, 6),
            "offaxis_indep_gain_norm": round(score_gain, 6),
            "offaxis_control_support_median": round(support_med, 6),
            "offaxis_control_support_sigma": round(support_sigma, 6),
            "offaxis_support_gain_norm": round(support_gain, 6),
            "offaxis_note": "offline_confirmation_feature_not_primary_detection",
        }
    )
    return out


def main() -> None:
    args = parse_args()
    clip = args.clip
    frame_filter = None
    if args.frames_csv:
        frame_filter = load_frame_filter(Path(args.frames_csv), args.frames_clip or clip)
    rows = load_rows(Path(args.top_tubes), clip, args.max_rank, args.frame_min, args.frame_max, frame_filter)
    if not rows:
        raise SystemExit("no rows to augment")
    top_path = Path(args.top_tubes)
    if not clip:
        clip = rows[0].get("clip") or align.clip_id_from_path(top_path)
    for row in rows:
        if not row.get("clip"):
            row["clip"] = clip

    frame_cache = align.FrameCache(Path(args.video_dir), args.detector_scale)
    flow_cache = FlowCache(frame_cache, args)
    offsets = control_offsets(args.control_radius, args.control_count)
    try:
        out_rows = [augment_row(row, flow_cache, offsets, args.radius, args.move_min_px) for row in rows]
    finally:
        frame_cache.close()

    write_csv(Path(args.out_csv), out_rows)
    flow_status_counts: dict[str, int] = {}
    for row in out_rows:
        key = str(row.get("flow_status", ""))
        flow_status_counts[key] = flow_status_counts.get(key, 0) + 1
    meta = {
        "top_tubes": str(top_path),
        "out_csv": args.out_csv,
        "video_dir": args.video_dir,
        "clip": clip,
        "rows": len(out_rows),
        "max_rank": args.max_rank,
        "detector_scale": args.detector_scale,
        "radius": args.radius,
        "move_min_px": args.move_min_px,
        "control_radius": args.control_radius,
        "control_count": args.control_count,
        "flow_status_counts": flow_status_counts,
        "interpretation": "Off-axis features are LK-supported terrain/clutter diagnostics; neutral values mean no local support.",
    }
    Path(args.out_csv).with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.out_csv)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
