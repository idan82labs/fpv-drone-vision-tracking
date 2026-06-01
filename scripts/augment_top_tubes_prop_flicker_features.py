#!/usr/bin/env python3
"""Add candidate-local prop/flicker temporal features to top_tubes.csv.

This is an offline diagnostic augmenter. It does not claim to resolve propeller
blades at normal detector ranges. Instead it asks whether a candidate tube has
localized temporal AC/periodic energy that is stronger than nearby controls.
At 30-60 Hz this mostly captures aliased prop/rolling-shutter flicker and close
target shimmer, so these columns should be treated as confirmation features for
rankers/JS1 observations, not as standalone proposals.
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
    p.add_argument("--window", type=int, default=15, help="Causal frame window including current frame.")
    p.add_argument("--crop_size", type=int, default=17)
    p.add_argument("--detector_scale", type=float, default=0.5)
    p.add_argument("--control_radius", type=float, default=16.0)
    p.add_argument("--control_count", type=int, default=8)
    p.add_argument("--frame_min", type=int, default=-1)
    p.add_argument("--frame_max", type=int, default=-1)
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
        rows.append(out)
    rows.sort(key=lambda r: (align.safe_int(r.get("frame"), 0), align.safe_int(r.get("rank"), 999999)))
    return rows


def control_offsets(radius: float, count: int) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    n = max(4, int(count))
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        r = radius * (1.0 if i % 2 == 0 else 1.35)
        offsets.append((float(r * math.cos(theta)), float(r * math.sin(theta))))
    return offsets


def crop_dark_signal(crop: np.ndarray) -> float:
    """Local dark-center signal for one crop.

    Positive means the crop center is darker than the surrounding ring. This
    removes much of the frame-level exposure drift before temporal analysis.
    """

    size = int(crop.shape[0])
    center_mask, ring_mask, _outer_mask = align.masks(size)
    center_val = float(np.mean(crop[center_mask]))
    ring = crop[ring_mask]
    if ring.size == 0:
        return 0.0
    ring_med = float(np.median(ring))
    sigma = align.robust_sigma(ring)
    return float((ring_med - center_val) / sigma)


def detrend_trace(trace: np.ndarray) -> np.ndarray:
    vals = np.asarray(trace, dtype=np.float32)
    if vals.size <= 2:
        return vals - float(np.mean(vals)) if vals.size else vals
    x = np.arange(vals.size, dtype=np.float32)
    slope, intercept = np.polyfit(x, vals, 1)
    return (vals - (slope * x + intercept)).astype(np.float32)


def temporal_features(trace: list[float], fps: float = 30.0) -> dict[str, float]:
    vals = np.asarray([v for v in trace if math.isfinite(float(v))], dtype=np.float32)
    n = int(vals.size)
    if n < 4:
        return {
            "samples": float(n),
            "mean": float(np.mean(vals)) if n else 0.0,
            "ac_rms": 0.0,
            "peak_ratio": 0.0,
            "highfreq_ratio": 0.0,
            "peak_bin": 0.0,
            "peak_freq_hz": 0.0,
            "periodic_score": 0.0,
        }
    detrended = detrend_trace(vals)
    ac_rms = float(np.sqrt(np.mean(detrended * detrended)))
    window = np.hanning(n).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(detrended * window)).astype(np.float32)
    non_dc = spectrum[1:]
    if non_dc.size == 0 or float(np.sum(non_dc)) <= 1e-9:
        peak_bin = 0
        peak_ratio = 0.0
        highfreq_ratio = 0.0
    else:
        peak_idx = int(np.argmax(non_dc))
        peak_bin = peak_idx + 1
        peak = float(non_dc[peak_idx])
        # Ratio to total non-DC energy is intentionally bounded and stable.
        peak_ratio = float(peak / (float(np.sum(non_dc)) + 1e-6))
        hi_start = max(1, int(math.ceil(non_dc.size * 0.5)))
        highfreq_ratio = float(np.sum(non_dc[hi_start:]) / (float(np.sum(non_dc)) + 1e-6))
    peak_freq = float(peak_bin) * float(fps) / float(n) if n and fps > 0 else 0.0
    return {
        "samples": float(n),
        "mean": float(np.mean(vals)),
        "ac_rms": ac_rms,
        "peak_ratio": peak_ratio,
        "highfreq_ratio": highfreq_ratio,
        "peak_bin": float(peak_bin),
        "peak_freq_hz": peak_freq,
        "periodic_score": float(ac_rms * peak_ratio),
    }


def robust_gain(value: float, controls: list[float]) -> tuple[float, float, float]:
    if not controls:
        return 0.0, 0.0, 1.0
    arr = np.asarray([float(v) for v in controls if math.isfinite(float(v))], dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(0.05, 1.4826 * mad)
    return float((float(value) - med) / sigma), med, sigma


def trace_for_offset(
    row: dict[str, str],
    clip: str,
    frame_cache: align.FrameCache,
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    window: int,
    crop_size: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[float]:
    frame = align.safe_int(row.get("frame"), -1)
    trace: list[float] = []
    start = max(0, frame - max(1, int(window)) + 1)
    for target_frame in range(start, frame + 1):
        gray = frame_cache.gray(clip, target_frame)
        if gray is None:
            continue
        tx, ty = align.target_point(row, tube_rows, clip, frame, target_frame)
        crop = align.extract_crop(gray, (tx + offset[0], ty + offset[1]), crop_size)
        trace.append(crop_dark_signal(crop))
    return trace


def augment_row(
    row: dict[str, str],
    frame_cache: align.FrameCache,
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    offsets: list[tuple[float, float]],
    window: int,
    crop_size: int,
) -> dict[str, Any]:
    clip = row.get("clip", "")
    fps = float(frame_cache.capture(clip).get(cv2.CAP_PROP_FPS) or 30.0)
    target_trace = trace_for_offset(row, clip, frame_cache, tube_rows, window, crop_size)
    target = temporal_features(target_trace, fps=fps)
    control_feats = [
        temporal_features(trace_for_offset(row, clip, frame_cache, tube_rows, window, crop_size, off), fps=fps)
        for off in offsets
    ]
    control_ac = [f["ac_rms"] for f in control_feats]
    control_periodic = [f["periodic_score"] for f in control_feats]
    ac_norm, ac_med, ac_sigma = robust_gain(target["ac_rms"], control_ac)
    periodic_norm, periodic_med, periodic_sigma = robust_gain(target["periodic_score"], control_periodic)
    out: dict[str, Any] = dict(row)
    out.update(
        {
            "prop_samples": int(target["samples"]),
            "prop_fps": round(fps, 4),
            "prop_dark_mean": round(target["mean"], 6),
            "prop_ac_rms": round(target["ac_rms"], 6),
            "prop_peak_ratio": round(target["peak_ratio"], 6),
            "prop_highfreq_ratio": round(target["highfreq_ratio"], 6),
            "prop_peak_bin": int(target["peak_bin"]),
            "prop_peak_freq_hz": round(target["peak_freq_hz"], 6),
            "prop_periodic_score": round(target["periodic_score"], 6),
            "prop_control_ac_median": round(ac_med, 6),
            "prop_control_ac_sigma": round(ac_sigma, 6),
            "prop_ac_gain_norm": round(ac_norm, 6),
            "prop_control_periodic_median": round(periodic_med, 6),
            "prop_control_periodic_sigma": round(periodic_sigma, 6),
            "prop_periodic_gain_norm": round(periodic_norm, 6),
            "prop_control_count": len(control_feats),
            "prop_note": "alias_or_flicker_confirmation_not_primary_detection",
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
    tube_rows = align.load_tube_rows([top_path])
    offsets = control_offsets(args.control_radius, args.control_count)
    try:
        out_rows = [
            augment_row(row, frame_cache, tube_rows, offsets, max(1, args.window), max(7, args.crop_size | 1))
            for row in rows
        ]
    finally:
        frame_cache.close()
    write_csv(Path(args.out_csv), out_rows)
    meta = {
        "top_tubes": str(top_path),
        "out_csv": args.out_csv,
        "video_dir": args.video_dir,
        "clip": clip,
        "rows": len(out_rows),
        "max_rank": args.max_rank,
        "window": args.window,
        "crop_size": max(7, args.crop_size | 1),
        "detector_scale": args.detector_scale,
        "control_radius": args.control_radius,
        "control_count": args.control_count,
        "interpretation": "Conventional-camera prop features are confirmation/alias diagnostics, not primary small-target detection.",
    }
    Path(args.out_csv).with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.out_csv)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

