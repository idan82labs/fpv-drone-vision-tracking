#!/usr/bin/env python3
"""Add candidate-local background-alignment features to exported top tubes.

This is the offline bridge for the CLBA experiment:

    target-aligned tube quality - background-aligned tube quality

The script is intentionally an augmenter. It writes a new top-tubes CSV with
extra ``clba_*`` columns so the existing OOF ranker and state-machine harnesses
can evaluate the observation primitive before the live detector path changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import profile_tube_alignment_features as align
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import profile_tube_alignment_features as align


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True, help="Input top_tubes.csv or scored_top_tubes.csv.")
    p.add_argument("--out_csv", required=True, help="Augmented output CSV.")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--clip", default="", help="Clip id when the top-tubes rows do not include a clip column.")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--window_radius", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=31)
    p.add_argument("--detector_scale", type=float, default=0.5)
    p.add_argument("--control_radius", type=float, default=18.0)
    p.add_argument("--control_count", type=int, default=12)
    p.add_argument("--orb_features", type=int, default=900)
    p.add_argument("--min_matches", type=int, default=18)
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


def sample_control_offsets(radius: float, count: int) -> list[tuple[float, float]]:
    """Return deterministic annulus control offsets around a candidate center."""

    n = max(4, int(count))
    offsets: list[tuple[float, float]] = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        # Alternate slightly different radii so controls are not all collinear
        # with local texture periodicity.
        r = radius * (1.0 if i % 2 == 0 else 1.45)
        offsets.append((float(r * math.cos(theta)), float(r * math.sin(theta))))
    return offsets


def broad_router_bucket(row: dict[str, str]) -> str:
    state = str(row.get("cand_router_state", "")).strip()
    if state == "surface_backed":
        return "surface"
    if state == "line_attached":
        return "line"
    if state in {"boundary_mixed", "sky_target_near_surface"}:
        return "boundary"
    if state == "clean_sky":
        return "clean_sky"
    rates = {
        "surface": align.safe_float(row.get("tube_router_surface_backed_rate")),
        "line": align.safe_float(row.get("tube_router_line_attached_rate")),
        "boundary": align.safe_float(row.get("tube_router_boundary_rate")),
        "clean_sky": align.safe_float(row.get("tube_router_clean_sky_rate")),
    }
    best, value = max(rates.items(), key=lambda item: item[1])
    return best if value >= 0.35 else "unknown"


def row_center_dist(a: dict[str, str], b: dict[str, str]) -> float:
    ax, ay = align.center(a)
    bx, by = align.center(b)
    return float(math.hypot(ax - bx, ay - by))


def context_similar(a: dict[str, str], b: dict[str, str]) -> bool:
    """Cheap same-router/same-texture check for local controls."""

    if broad_router_bucket(a) != broad_router_bucket(b):
        return False
    texture_a = max(align.safe_float(a.get("cand_texture")), align.safe_float(a.get("tube_mean_texture")))
    texture_b = max(align.safe_float(b.get("cand_texture")), align.safe_float(b.get("tube_mean_texture")))
    sky_a = max(align.safe_float(a.get("cand_sky_like")), align.safe_float(a.get("tube_mean_sky_like")))
    sky_b = max(align.safe_float(b.get("cand_sky_like")), align.safe_float(b.get("tube_mean_sky_like")))
    line_a = max(align.safe_float(a.get("cand_line_context")), align.safe_float(a.get("tube_mean_line_context")))
    line_b = max(align.safe_float(b.get("cand_line_context")), align.safe_float(b.get("tube_mean_line_context")))
    return (
        abs(texture_a - texture_b) <= 0.35
        and abs(sky_a - sky_b) <= 0.35
        and abs(line_a - line_b) <= 0.45
    )


def matched_control_offsets(
    row: dict[str, str],
    frame_rows: list[dict[str, str]],
    fallback_offsets: list[tuple[float, float]],
    max_controls: int,
) -> tuple[list[tuple[float, float]], int]:
    """Prefer nearby same-context candidate centers, then fill from annulus offsets."""

    anchor = align.center(row)
    offsets: list[tuple[float, float]] = []
    for other in frame_rows:
        if other is row:
            continue
        if other.get("track_id") and row.get("track_id") and other.get("track_id") == row.get("track_id"):
            continue
        dist = row_center_dist(row, other)
        if dist < 8.0 or dist > 48.0:
            continue
        if not context_similar(row, other):
            continue
        ox, oy = align.center(other)
        offsets.append((float(ox - anchor[0]), float(oy - anchor[1])))
        if len(offsets) >= max_controls:
            break
    matched_count = len(offsets)
    for offset in fallback_offsets:
        if len(offsets) >= max_controls:
            break
        offsets.append(offset)
    return offsets, matched_count


def finite_quality(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def gain_zscore(raw_gain: float, controls: list[float]) -> tuple[float, float, float]:
    if not controls:
        return 0.0, 0.0, 1.0
    arr = np.asarray([finite_quality(v) for v in controls], dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(0.25, 1.4826 * mad)
    return float((raw_gain - med) / sigma), med, sigma


def shrink_low_control_gain(norm_gain: float, matched_count: int, min_matched: int = 6) -> float:
    if matched_count >= min_matched:
        return norm_gain
    return norm_gain * max(0.0, matched_count / float(min_matched))


def path_quality(
    clip: str,
    row: dict[str, str],
    frame_cache: align.FrameCache,
    transforms: align.TransformCache,
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    radius: int,
    crop_size: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[dict[str, float], dict[str, float], list[float], int, int]:
    frame = align.safe_int(row.get("frame"), -1)
    anchor = align.center(row)
    anchor = (anchor[0] + offset[0], anchor[1] + offset[1])
    target_crops: list[np.ndarray] = []
    bg_crops: list[np.ndarray] = []
    path_bg_dist: list[float] = []
    used_frames = 0
    transform_failures = 0
    for target_frame in range(frame - radius, frame + 1):
        gray = frame_cache.gray(clip, target_frame)
        if gray is None:
            continue
        tpt = align.target_point(row, tube_rows, clip, frame, target_frame)
        tpt = (tpt[0] + offset[0], tpt[1] + offset[1])
        mat = transforms.transform(clip, frame, target_frame)
        bpt = align.project(mat, anchor)
        if np.allclose(mat, np.eye(3), atol=1e-6) and target_frame != frame:
            transform_failures += 1
        target_crops.append(align.extract_crop(gray, tpt, crop_size))
        bg_crops.append(align.extract_crop(gray, bpt, crop_size))
        path_bg_dist.append(float(math.hypot(tpt[0] - bpt[0], tpt[1] - bpt[1])))
        used_frames += 1
    return (
        align.stack_quality(target_crops, crop_size),
        align.stack_quality(bg_crops, crop_size),
        path_bg_dist,
        used_frames,
        transform_failures,
    )


def clutter_context(row: dict[str, str]) -> tuple[float, float, float]:
    line = max(
        align.safe_float(row.get("cand_line_context")),
        align.safe_float(row.get("tube_mean_line_context")),
    )
    support = max(
        align.safe_float(row.get("cand_attached_support")),
        align.safe_float(row.get("tube_mean_attached_support")),
    )
    density = align.safe_float(row.get("tube_log_cand_density"), math.log1p(align.safe_float(row.get("tube_mean_cand_density"))))
    return line, support, density


def score_row(
    row: dict[str, str],
    frame_rows: list[dict[str, str]],
    frame_cache: align.FrameCache,
    transforms: align.TransformCache,
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    window_radius: int,
    crop_size: int,
    control_offsets: list[tuple[float, float]],
) -> dict[str, Any]:
    clip = row.get("clip", "")
    tq, bq, path_dist, used, failures = path_quality(
        clip, row, frame_cache, transforms, tube_rows, window_radius, crop_size
    )
    raw_gain = tq["q"] - bq["q"]
    selected_offsets, matched_count = matched_control_offsets(
        row, frame_rows, control_offsets, max_controls=max(4, len(control_offsets))
    )
    control_gains: list[float] = []
    for offset in selected_offsets:
        ctq, cbq, _, _, _ = path_quality(
            clip, row, frame_cache, transforms, tube_rows, window_radius, crop_size, offset
        )
        control_gains.append(ctq["q"] - cbq["q"])
    norm_gain, control_median, control_sigma = gain_zscore(raw_gain, control_gains)
    norm_gain = shrink_low_control_gain(norm_gain, matched_count)
    line, support, density = clutter_context(row)
    bg_dist_mean = float(np.mean(path_dist)) if path_dist else 0.0
    bg_dist_max = float(np.max(path_dist)) if path_dist else 0.0
    bg_static = bq["q"] + max(0.0, 2.5 - bg_dist_mean) / 2.5 + 0.25 * line + 0.025 * support
    attached = 0.9 * line + 0.04 * support + 0.05 * density - 0.25 * max(0.0, norm_gain)
    target_llr = norm_gain + 0.08 * min(bg_dist_mean, 12.0) - 0.15 * max(0.0, bq["q"] - tq["q"])
    out: dict[str, Any] = dict(row)
    out.update(
        {
            "clba_target_q": round(float(tq["q"]), 6),
            "clba_bg_q": round(float(bq["q"]), 6),
            "clba_gain": round(float(raw_gain), 6),
            "clba_gain_norm": round(float(norm_gain), 6),
            "clba_control_median": round(float(control_median), 6),
            "clba_control_sigma": round(float(control_sigma), 6),
            "clba_control_count": len(control_gains),
            "clba_matched_control_count": matched_count,
            "clba_control_mode": "matched" if matched_count >= 6 else "fallback_shrunk",
            "clba_path_bg_dist_mean": round(bg_dist_mean, 6),
            "clba_path_bg_dist_max": round(bg_dist_max, 6),
            "clba_target_stack_dark_z": round(float(tq["stack_dark_z"]), 6),
            "clba_bg_stack_dark_z": round(float(bq["stack_dark_z"]), 6),
            "clba_target_anisotropy": round(float(tq["anisotropy"]), 6),
            "clba_bg_anisotropy": round(float(bq["anisotropy"]), 6),
            "clba_bg_static_likelihood": round(float(bg_static), 6),
            "clba_attached_likelihood": round(float(attached), 6),
            "clba_target_likelihood": round(float(target_llr), 6),
            "clba_used_frames": used,
            "clba_transform_failures": failures,
        }
    )
    return out


def main() -> None:
    args = parse_args()
    top_path = Path(args.top_tubes)
    frame_filter = None
    if args.frames_csv:
        frame_filter = load_frame_filter(Path(args.frames_csv), args.frames_clip or args.clip)
        if not frame_filter:
            raise SystemExit("frame filter is empty")
    rows = load_rows(top_path, args.clip, args.max_rank, args.frame_min, args.frame_max, frame_filter)
    if not rows:
        raise SystemExit("no top-tube rows loaded")
    clip = rows[0].get("clip", args.clip)
    if not clip:
        raise SystemExit("clip is required when rows do not contain a clip column")

    tube_rows = align.load_tube_rows([top_path])
    if args.clip and not any(k[0] == args.clip for k in tube_rows):
        # load_tube_rows infers the parent folder as clip for bare top_tubes paths;
        # add explicit clip aliases so target path lookup works for augmented rows.
        aliased: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
        for (_old_clip, tid), by_frame in tube_rows.items():
            aliased[(args.clip, tid)] = by_frame
        tube_rows.update(aliased)

    control_offsets = sample_control_offsets(args.control_radius, args.control_count)
    rows_by_frame: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_frame.setdefault(align.safe_int(row.get("frame"), -1), []).append(row)
    frame_cache = align.FrameCache(Path(args.video_dir), args.detector_scale)
    transforms = align.TransformCache(frame_cache, args.orb_features, args.min_matches)
    try:
        out_rows = [
            score_row(
                row,
                rows_by_frame.get(align.safe_int(row.get("frame"), -1), []),
                frame_cache,
                transforms,
                tube_rows,
                args.window_radius,
                args.crop_size,
                control_offsets,
            )
            for row in rows
        ]
    finally:
        frame_cache.close()

    out_csv = Path(args.out_csv)
    write_csv(out_csv, out_rows)
    meta = {
        "top_tubes": str(top_path),
        "out_csv": str(out_csv),
        "clip": clip,
        "rows": len(out_rows),
        "max_rank": args.max_rank,
        "window_radius": args.window_radius,
        "crop_size": args.crop_size,
        "detector_scale": args.detector_scale,
        "control_radius": args.control_radius,
        "control_count": args.control_count,
        "frame_min": args.frame_min,
        "frame_max": args.frame_max,
        "frames_csv": args.frames_csv,
        "frame_filter_count": 0 if frame_filter is None else len(frame_filter),
    }
    (out_csv.parent / "clba_augment_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
