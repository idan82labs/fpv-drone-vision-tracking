#!/usr/bin/env python3
"""Locally recenter exported top-tube candidates on high-resolution evidence.

This is an offline proposal-recovery harness. It does not change detector
runtime defaults. The intended experiment is narrow: when the current top-tube
export has many loose near-misses, can a high-resolution local search produce
strictly centered candidates that the selector/ranker could use later?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Peak:
    x_orig: float
    y_orig: float
    score: float
    radius_orig: int
    method: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True, help="Input top_tubes.csv.")
    p.add_argument("--video", required=True, help="Source video used by the detector.")
    p.add_argument("--out_csv", required=True, help="Merged original+recentered top-tubes CSV.")
    p.add_argument("--summary_json", default="", help="Optional run summary JSON.")
    p.add_argument("--clip", default="", help="Clip id to fill when input rows lack a clip column.")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--frame_min", type=int, default=-1)
    p.add_argument("--frame_max", type=int, default=-1)
    p.add_argument("--frames_csv", default="", help="Optional CSV limiting processing to listed frames.")
    p.add_argument("--frames_clip", default="", help="Clip filter for --frames_csv. Defaults to --clip.")
    p.add_argument("--detector_scale", type=float, default=0.5, help="detector_px = original_px * scale.")
    p.add_argument("--search_radius_det_px", type=float, default=16.0)
    p.add_argument("--radii_orig", default="2,3,4", help="Comma-separated compact-dark radii in original pixels.")
    p.add_argument("--texture_weight", type=float, default=0.010)
    p.add_argument("--box_size_det_px", type=float, default=3.0)
    p.add_argument("--peaks_per_candidate", type=int, default=1)
    p.add_argument("--grid_step_det_px", type=float, default=0.0, help="Optional scored halo-grid step. Disabled at 0.")
    p.add_argument("--grid_per_candidate", type=int, default=0, help="Top scored halo-grid offsets to add per candidate.")
    p.add_argument("--max_recenter_per_frame", type=int, default=100)
    p.add_argument("--max_output_per_frame", type=int, default=180)
    p.add_argument("--nms_det_px", type=float, default=2.5)
    p.add_argument("--recenter_score_weight", type=float, default=1.25)
    p.add_argument("--shift_penalty", type=float, default=0.025)
    p.add_argument("--keep_originals", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--router_include", default="", help="Optional comma-separated candidate router states to recenter.")
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def parse_ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def parse_set(text: str) -> set[str]:
    return {v.strip() for v in text.split(",") if v.strip()}


def load_frame_filter(path: Path, clip: str) -> set[int]:
    frames: set[int] = set()
    for row in read_csv(path):
        if clip and row.get("clip", clip) != clip:
            continue
        frame = safe_int(row.get("frame"), -1)
        if frame >= 0:
            frames.add(frame)
    return frames


def load_top_rows(
    path: Path,
    clip: str,
    max_rank: int,
    frame_min: int,
    frame_max: int,
    frame_filter: set[int] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(path):
        rank = safe_int(row.get("rank"), 999999)
        frame = safe_int(row.get("frame"), -1)
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
    rows.sort(key=lambda r: (safe_int(r.get("frame"), 0), safe_int(r.get("rank"), 999999)))
    return rows


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


def compact_dark_map(gray: np.ndarray, radius: int, texture_weight: float) -> np.ndarray:
    """Return a local dark-center/ring-normalized response map."""

    img = gray.astype(np.float32)
    r = max(1, int(radius))
    inner_k = 2 * r + 1
    outer_k = 6 * r + 3
    inner_n = float(inner_k * inner_k)
    outer_n = float(outer_k * outer_k)
    ring_n = max(1.0, outer_n - inner_n)

    inner_sum = cv2.boxFilter(img, cv2.CV_32F, (inner_k, inner_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    outer_sum = cv2.boxFilter(img, cv2.CV_32F, (outer_k, outer_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    inner_mean = inner_sum / inner_n
    ring_mean = (outer_sum - inner_sum) / ring_n

    outer_sq = cv2.boxFilter(img * img, cv2.CV_32F, (outer_k, outer_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    outer_mean = outer_sum / outer_n
    outer_var = np.maximum(0.0, outer_sq / outer_n - outer_mean * outer_mean)
    outer_std = np.sqrt(outer_var + 9.0)
    dark_z = (ring_mean - inner_mean) / outer_std

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    texture = cv2.boxFilter(grad, cv2.CV_32F, (outer_k, outer_k), normalize=True, borderType=cv2.BORDER_REFLECT101)

    score = dark_z - texture_weight * texture
    border = max(outer_k, 10)
    score[:border, :] = -999.0
    score[-border:, :] = -999.0
    score[:, :border] = -999.0
    score[:, -border:] = -999.0
    return score.astype(np.float32)


def detector_center(row: dict[str, str]) -> tuple[float, float]:
    return (
        safe_float(row.get("x")) + 0.5 * safe_float(row.get("w"), 3.0),
        safe_float(row.get("y")) + 0.5 * safe_float(row.get("h"), 3.0),
    )


def priority_score(row: dict[str, Any]) -> float:
    for key in ("recenter_combined_score", "verified_score", "score", "cand_score"):
        if key in row and row.get(key) not in (None, ""):
            return safe_float(row.get(key), 0.0)
    return 0.0


def find_local_peaks(
    score_maps: dict[int, np.ndarray],
    center_orig: tuple[float, float],
    search_radius_orig: float,
    peaks_per_candidate: int,
) -> list[Peak]:
    cx, cy = center_orig
    peaks: list[Peak] = []
    for radius, score in score_maps.items():
        h, w = score.shape[:2]
        x0 = max(0, int(math.floor(cx - search_radius_orig)))
        y0 = max(0, int(math.floor(cy - search_radius_orig)))
        x1 = min(w, int(math.ceil(cx + search_radius_orig + 1)))
        y1 = min(h, int(math.ceil(cy + search_radius_orig + 1)))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = score[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        local = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        ys, xs = np.nonzero((crop >= local - 1e-6) & np.isfinite(crop) & (crop > -100.0))
        if len(xs) == 0:
            continue
        vals = crop[ys, xs]
        order = np.argsort(vals)[::-1][: max(1, peaks_per_candidate)]
        for idx in order:
            peaks.append(
                Peak(
                    float(x0 + xs[idx]),
                    float(y0 + ys[idx]),
                    float(vals[idx]),
                    radius,
                    "compact_dark_ring",
                )
            )
    peaks.sort(key=lambda p: p.score, reverse=True)
    deduped: list[Peak] = []
    for peak in peaks:
        if any(math.hypot(peak.x_orig - prev.x_orig, peak.y_orig - prev.y_orig) <= 2.0 for prev in deduped):
            continue
        deduped.append(peak)
        if len(deduped) >= max(1, peaks_per_candidate):
            break
    return deduped


def best_score_at(score_maps: dict[int, np.ndarray], x_orig: float, y_orig: float) -> Peak | None:
    best: Peak | None = None
    for radius, score in score_maps.items():
        h, w = score.shape[:2]
        ix = int(round(x_orig))
        iy = int(round(y_orig))
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            continue
        peak = Peak(float(ix), float(iy), float(score[iy, ix]), radius, "surface_halo_grid")
        if best is None or peak.score > best.score:
            best = peak
    return best


def grid_halo_peaks(
    score_maps: dict[int, np.ndarray],
    center_orig: tuple[float, float],
    detector_scale: float,
    search_radius_det_px: float,
    grid_step_det_px: float,
    grid_per_candidate: int,
) -> list[Peak]:
    if grid_step_det_px <= 0 or grid_per_candidate <= 0:
        return []
    cx, cy = center_orig
    offsets_det = np.arange(-search_radius_det_px, search_radius_det_px + 1e-6, grid_step_det_px)
    peaks: list[Peak] = []
    for dy_det in offsets_det:
        for dx_det in offsets_det:
            if math.hypot(float(dx_det), float(dy_det)) > search_radius_det_px + 1e-6:
                continue
            x_orig = cx + float(dx_det) / detector_scale
            y_orig = cy + float(dy_det) / detector_scale
            peak = best_score_at(score_maps, x_orig, y_orig)
            if peak is not None and peak.score > -100.0:
                peaks.append(peak)
    peaks.sort(key=lambda p: p.score, reverse=True)
    deduped: list[Peak] = []
    for peak in peaks:
        if any(math.hypot(peak.x_orig - prev.x_orig, peak.y_orig - prev.y_orig) <= 2.0 for prev in deduped):
            continue
        deduped.append(peak)
        if len(deduped) >= grid_per_candidate:
            break
    return deduped


def recenter_row_to_peak(
    row: dict[str, str],
    peak: Peak,
    detector_scale: float,
    box_size_det_px: float,
    recenter_score_weight: float,
    shift_penalty: float,
) -> dict[str, Any]:
    if detector_scale <= 0:
        raise ValueError("detector_scale must be positive")
    dcx, dcy = detector_center(row)

    new_cx_det = peak.x_orig * detector_scale
    new_cy_det = peak.y_orig * detector_scale
    shift_det = float(math.hypot(new_cx_det - dcx, new_cy_det - dcy))
    base_score = priority_score(row)
    combined = base_score + recenter_score_weight * peak.score - shift_penalty * shift_det

    out: dict[str, Any] = dict(row)
    out["proposal_variant"] = "recenter_highres_dark_ring"
    out["track_id"] = f"{row.get('track_id', '')}:recenter:{row.get('rank', '')}"
    out["cand_source"] = "recenter_highres_dark_ring"
    out["x"] = f"{new_cx_det - 0.5 * box_size_det_px:.3f}"
    out["y"] = f"{new_cy_det - 0.5 * box_size_det_px:.3f}"
    out["w"] = f"{box_size_det_px:.3f}"
    out["h"] = f"{box_size_det_px:.3f}"
    out["recenter_parent_rank"] = row.get("rank", "")
    out["recenter_parent_x"] = row.get("x", "")
    out["recenter_parent_y"] = row.get("y", "")
    out["recenter_parent_w"] = row.get("w", "")
    out["recenter_parent_h"] = row.get("h", "")
    out["recenter_dx_det"] = f"{new_cx_det - dcx:.3f}"
    out["recenter_dy_det"] = f"{new_cy_det - dcy:.3f}"
    out["recenter_shift_det"] = f"{shift_det:.3f}"
    out["recenter_score"] = f"{peak.score:.6f}"
    out["recenter_radius_orig"] = str(peak.radius_orig)
    out["recenter_method"] = peak.method
    out["recenter_combined_score"] = f"{combined:.6f}"
    out["score"] = f"{combined:.6f}"
    out["verified_score"] = f"{combined:.6f}"
    out["selected"] = "0"
    out["eligible"] = out.get("eligible", "1") or "1"
    return out


def recenter_row(
    row: dict[str, str],
    score_maps: dict[int, np.ndarray],
    detector_scale: float,
    search_radius_det_px: float,
    box_size_det_px: float,
    recenter_score_weight: float,
    shift_penalty: float,
    peaks_per_candidate: int = 1,
    grid_step_det_px: float = 0.0,
    grid_per_candidate: int = 0,
) -> list[dict[str, Any]]:
    dcx, dcy = detector_center(row)
    center_orig = (dcx / detector_scale, dcy / detector_scale)
    peaks = find_local_peaks(score_maps, center_orig, search_radius_det_px / detector_scale, peaks_per_candidate)
    peaks.extend(
        grid_halo_peaks(
            score_maps,
            center_orig,
            detector_scale,
            search_radius_det_px,
            grid_step_det_px,
            grid_per_candidate,
        )
    )
    peaks.sort(key=lambda p: p.score, reverse=True)
    deduped: list[Peak] = []
    for peak in peaks:
        if any(math.hypot(peak.x_orig - prev.x_orig, peak.y_orig - prev.y_orig) <= 2.0 for prev in deduped):
            continue
        deduped.append(peak)
    return [
        recenter_row_to_peak(
            row,
            peak,
            detector_scale,
            box_size_det_px,
            recenter_score_weight,
            shift_penalty,
        )
        for peak in deduped
        if peak.score > -100.0
    ]


def center_dist(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax = safe_float(a.get("x")) + 0.5 * safe_float(a.get("w"), 3.0)
    ay = safe_float(a.get("y")) + 0.5 * safe_float(a.get("h"), 3.0)
    bx = safe_float(b.get("x")) + 0.5 * safe_float(b.get("w"), 3.0)
    by = safe_float(b.get("y")) + 0.5 * safe_float(b.get("h"), 3.0)
    return float(math.hypot(ax - bx, ay - by))


def dedupe_and_rank(
    frame_rows: list[dict[str, Any]],
    nms_det_px: float,
    max_output: int,
) -> list[dict[str, Any]]:
    ordered = sorted(frame_rows, key=lambda r: priority_score(r), reverse=True)
    kept: list[dict[str, Any]] = []
    for row in ordered:
        if any(center_dist(row, prev) <= nms_det_px for prev in kept):
            continue
        kept.append(dict(row))
        if len(kept) >= max_output:
            break
    for idx, row in enumerate(kept, start=1):
        row["rank"] = str(idx)
    return kept


def recenter_rows(
    rows: list[dict[str, str]],
    frames: dict[int, np.ndarray],
    radii_orig: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[safe_int(row.get("frame"), -1)].append(row)

    router_filter = parse_set(args.router_include)
    all_rows: list[dict[str, Any]] = []
    input_rows = 0
    recentered_rows = 0
    missing_frames = 0

    for frame, frame_rows in sorted(by_frame.items()):
        gray = frames.get(frame)
        if gray is None:
            missing_frames += 1
            continue
        score_maps = {r: compact_dark_map(gray, r, args.texture_weight) for r in radii_orig}
        merged: list[dict[str, Any]] = []
        if args.keep_originals:
            for row in frame_rows:
                original: dict[str, Any] = dict(row)
                original.setdefault("proposal_variant", "original")
                original.setdefault("recenter_combined_score", f"{priority_score(original):.6f}")
                merged.append(original)

        recentered_for_frame: list[dict[str, Any]] = []
        for row in frame_rows:
            input_rows += 1
            if router_filter and row.get("cand_router_state", "") not in router_filter:
                continue
            candidate_recentered_rows = recenter_row(
                row,
                score_maps,
                args.detector_scale,
                args.search_radius_det_px,
                args.box_size_det_px,
                args.recenter_score_weight,
                args.shift_penalty,
                args.peaks_per_candidate,
                args.grid_step_det_px,
                args.grid_per_candidate,
            )
            recentered_for_frame.extend(candidate_recentered_rows)
        recentered_for_frame.sort(key=lambda r: priority_score(r), reverse=True)
        recentered_for_frame = recentered_for_frame[: args.max_recenter_per_frame]
        recentered_rows += len(recentered_for_frame)
        merged.extend(recentered_for_frame)
        all_rows.extend(dedupe_and_rank(merged, args.nms_det_px, args.max_output_per_frame))

    summary = {
        "frames_seen": len(by_frame),
        "frames_loaded": len(frames),
        "missing_video_frames": missing_frames,
        "input_rows": input_rows,
        "output_rows": len(all_rows),
        "recentered_rows_before_nms": recentered_rows,
        "radii_orig": radii_orig,
        "detector_scale": args.detector_scale,
        "search_radius_det_px": args.search_radius_det_px,
        "peaks_per_candidate": args.peaks_per_candidate,
        "grid_step_det_px": args.grid_step_det_px,
        "grid_per_candidate": args.grid_per_candidate,
        "max_rank": args.max_rank,
        "keep_originals": args.keep_originals,
    }
    return all_rows, summary


def main() -> None:
    args = parse_args()
    frame_filter = None
    if args.frames_csv:
        frame_filter = load_frame_filter(Path(args.frames_csv), args.frames_clip or args.clip)
    rows = load_top_rows(
        Path(args.top_tubes),
        args.clip,
        args.max_rank,
        args.frame_min,
        args.frame_max,
        frame_filter,
    )
    frames = load_gray_frames(Path(args.video), {safe_int(r.get("frame"), -1) for r in rows})
    out_rows, summary = recenter_rows(rows, frames, parse_ints(args.radii_orig), args)
    write_csv(Path(args.out_csv), out_rows)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
