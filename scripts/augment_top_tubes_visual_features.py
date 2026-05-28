#!/usr/bin/env python3
"""Add lightweight crop/tube visual features to top_tubes.csv.

The existing TBD exports mostly scalar trajectory/proposal fields. This script
adds visual evidence that compares two explanations for each candidate:

1. target-aligned path: previous centers follow the tube velocity;
2. background-aligned path: previous centers follow the estimated homography.

It is intentionally offline and heavier than the onboard detector. The purpose
is to test whether visual tube evidence can rank true drone candidates over
cloud/terrain/skyline clutter before folding a smaller version into runtime.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import motion_detector_v2 as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--top_tubes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--offsets", default="-12,-9,-7,-5,-3,-2,-1")
    p.add_argument("--max_rank", type=int, default=220)
    p.add_argument("--max_corners", type=int, default=1200)
    p.add_argument("--quality", type=float, default=0.006)
    p.add_argument("--min_distance", type=int, default=7)
    p.add_argument("--ransac_px", type=float, default=2.2)
    p.add_argument("--model", choices=("partial_affine", "full_affine", "homography", "auto"), default="auto")
    return p.parse_args()


def parse_offsets(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def write_csv(path: Path, rows: list[dict[str, Any]], base_fields: list[str]) -> None:
    fields = list(base_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_gray_frames(video: Path, downscale: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = base.ensure_gray(frame)
        if abs(downscale - 1.0) > 1e-6:
            gray = cv2.resize(gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()
    if not frames:
        raise SystemExit("no frames read")
    return frames


def transform_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        max_corners=args.max_corners,
        quality=args.quality,
        min_distance=args.min_distance,
        ransac_px=args.ransac_px,
        model=args.model,
    )


def estimate_h_to_current(
    src: np.ndarray,
    dst: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray | None:
    targs = transform_args(args)
    g0, g1 = base.lk_tracks(src, dst, None, targs)
    if g0 is None or g1 is None:
        return None
    chosen = base.choose_model(src, dst, g0, g1, targs)
    if chosen is None:
        return None
    return chosen["h"].astype(np.float32)


def compact_dark_map(gray: np.ndarray, radius: int, texture_weight: float = 0.025) -> np.ndarray:
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
    return (dark_z - texture_weight * texture).astype(np.float32)


def sample_bilinear(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = x - x0
    dy = y - y0
    return float(
        (1 - dx) * (1 - dy) * img[y0, x0]
        + dx * (1 - dy) * img[y0, x0 + 1]
        + (1 - dx) * dy * img[y0 + 1, x0]
        + dx * dy * img[y0 + 1, x0 + 1]
    )


def patch_values(img: np.ndarray, cx: float, cy: float, radius: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = max(1, int(radius))
    x0 = max(0, int(round(cx)) - r)
    y0 = max(0, int(round(cy)) - r)
    x1 = min(w, int(round(cx)) + r + 1)
    y1 = min(h, int(round(cy)) + r + 1)
    patch = img[y0:y1, x0:x1]
    return patch.reshape(-1).astype(np.float32)


def z_residual_map(cur: np.ndarray, warped_stack: list[np.ndarray]) -> np.ndarray | None:
    if len(warped_stack) < 2:
        return None
    med = np.median(np.stack(warped_stack, axis=0), axis=0).astype(np.float32)
    residual_dark = med - cur.astype(np.float32)
    local_mean = cv2.GaussianBlur(residual_dark, (0, 0), 5.0)
    local_sq = cv2.GaussianBlur(residual_dark * residual_dark, (0, 0), 5.0)
    local_std = np.sqrt(np.maximum(4.0, local_sq - local_mean * local_mean))
    return ((residual_dark - local_mean) / local_std).astype(np.float32)


def main() -> None:
    args = parse_args()
    offsets = parse_offsets(args.offsets)
    rows_all = read_csv(Path(args.top_tubes))
    base_fields = list(rows_all[0].keys()) if rows_all else []
    rows = [r for r in rows_all if int(safe_float(r.get("rank"), 999999)) <= args.max_rank]
    rows_by_frame: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_frame.setdefault(int(safe_float(row.get("frame"))), []).append(row)

    frames = load_gray_frames(Path(args.video), args.downscale)
    h_img, w_img = frames[0].shape[:2]
    h_cache: dict[tuple[int, int], np.ndarray | None] = {}
    dark_cache: dict[tuple[int, int], np.ndarray] = {}
    temporal_cache: dict[int, np.ndarray | None] = {}

    def dark(frame_no: int, radius: int) -> np.ndarray:
        key = (frame_no, radius)
        if key not in dark_cache:
            dark_cache[key] = compact_dark_map(frames[frame_no], radius)
        return dark_cache[key]

    def h_to_current(src_frame: int, cur_frame: int) -> np.ndarray | None:
        key = (src_frame, cur_frame)
        if key not in h_cache:
            if src_frame < 0 or cur_frame < 0 or src_frame >= len(frames) or cur_frame >= len(frames):
                h_cache[key] = None
            else:
                h_cache[key] = estimate_h_to_current(frames[src_frame], frames[cur_frame], args)
        return h_cache[key]

    def temporal_z(frame_no: int) -> np.ndarray | None:
        if frame_no in temporal_cache:
            return temporal_cache[frame_no]
        cur = frames[frame_no]
        warped: list[np.ndarray] = []
        for off in offsets:
            src_frame = frame_no + off
            if src_frame < 0 or src_frame >= len(frames):
                continue
            h_mat = h_to_current(src_frame, frame_no)
            if h_mat is None:
                continue
            warped.append(
                cv2.warpPerspective(
                    frames[src_frame],
                    h_mat,
                    (w_img, h_img),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101,
                )
            )
        temporal_cache[frame_no] = z_residual_map(cur, warped)
        return temporal_cache[frame_no]

    augmented: list[dict[str, Any]] = []
    for n, frame_no in enumerate(sorted(rows_by_frame), start=1):
        if frame_no < 0 or frame_no >= len(frames):
            continue
        temp = temporal_z(frame_no)
        cur_gray = frames[frame_no]
        for row in rows_by_frame[frame_no]:
            rec: dict[str, Any] = dict(row)
            x = safe_float(row.get("x"))
            y = safe_float(row.get("y"))
            bw = max(1.0, safe_float(row.get("w"), 3.0))
            bh = max(1.0, safe_float(row.get("h"), 3.0))
            cx = x + 0.5 * bw
            cy = y + 0.5 * bh
            vx = safe_float(row.get("vx"))
            vy = safe_float(row.get("vy"))
            radius = max(1, int(round(0.65 * max(bw, bh))))

            cur_dark = dark(frame_no, radius)
            cur_dark_vals = patch_values(cur_dark, cx, cy, radius)
            rec["vis_cur_dark"] = round(float(np.mean(cur_dark_vals)) if cur_dark_vals.size else 0.0, 6)
            rec["vis_cur_dark_max"] = round(float(np.max(cur_dark_vals)) if cur_dark_vals.size else 0.0, 6)
            if temp is not None:
                temp_vals = patch_values(temp, cx, cy, radius)
                rec["vis_temp_z"] = round(float(np.mean(temp_vals)) if temp_vals.size else 0.0, 6)
                rec["vis_temp_z_max"] = round(float(np.max(temp_vals)) if temp_vals.size else 0.0, 6)
            else:
                rec["vis_temp_z"] = 0.0
                rec["vis_temp_z_max"] = 0.0

            tgt_dark_vals: list[float] = []
            bg_dark_vals: list[float] = []
            pair_vals: list[float] = []
            for off in offsets:
                src_frame = frame_no + off
                if src_frame < 0 or src_frame >= len(frames):
                    continue
                src_dark = dark(src_frame, radius)
                dt = -off
                tgt_x = cx - vx * dt
                tgt_y = cy - vy * dt
                tgt_dark_vals.append(sample_bilinear(src_dark, tgt_x, tgt_y))

                h_mat = h_to_current(src_frame, frame_no)
                if h_mat is not None:
                    try:
                        inv_h = np.linalg.inv(h_mat)
                        pt = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), inv_h.astype(np.float32))
                        bg_x, bg_y = float(pt[0, 0, 0]), float(pt[0, 0, 1])
                        bg_dark_vals.append(sample_bilinear(src_dark, bg_x, bg_y))
                        warped = cv2.warpPerspective(
                            frames[src_frame],
                            h_mat,
                            (w_img, h_img),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101,
                        )
                        old_bg_mean = float(np.mean(patch_values(warped.astype(np.float32), cx, cy, radius)))
                        new_mean = float(np.mean(patch_values(cur_gray.astype(np.float32), cx, cy, radius)))
                        pair_vals.append(old_bg_mean - new_mean)
                    except (np.linalg.LinAlgError, cv2.error):
                        pass

            if tgt_dark_vals:
                rec["vis_tgt_dark_mean"] = round(float(np.mean(tgt_dark_vals)), 6)
                rec["vis_tgt_dark_std"] = round(float(np.std(tgt_dark_vals)), 6)
                rec["vis_tgt_dark_max"] = round(float(np.max(tgt_dark_vals)), 6)
            else:
                rec["vis_tgt_dark_mean"] = 0.0
                rec["vis_tgt_dark_std"] = 0.0
                rec["vis_tgt_dark_max"] = 0.0
            if bg_dark_vals:
                rec["vis_bg_dark_mean"] = round(float(np.mean(bg_dark_vals)), 6)
                rec["vis_bg_dark_std"] = round(float(np.std(bg_dark_vals)), 6)
                rec["vis_bg_dark_max"] = round(float(np.max(bg_dark_vals)), 6)
            else:
                rec["vis_bg_dark_mean"] = 0.0
                rec["vis_bg_dark_std"] = 0.0
                rec["vis_bg_dark_max"] = 0.0
            rec["vis_align_dark_gain"] = round(float(rec["vis_tgt_dark_mean"] - rec["vis_bg_dark_mean"]), 6)
            rec["vis_align_dark_max_gain"] = round(float(rec["vis_tgt_dark_max"] - rec["vis_bg_dark_max"]), 6)
            rec["vis_pair_dark_mean"] = round(float(np.mean(pair_vals)) if pair_vals else 0.0, 6)
            rec["vis_pair_dark_max"] = round(float(np.max(pair_vals)) if pair_vals else 0.0, 6)
            rec["vis_n_offsets"] = len(tgt_dark_vals)
            augmented.append(rec)
        if n % 25 == 0:
            print(f"processed {n}/{len(rows_by_frame)} frames", flush=True)

    write_csv(Path(args.out), augmented, base_fields)
    print(args.out)


if __name__ == "__main__":
    main()
