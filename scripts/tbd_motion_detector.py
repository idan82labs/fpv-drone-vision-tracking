#!/usr/bin/env python3
"""
Candidate-level track-before-detect prototype.

This keeps the current cheap proposal generator from motion_detector_v2, then
selects one box using a short-window beam search over plausible trajectories.
It is deliberately candidate-based first: cheap, debuggable, and comparable to
the existing per-frame/track selector before attempting dense x-y-t Viterbi.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import motion_detector_v2 as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--output_dir", default="results_tbd")
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--save_every", type=int, default=30)
    p.add_argument("--write_video", action="store_true")

    p.add_argument("--model", choices=("partial_affine", "full_affine", "homography", "auto"), default="auto")
    p.add_argument("--max_corners", type=int, default=900)
    p.add_argument("--quality", type=float, default=0.008)
    p.add_argument("--min_distance", type=int, default=7)
    p.add_argument("--ransac_px", type=float, default=2.0)

    p.add_argument("--threshold_sigma", type=float, default=5.5)
    p.add_argument("--threshold_percentile", type=float, default=96.5)
    p.add_argument("--min_threshold", type=float, default=9.0)
    p.add_argument("--appearance", choices=("off", "dark", "bright", "both"), default="dark")
    p.add_argument("--appearance_sigma", type=float, default=7.0)
    p.add_argument("--appearance_percentile", type=float, default=99.5)
    p.add_argument("--appearance_blur", type=float, default=6.0)
    p.add_argument("--min_app_residual", type=float, default=1.5)
    p.add_argument("--strong_appearance", type=float, default=45.0)
    p.add_argument("--min_appearance_mean", type=float, default=0.0)
    p.add_argument("--max_appearance_isolation", type=float, default=999.0)

    p.add_argument("--min_area", type=int, default=3)
    p.add_argument("--max_area", type=int, default=650)
    p.add_argument("--min_fill", type=float, default=0.08)
    p.add_argument("--max_aspect", type=float, default=5.0)
    p.add_argument("--border_frac", type=float, default=0.025)
    p.add_argument("--temporal_radius", type=int, default=7)

    p.add_argument("--kinematic_gate", action="store_true")
    p.add_argument("--horizontal_fov_deg", type=float, default=120.0)
    p.add_argument("--max_relative_speed_mps", type=float, default=10.0)
    p.add_argument("--min_range_m", type=float, default=2.0)
    p.add_argument("--kinematic_slack_px", type=float, default=20.0)
    p.add_argument("--max_match_dist", type=float, default=48.0)

    p.add_argument("--beam_width", type=int, default=90)
    p.add_argument("--top_k_candidates", type=int, default=80)
    p.add_argument("--map_peaks", action="store_true")
    p.add_argument("--map_radii", default="2,3,5")
    p.add_argument("--map_top_k", type=int, default=120)
    p.add_argument("--map_score_floor", type=float, default=1.8)
    p.add_argument("--map_nms_px", type=int, default=5)
    p.add_argument("--map_dark_weight", type=float, default=1.15)
    p.add_argument("--map_residual_weight", type=float, default=0.65)
    p.add_argument("--map_line_weight", type=float, default=0.35)
    p.add_argument("--map_texture_weight", type=float, default=0.08)
    p.add_argument("--map_score_weight", type=float, default=0.9)
    p.add_argument("--micro_map_score_floor", type=float, default=0.95)
    p.add_argument("--micro_map_edge_score_floor", type=float, default=1.55)
    p.add_argument("--micro_map_max_radius", type=int, default=1)
    p.add_argument("--micro_map_max_line", type=float, default=0.35)
    p.add_argument("--micro_map_edge_max_line", type=float, default=0.65)
    p.add_argument("--micro_map_max_texture", type=float, default=0.95)
    p.add_argument("--native_micro_peaks", action="store_true")
    p.add_argument("--native_micro_radii", default="2,3")
    p.add_argument("--native_micro_top_k", type=int, default=40)
    p.add_argument("--native_micro_score_floor", type=float, default=1.25)
    p.add_argument("--native_micro_nms_px", type=int, default=9)
    p.add_argument("--native_micro_line_weight", type=float, default=0.25)
    p.add_argument("--native_micro_texture_weight", type=float, default=0.05)
    p.add_argument("--native_micro_score_weight", type=float, default=1.0)
    p.add_argument("--native_roi_score", action="store_true")
    p.add_argument("--native_roi_weight", type=float, default=0.35)
    p.add_argument("--native_roi_neutral", type=float, default=1.0)
    p.add_argument("--large_dark_peaks", action="store_true")
    p.add_argument("--large_dark_top_k", type=int, default=40)
    p.add_argument("--large_dark_score_floor", type=float, default=35.0)
    p.add_argument("--large_dark_nms_px", type=float, default=18.0)
    p.add_argument("--large_dark_box_full", type=float, default=28.0)
    p.add_argument("--large_dark_bg_sigma", type=float, default=5.0)
    p.add_argument("--large_dark_score_weight", type=float, default=0.08)
    p.add_argument("--hybrid_coast_proposals", action="store_true")
    p.add_argument("--hybrid_coast_top_k", type=int, default=18)
    p.add_argument("--hybrid_coast_max_misses", type=int, default=8)
    p.add_argument("--hybrid_coast_offsets", default="0:0,-4:0,4:0,0:-4,0:4,-7:0,7:0,0:-7,0:7")
    p.add_argument("--hybrid_coast_score_weight", type=float, default=0.18)
    p.add_argument("--scenario_balance", action="store_true")
    p.add_argument("--scenario_pool_factor", type=float, default=3.0)
    p.add_argument("--scenario_sky_top_k", type=int, default=24)
    p.add_argument("--scenario_surface_top_k", type=int, default=34)
    p.add_argument("--scenario_boundary_top_k", type=int, default=20)
    p.add_argument("--scenario_large_top_k", type=int, default=20)
    p.add_argument("--scenario_coast_top_k", type=int, default=18)
    p.add_argument("--temporal_stack_peaks", action="store_true")
    p.add_argument("--temporal_stack_offsets", default="-8,-5,-3,-2,-1")
    p.add_argument("--temporal_stack_radii", default="2,3,4,5,7")
    p.add_argument("--temporal_stack_top_k", type=int, default=120)
    p.add_argument("--temporal_stack_min_frames", type=int, default=2)
    p.add_argument("--temporal_stack_nms_px", type=float, default=5.5)
    p.add_argument("--temporal_stack_halo_bases", type=int, default=32)
    p.add_argument(
        "--temporal_stack_halo_offsets",
        default="0:0,-6:0,6:0,0:-6,0:6,-9:0,9:0,0:-9,0:9,-6:-6,-6:6,6:-6,6:6",
    )
    p.add_argument("--temporal_stack_halo_penalty", type=float, default=0.025)
    p.add_argument("--temporal_stack_score_weight", type=float, default=0.78)
    p.add_argument("--temporal_stack_native_weight", type=float, default=0.55)
    p.add_argument("--temporal_stack_clahe_weight", type=float, default=0.25)
    p.add_argument(
        "--temporal_stack_direct_warp",
        action="store_true",
        help="Estimate each temporal-stack history frame directly to the current frame instead of composing frame-to-frame warps.",
    )
    p.add_argument("--window", type=int, default=9)
    p.add_argument("--min_path_hits", type=int, default=3)
    p.add_argument("--max_misses", type=int, default=2)
    p.add_argument("--max_selected_misses", type=int, default=1)
    p.add_argument("--selected_score", type=float, default=6.0)
    p.add_argument("--birth_penalty", type=float, default=1.8)
    p.add_argument("--miss_penalty", type=float, default=1.1)
    p.add_argument("--obs_weight", type=float, default=0.95)
    p.add_argument("--pair_weight", type=float, default=0.85)
    p.add_argument("--speed_weight", type=float, default=0.16)
    p.add_argument("--accel_weight", type=float, default=0.12)
    p.add_argument("--static_penalty_weight", type=float, default=0.0)
    p.add_argument("--static_penalty_px", type=float, default=1.2)
    p.add_argument("--line_weight", type=float, default=0.35)
    p.add_argument("--support_penalty_weight", type=float, default=0.0)
    p.add_argument("--support_penalty_threshold", type=float, default=3.5)
    p.add_argument("--app_low_residual_penalty", type=float, default=0.0)
    p.add_argument("--app_low_residual_px", type=float, default=4.0)
    p.add_argument("--tube_verifier", choices=("off", "heuristic", "likelihood"), default="off")
    p.add_argument("--tube_verifier_weight", type=float, default=1.0)
    p.add_argument("--tube_verifier_floor", type=float, default=-999.0)
    p.add_argument("--density_penalty_weight", type=float, default=0.0)
    p.add_argument("--selection_margin", type=float, default=0.0)
    p.add_argument("--bg_motion_transition", action="store_true")
    p.add_argument("--bg_pair_geometry", action="store_true")
    p.add_argument("--local_pair_norm", action="store_true")
    p.add_argument("--pair_search_penalty", type=float, default=0.0)
    p.add_argument("--sky_bonus_weight", type=float, default=0.0)
    p.add_argument("--sky_rescue", action="store_true")
    p.add_argument("--sky_rescue_score", type=float, default=6.0)
    p.add_argument("--sky_rescue_min_sky", type=float, default=0.25)
    p.add_argument("--sky_rescue_max_line", type=float, default=0.45)
    p.add_argument("--sky_rescue_max_support", type=float, default=1.4)
    p.add_argument("--sky_rescue_min_hits", type=int, default=4)
    p.add_argument("--pair_radius", type=int, default=3)
    p.add_argument("--pair_min_step_px", type=float, default=1.0)
    p.add_argument("--top_k_debug", type=int, default=12)
    p.add_argument("--export_top_tubes", type=int, default=0)
    p.add_argument("--draw_debug", action="store_true")
    return p.parse_args()


@dataclass
class PathState:
    sid: int
    bbox: tuple[int, int, int, int]
    vx: float = 0.0
    vy: float = 0.0
    last_frame: int = 0
    misses: int = 0
    age: int = 1
    hits: int = 1
    last_candidate: base.Candidate | None = None
    contribs: list[float] = field(default_factory=list)
    hit_flags: list[bool] = field(default_factory=list)
    history: list[tuple[int, tuple[int, int, int, int], float, bool]] = field(default_factory=list)
    candidate_history: list[dict | None] = field(default_factory=list)
    pair_history: list[float] = field(default_factory=list)
    pair_raw_history: list[float] = field(default_factory=list)
    pair_bg_history: list[float] = field(default_factory=list)
    pair_bg_local_history: list[float] = field(default_factory=list)
    align_gain_history: list[float] = field(default_factory=list)
    bg_dist_history: list[float] = field(default_factory=list)
    cv_resid_history: list[float] = field(default_factory=list)
    bg_minus_cv_history: list[float] = field(default_factory=list)
    cand_density_history: list[float] = field(default_factory=list)
    speed_history: list[float] = field(default_factory=list)
    accel_history: list[float] = field(default_factory=list)

    def score(self) -> float:
        return float(sum(self.contribs))

    def hit_count(self) -> int:
        return int(sum(1 for h in self.hit_flags if h))

    def predict_bbox(self, w_img: int, h_img: int) -> tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return base.clip_bbox_float((x + self.vx, y + self.vy, w, h), w_img, h_img)


@dataclass
class TemporalStackFrame:
    gray_full: np.ndarray
    h_to_current_full: np.ndarray


def robust_sigma(img: np.ndarray) -> float:
    vals = img.reshape(-1).astype(np.float32)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(3.0, 1.4826 * mad)


def patch_mean(img: np.ndarray, cx: float, cy: float, radius: int) -> float:
    h, w = img.shape[:2]
    r = max(1, int(radius))
    x0 = max(0, int(round(cx)) - r)
    y0 = max(0, int(round(cy)) - r)
    x1 = min(w, int(round(cx)) + r + 1)
    y1 = min(h, int(round(cy)) + r + 1)
    patch = img[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else 0.0


def annulus_stats(
    img: np.ndarray,
    cx: float,
    cy: float,
    inner_radius: int,
    outer_radius: int,
) -> tuple[float, float]:
    h, w = img.shape[:2]
    outer = max(inner_radius + 1, int(outer_radius))
    inner = max(1, int(inner_radius))
    x0 = max(0, int(round(cx)) - outer)
    y0 = max(0, int(round(cy)) - outer)
    x1 = min(w, int(round(cx)) + outer + 1)
    y1 = min(h, int(round(cy)) + outer + 1)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0, 3.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    mask = (dist2 > inner * inner) & (dist2 <= outer * outer)
    vals = patch[mask]
    if vals.size < 4:
        vals = patch.reshape(-1)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, max(3.0, 1.4826 * mad)


def warp_bbox(
    bbox: tuple[int, int, int, int],
    h_mat: np.ndarray | None,
    w_img: int,
    h_img: int,
) -> tuple[int, int, int, int]:
    if h_mat is None:
        return bbox
    x, y, w, h = bbox
    cx, cy = base.bbox_center(bbox)
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    try:
        warped = cv2.perspectiveTransform(pt, h_mat).reshape(2)
    except cv2.error:
        return bbox
    if not np.isfinite(warped).all():
        return bbox
    return base.clip_bbox_float((float(warped[0]) - 0.5 * w, float(warped[1]) - 0.5 * h, w, h), w_img, h_img)


def dark_dipole_score(
    signed_diff: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    cur_bbox: tuple[int, int, int, int],
    sigma: float,
    radius: int,
    min_step_px: float,
    local_norm: bool = False,
    search_penalty: float = 0.0,
) -> float:
    pcx, pcy = base.bbox_center(prev_bbox)
    ccx, ccy = base.bbox_center(cur_bbox)
    step = math.hypot(ccx - pcx, ccy - pcy)
    if step < min_step_px:
        return 0.0
    old_mean = patch_mean(signed_diff, pcx, pcy, radius)
    new_mean = patch_mean(signed_diff, ccx, ccy, radius)
    if local_norm:
        old_bg, old_sigma = annulus_stats(signed_diff, pcx, pcy, radius + 1, max(radius + 3, 4 * radius))
        new_bg, new_sigma = annulus_stats(signed_diff, ccx, ccy, radius + 1, max(radius + 3, 4 * radius))
        old_positive = old_mean - old_bg
        new_negative = -(new_mean - new_bg)
        denom = math.sqrt(old_sigma * old_sigma + new_sigma * new_sigma)
    else:
        old_positive = old_mean
        new_negative = -new_mean
        denom = 2.0 * sigma
    z = (old_positive + new_negative) / max(1e-3, denom)
    if local_norm or search_penalty != 0.0:
        overlap = min(1.0, max(0.0, (step - min_step_px) / max(1.0, 1.5 * radius)))
        z = overlap * z - search_penalty
        return max(-4.0, min(4.0, z))
    return max(-3.0, min(3.0, z))


def alignment_gain_score(
    cand: base.Candidate,
    bg_pred_bbox: tuple[int, int, int, int],
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
) -> float:
    ccx, ccy = base.bbox_center(cand.bbox)
    bcx, bcy = base.bbox_center(bg_pred_bbox)
    radius = max(1, int(round(0.55 * max(cand.bbox[2], cand.bbox[3]))))
    cur_app = patch_mean(app_resp, ccx, ccy, radius)
    bg_app = patch_mean(app_resp, bcx, bcy, radius)
    cur_res = patch_mean(residual_blur, ccx, ccy, radius)
    bg_res = patch_mean(residual_blur, bcx, bcy, radius)
    app_scale = max(5.0, 0.5 * (abs(cur_app) + abs(bg_app)) + 3.0)
    res_scale = max(5.0, 0.5 * (abs(cur_res) + abs(bg_res)) + 3.0)
    app_gain = (cur_app - bg_app) / app_scale
    res_gain = (cur_res - bg_res) / res_scale
    return float(max(-3.0, min(3.0, 1.6 * app_gain + 0.9 * res_gain)))


def center_distance(
    a: tuple[int, int, int, int],
    b: tuple[float, float, float, float] | tuple[int, int, int, int],
) -> float:
    ax, ay = base.bbox_center(a)
    bx = float(b[0]) + 0.5 * float(b[2])
    by = float(b[1]) + 0.5 * float(b[3])
    return float(math.hypot(ax - bx, ay - by))


def parse_radii(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(max(1, int(round(float(part)))))
    return vals or [2, 3, 5]


def parse_int_offsets(text: str) -> list[int]:
    vals: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(round(float(part))))
    # The online tracker only has past frames. Ignore accidental future offsets.
    vals = sorted({v for v in vals if v < 0})
    return vals or [-5, -3, -2, -1]


def parse_xy_offsets(text: str) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        x_text, y_text = part.split(":")
        offsets.append((float(x_text), float(y_text)))
    return offsets or [(0.0, 0.0)]


def structure_maps(gray: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    gray_f = gray.astype(np.float32)
    k = max(3, 2 * radius + 1)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.blur(gx * gx, (k, k))
    jyy = cv2.blur(gy * gy, (k, k))
    jxy = cv2.blur(gx * gy, (k, k))
    trace = jxx + jyy
    disc = np.maximum(0.0, (jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy)
    lam1 = 0.5 * (trace + np.sqrt(disc))
    lam2 = 0.5 * (trace - np.sqrt(disc))
    anisotropy = (lam1 - lam2) / np.maximum(1e-3, lam1 + lam2)
    texture = np.sqrt(np.maximum(0.0, trace))
    tex_scale = max(5.0, float(np.percentile(texture, 85)))
    texture_z = np.clip(texture / tex_scale, 0.0, 2.0)
    line_context = anisotropy * texture_z
    return line_context.astype(np.float32), texture_z.astype(np.float32)


def likelihood_map_for_radius(
    cur_g: np.ndarray,
    residual_blur: np.ndarray,
    radius: int,
    args: argparse.Namespace,
    structure_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cur_g.astype(np.float32)
    residual = residual_blur.astype(np.float32)
    disk_k = max(3, 2 * radius + 1)
    outer_k = max(disk_k + 2, 2 * int(round(3.0 * radius)) + 1)
    if outer_k % 2 == 0:
        outer_k += 1

    disk_mean = cv2.blur(gray, (disk_k, disk_k))
    bg_mean = cv2.blur(gray, (outer_k, outer_k))
    local_dev = cv2.blur(np.abs(gray - bg_mean), (outer_k, outer_k))
    dark_z = (bg_mean - disk_mean) / (1.4826 * local_dev + 3.0)
    dark_z = np.maximum(0.0, dark_z)

    res_bg = cv2.blur(residual, (outer_k, outer_k))
    res_dev = cv2.blur(np.abs(residual - res_bg), (outer_k, outer_k))
    residual_z = (residual - res_bg) / (1.4826 * res_dev + 2.0)
    residual_z = np.maximum(0.0, residual_z)

    structure_radius = max(radius, 3)
    if structure_cache is not None and structure_radius in structure_cache:
        line_context, texture_z = structure_cache[structure_radius]
    else:
        line_context, texture_z = structure_maps(cur_g, radius=structure_radius)
        if structure_cache is not None:
            structure_cache[structure_radius] = (line_context, texture_z)
    score = (
        args.map_dark_weight * dark_z
        + args.map_residual_weight * residual_z
        - args.map_line_weight * line_context
        - args.map_texture_weight * texture_z
    )
    border = max(radius + 2, int(round(args.border_frac * min(cur_g.shape[:2]))))
    if border:
        score[:border, :] = -999.0
        score[-border:, :] = -999.0
        score[:, :border] = -999.0
        score[:, -border:] = -999.0
    return score.astype(np.float32), line_context, texture_z


def make_map_candidate(
    x: int,
    y: int,
    radius: int,
    map_score: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> base.Candidate | None:
    h_img, w_img = cur_g.shape[:2]
    side = max(3, int(round(2.4 * radius)))
    bx = int(round(x - 0.5 * side))
    by = int(round(y - 0.5 * side))
    bx = max(0, min(w_img - side, bx))
    by = max(0, min(h_img - side, by))
    bbox = (bx, by, side, side)
    mask = np.zeros_like(cur_g, dtype=np.uint8)
    mask[by : by + side, bx : bx + side] = 255
    cand = base.candidate_score("map", bbox, side * side, residual_blur, app_resp, mask, cur_g)
    cand.map_score = float(map_score)
    cand.score = 0.55 * cand.score + args.map_score_weight * float(map_score)
    if cand.score <= 0.1:
        return None
    return cand


def map_peak_candidates(
    cur_g: np.ndarray,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not args.map_peaks:
        return []
    peaks: list[tuple[float, int, int, int]] = []
    structure_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for radius in parse_radii(args.map_radii):
        score_map, line_context, texture_z = likelihood_map_for_radius(
            cur_g,
            residual_blur,
            radius,
            args,
            structure_cache,
        )
        nms = max(args.map_nms_px, 2 * radius + 1)
        if nms % 2 == 0:
            nms += 1
        dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
        if radius <= args.micro_map_max_radius:
            smooth_micro = (line_context <= args.micro_map_max_line) & (texture_z <= args.micro_map_max_texture)
            edge_micro = line_context <= args.micro_map_edge_max_line
            weak_smooth = (score_map >= args.micro_map_score_floor) & smooth_micro
            strong_edge = (score_map >= args.micro_map_edge_score_floor) & edge_micro
            mask = (weak_smooth | strong_edge) & (score_map >= dilated - 1e-6)
        else:
            mask = (score_map >= args.map_score_floor) & (score_map >= dilated - 1e-6)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        vals = score_map[ys, xs]
        order = np.argsort(vals)[::-1][: args.map_top_k]
        for idx in order:
            peaks.append((float(vals[idx]), int(xs[idx]), int(ys[idx]), radius))

    cands: list[base.Candidate] = []
    occupied: list[tuple[int, int, int, int]] = []
    for score, x, y, radius in sorted(peaks, reverse=True):
        if len(cands) >= args.map_top_k:
            break
        cand = make_map_candidate(x, y, radius, score, residual_blur, app_resp, cur_g, args)
        if cand is None:
            continue
        duplicate = False
        ccx, ccy = base.bbox_center(cand.bbox)
        for b in occupied:
            bcx, bcy = base.bbox_center(b)
            if math.hypot(ccx - bcx, ccy - bcy) <= max(3.0, 1.5 * radius):
                duplicate = True
                break
        if duplicate:
            continue
        occupied.append(cand.bbox)
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def native_micro_candidates(
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not args.native_micro_peaks or cur_full is None or abs(downscale - 1.0) < 1e-6:
        return []
    full_g = base.ensure_gray(cur_full)
    h_full, w_full = full_g.shape[:2]
    h_img, w_img = cur_g.shape[:2]
    gray = cv2.GaussianBlur(full_g, (3, 3), 0).astype(np.float32)
    peaks: list[tuple[float, int, int, int]] = []
    structure_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for radius in parse_radii(args.native_micro_radii):
        disk_k = max(3, 2 * radius + 1)
        outer_k = max(disk_k + 2, 2 * int(round(3.0 * radius)) + 1)
        if outer_k % 2 == 0:
            outer_k += 1
        disk_mean = cv2.blur(gray, (disk_k, disk_k))
        bg_mean = cv2.blur(gray, (outer_k, outer_k))
        local_dev = cv2.blur(np.abs(gray - bg_mean), (outer_k, outer_k))
        dark_z = np.maximum(0.0, (bg_mean - disk_mean) / (1.4826 * local_dev + 3.0))
        structure_radius = max(3, radius)
        if structure_radius not in structure_cache:
            structure_cache[structure_radius] = structure_maps(full_g, structure_radius)
        line_context, texture_z = structure_cache[structure_radius]
        score_map = (
            dark_z
            - args.native_micro_line_weight * line_context
            - args.native_micro_texture_weight * texture_z
        ).astype(np.float32)
        border = max(radius + 2, int(round(args.border_frac * min(w_full, h_full))))
        if border:
            score_map[:border, :] = -999.0
            score_map[-border:, :] = -999.0
            score_map[:, :border] = -999.0
            score_map[:, -border:] = -999.0
        nms = max(args.native_micro_nms_px, 2 * radius + 1)
        if nms % 2 == 0:
            nms += 1
        dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
        mask = (score_map >= args.native_micro_score_floor) & (score_map >= dilated - 1e-6)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        vals = score_map[ys, xs]
        order = np.argsort(vals)[::-1][: args.native_micro_top_k]
        for idx in order:
            peaks.append((float(vals[idx]), int(xs[idx]), int(ys[idx]), radius))

    cands: list[base.Candidate] = []
    occupied: list[tuple[int, int, int, int]] = []
    for score, x_full, y_full, radius in sorted(peaks, reverse=True):
        if len(cands) >= args.native_micro_top_k:
            break
        cx = x_full * downscale
        cy = y_full * downscale
        side = max(3, int(round(2.4 * radius * downscale)))
        bx = int(round(cx - 0.5 * side))
        by = int(round(cy - 0.5 * side))
        bx = max(0, min(w_img - side, bx))
        by = max(0, min(h_img - side, by))
        bbox = (bx, by, side, side)
        duplicate = False
        ccx, ccy = base.bbox_center(bbox)
        for kept in occupied:
            kcx, kcy = base.bbox_center(kept)
            if math.hypot(ccx - kcx, ccy - kcy) <= max(3.0, 1.5 * side):
                duplicate = True
                break
        if duplicate:
            continue
        mask = np.zeros_like(cur_g, dtype=np.uint8)
        mask[by : by + side, bx : bx + side] = 255
        cand = base.candidate_score("native_map", bbox, side * side, residual_blur, app_resp, mask, cur_g)
        cand.map_score = float(score)
        cand.score = 0.45 * cand.score + args.native_micro_score_weight * float(score)
        if cand.score <= 0.1:
            continue
        occupied.append(bbox)
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def large_dark_candidates(
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    """Full-frame dark-silhouette proposals for close, clearly visible drones.

    The existing micro and temporal-stack proposal paths are tuned for tiny
    3-10 px events and motion residuals. In the e271 gap, the drone is larger
    and often obvious as a dark local-contrast object, while residual ranking
    can suppress it. This source deliberately adds a small number of large
    dark-object proposals so the tube/ranker can choose them.
    """
    if not args.large_dark_peaks or cur_full is None or abs(downscale - 1.0) < 1e-6:
        return []
    full_g = base.ensure_gray(cur_full)
    h_full, w_full = full_g.shape[:2]
    h_img, w_img = cur_g.shape[:2]
    gray = cv2.GaussianBlur(full_g, (3, 3), 0).astype(np.float32)
    bg = cv2.GaussianBlur(full_g.astype(np.float32), (0, 0), args.large_dark_bg_sigma)
    score_map = (bg - gray).astype(np.float32)
    border = max(8, int(round(args.border_frac * min(w_full, h_full))))
    if border:
        score_map[:border, :] = -999.0
        score_map[-border:, :] = -999.0
        score_map[:, :border] = -999.0
        score_map[:, -border:] = -999.0

    nms = max(3, int(round(args.large_dark_nms_px)))
    if nms % 2 == 0:
        nms += 1
    dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
    mask = (score_map >= args.large_dark_score_floor) & (score_map >= dilated - 1e-6)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    vals = score_map[ys, xs]
    order = np.argsort(vals)[::-1][: args.large_dark_top_k]

    cands: list[base.Candidate] = []
    occupied: list[tuple[int, int, int, int]] = []
    side = max(3, int(round(args.large_dark_box_full * downscale)))
    for idx in order:
        score = float(vals[idx])
        x_full = int(xs[idx])
        y_full = int(ys[idx])
        cx = x_full * downscale
        cy = y_full * downscale
        bx = max(0, min(w_img - side, int(round(cx - 0.5 * side))))
        by = max(0, min(h_img - side, int(round(cy - 0.5 * side))))
        bbox = (bx, by, side, side)
        ccx, ccy = base.bbox_center(bbox)
        if any(math.hypot(ccx - kcx, ccy - kcy) <= max(3.0, side) for kcx, kcy in occupied):
            continue
        mask_img = np.zeros_like(cur_g, dtype=np.uint8)
        mask_img[by : by + side, bx : bx + side] = 255
        cand = base.candidate_score("large_dark", bbox, side * side, residual_blur, app_resp, mask_img, cur_g)
        cand.map_score = score
        cand.score = 0.45 * cand.score + args.large_dark_score_weight * score
        if cand.score <= 0.1:
            continue
        occupied.append((ccx, ccy))
        cands.append(cand)
        if len(cands) >= args.large_dark_top_k:
            break
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def compact_dark_map_native(gray: np.ndarray, radius: int, texture_weight: float = 0.025) -> np.ndarray:
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


def scale_h_to_full(h_down: np.ndarray, downscale: float) -> np.ndarray:
    if abs(downscale - 1.0) < 1e-6:
        return h_down.astype(np.float32)
    s = float(downscale)
    scale = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    inv_scale = np.array([[1.0 / s, 0.0, 0.0], [0.0, 1.0 / s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (inv_scale @ h_down.astype(np.float32) @ scale).astype(np.float32)


def estimate_direct_h_to_current_full(
    prev_full_gray: np.ndarray,
    cur_full_gray: np.ndarray,
    downscale: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, float]:
    if abs(downscale - 1.0) < 1e-6:
        prev_match = prev_full_gray
        cur_match = cur_full_gray
    else:
        prev_match = cv2.resize(prev_full_gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        cur_match = cv2.resize(cur_full_gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    g0, g1 = base.lk_tracks(prev_match, cur_match, None, args)
    if g0 is None or g1 is None:
        return None, 0.0
    chosen = base.choose_model(prev_match, cur_match, g0, g1, args)
    if chosen is None:
        return None, 0.0
    return scale_h_to_full(chosen["h"], downscale), float(chosen["inlier_ratio"])


def update_temporal_stack_history(
    history: list[TemporalStackFrame],
    prev_full_gray: np.ndarray,
    h_prev_to_cur_down: np.ndarray,
    downscale: float,
    max_age: int,
) -> list[TemporalStackFrame]:
    h_full = scale_h_to_full(h_prev_to_cur_down, downscale)
    updated = [
        TemporalStackFrame(entry.gray_full, (h_full @ entry.h_to_current_full).astype(np.float32))
        for entry in history
    ]
    updated.append(TemporalStackFrame(prev_full_gray, h_full))
    return updated[-max_age:]


def temporal_stack_residual_map(
    cur_full_gray: np.ndarray,
    history: list[TemporalStackFrame],
    offsets: list[int],
    min_frames: int,
    downscale: float,
    args: argparse.Namespace,
) -> np.ndarray | None:
    h_full, w_full = cur_full_gray.shape[:2]
    warped: list[np.ndarray] = []
    for off in offsets:
        age = abs(off)
        if age <= 0 or age > len(history):
            continue
        entry = history[-age]
        if args.temporal_stack_direct_warp:
            h_to_current, _inlier = estimate_direct_h_to_current_full(
                entry.gray_full,
                cur_full_gray,
                downscale,
                args,
            )
            if h_to_current is None:
                continue
        else:
            h_to_current = entry.h_to_current_full
        warped.append(
            cv2.warpPerspective(
                entry.gray_full,
                h_to_current,
                (w_full, h_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        )
    if len(warped) < min_frames:
        return None
    med = np.median(np.stack(warped, axis=0), axis=0).astype(np.float32)
    residual_dark = med - cur_full_gray.astype(np.float32)
    local_mean = cv2.GaussianBlur(residual_dark, (0, 0), 5.0)
    local_sq = cv2.GaussianBlur(residual_dark * residual_dark, (0, 0), 5.0)
    local_std = np.sqrt(np.maximum(4.0, local_sq - local_mean * local_mean))
    return ((residual_dark - local_mean) / local_std).astype(np.float32)


def local_maxima_peaks(score_map: np.ndarray, radius: int, top_k: int) -> list[tuple[float, float, float, int]]:
    nms = max(3, int(round(2 * radius + 1)))
    if nms % 2 == 0:
        nms += 1
    dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
    mask = (score_map >= dilated - 1e-6) & np.isfinite(score_map) & (score_map > -100.0)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    vals = score_map[ys, xs]
    order = np.argsort(vals)[::-1][:top_k]
    return [(float(vals[i]), float(xs[i]), float(ys[i]), radius) for i in order]


def dedupe_full_peaks(
    peaks: list[tuple[float, float, float, int]],
    nms_px: float,
    max_n: int,
) -> list[tuple[float, float, float, int]]:
    out: list[tuple[float, float, float, int]] = []
    for score, x, y, radius in sorted(peaks, key=lambda p: p[0], reverse=True):
        if len(out) >= max_n:
            break
        if any(math.hypot(x - ox, y - oy) <= nms_px for _os, ox, oy, _or in out):
            continue
        out.append((score, x, y, radius))
    return out


def temporal_stack_candidates(
    cur_full: np.ndarray | None,
    downscale: float,
    history: list[TemporalStackFrame],
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not args.temporal_stack_peaks or cur_full is None:
        return []
    full_g = base.ensure_gray(cur_full)
    temp_map = temporal_stack_residual_map(
        full_g,
        history,
        parse_int_offsets(args.temporal_stack_offsets),
        args.temporal_stack_min_frames,
        downscale,
        args,
    )
    if temp_map is None:
        return []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(full_g)
    base_peaks: list[tuple[float, float, float, int]] = []
    for radius in parse_radii(args.temporal_stack_radii):
        native_map = compact_dark_map_native(full_g, radius)
        clahe_map = compact_dark_map_native(clahe, radius, texture_weight=0.015)
        score_map = (
            temp_map
            + args.temporal_stack_native_weight * native_map
            + args.temporal_stack_clahe_weight * clahe_map
        )
        base_peaks.extend(local_maxima_peaks(score_map, radius, args.temporal_stack_top_k))

    base_peaks = dedupe_full_peaks(base_peaks, args.temporal_stack_nms_px, args.temporal_stack_top_k)
    halo_offsets = parse_xy_offsets(args.temporal_stack_halo_offsets)
    halo_peaks: list[tuple[float, float, float, int]] = []
    for score, x, y, radius in base_peaks[: args.temporal_stack_halo_bases]:
        for dx, dy in halo_offsets:
            shifted_score = score - args.temporal_stack_halo_penalty * math.hypot(dx, dy)
            halo_peaks.append((shifted_score, x + dx, y + dy, radius))
    peaks = dedupe_full_peaks(
        halo_peaks,
        max(2.5, 0.45 * args.temporal_stack_nms_px),
        args.temporal_stack_top_k,
    )

    h_img, w_img = cur_g.shape[:2]
    cands: list[base.Candidate] = []
    for score, x_full, y_full, radius in peaks:
        side_full = max(3, int(round(2 * radius + 1)))
        side = max(3, int(round(side_full * downscale)))
        cx = x_full * downscale
        cy = y_full * downscale
        bx = max(0, min(w_img - side, int(round(cx - 0.5 * side))))
        by = max(0, min(h_img - side, int(round(cy - 0.5 * side))))
        bbox = (bx, by, side, side)
        mask = np.zeros_like(cur_g, dtype=np.uint8)
        mask[by : by + side, bx : bx + side] = 255
        cand = base.candidate_score("temporal_stack", bbox, side * side, residual_blur, app_resp, mask, cur_g)
        cand.map_score = float(score)
        cand.score = 0.35 * cand.score + args.temporal_stack_score_weight * float(score)
        if cand.score <= 0.1:
            continue
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def candidate_duplicate(cand: base.Candidate, kept: list[base.Candidate]) -> bool:
    ccx, ccy = base.bbox_center(cand.bbox)
    for other in kept:
        ocx, ocy = base.bbox_center(other.bbox)
        if math.hypot(ccx - ocx, ccy - ocy) <= 2.5 or base.bbox_iou(cand.bbox, other.bbox) > 0.25:
            return True
    return False


def dedupe_candidates(cands: list[base.Candidate], max_n: int) -> list[base.Candidate]:
    deduped: list[base.Candidate] = []
    for cand in sorted(cands, key=lambda c: c.score, reverse=True):
        if not candidate_duplicate(cand, deduped):
            deduped.append(cand)
        if len(deduped) >= max_n:
            break
    return deduped


def hybrid_coast_candidates(
    states: list[PathState],
    tbd: BeamTBD,
    frame_h: np.ndarray | None,
    w_img: int,
    h_img: int,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    """Use current beam states as cheap predicted proposals, not decisions."""
    if not args.hybrid_coast_proposals or not states:
        return []
    offsets = parse_xy_offsets(args.hybrid_coast_offsets)
    cands: list[base.Candidate] = []
    occupied: list[base.Candidate] = []
    ranked_states = sorted(states, key=lambda st: tbd.verified_score(st), reverse=True)
    for st in ranked_states[: args.hybrid_coast_top_k]:
        if st.misses > args.hybrid_coast_max_misses:
            continue
        base_bbox = warp_bbox(st.bbox, frame_h, w_img, h_img)
        bx, by, bw, bh = base_bbox
        for dx, dy in offsets:
            bbox = base.clip_bbox_float((bx + st.vx + dx, by + st.vy + dy, bw, bh), w_img, h_img)
            mask = np.zeros_like(cur_g, dtype=np.uint8)
            x, y, w, h = bbox
            mask[y : y + h, x : x + w] = 255
            cand = base.candidate_score("hybrid_coast", bbox, max(1, w * h), residual_blur, app_resp, mask, cur_g)
            state_score = max(0.0, min(12.0, tbd.verified_score(st)))
            cand.map_score = float(state_score)
            cand.score = 0.70 * cand.score + args.hybrid_coast_score_weight * state_score
            if cand.score <= 0.1 or candidate_duplicate(cand, occupied):
                continue
            occupied.append(cand)
            cands.append(cand)
            if len(cands) >= args.hybrid_coast_top_k:
                return sorted(cands, key=lambda c: c.score, reverse=True)
    return sorted(cands, key=lambda c: c.score, reverse=True)


def candidate_scenario(cand: base.Candidate) -> str:
    if cand.source == "hybrid_coast":
        return "coast"
    if cand.source == "large_dark" or max(cand.bbox[2], cand.bbox[3]) >= 10:
        return "large"
    sky = float(getattr(cand, "sky_like", 0.0))
    texture = float(getattr(cand, "texture", 0.0))
    line = float(getattr(cand, "line_context", 0.0))
    if sky >= 0.25 and texture < 45.0 and line < 0.55:
        return "sky"
    if sky < 0.10 or texture >= 45.0:
        return "surface"
    return "boundary"


def scenario_balanced_candidates(cands: list[base.Candidate], args: argparse.Namespace) -> list[base.Candidate]:
    if not args.scenario_balance:
        return dedupe_candidates(cands, args.top_k_candidates)
    quotas = {
        "sky": max(0, args.scenario_sky_top_k),
        "surface": max(0, args.scenario_surface_top_k),
        "boundary": max(0, args.scenario_boundary_top_k),
        "large": max(0, args.scenario_large_top_k),
        "coast": max(0, args.scenario_coast_top_k),
    }
    kept: list[base.Candidate] = []
    by_bucket: dict[str, list[base.Candidate]] = {k: [] for k in quotas}
    for cand in sorted(cands, key=lambda c: c.score, reverse=True):
        by_bucket.setdefault(candidate_scenario(cand), []).append(cand)
    for bucket, quota in quotas.items():
        for cand in by_bucket.get(bucket, [])[:quota]:
            if len(kept) >= args.top_k_candidates:
                return kept
            if not candidate_duplicate(cand, kept):
                kept.append(cand)
    for cand in sorted(cands, key=lambda c: c.score, reverse=True):
        if len(kept) >= args.top_k_candidates:
            break
        if not candidate_duplicate(cand, kept):
            kept.append(cand)
    return kept


def candidate_obs(c: base.Candidate, args: argparse.Namespace) -> float:
    # Existing score is the main high-recall objectness cue. Add small explicit
    # penalties for line/texture contexts that repeatedly produce false boxes.
    line = getattr(c, "line_context", 0.0)
    obs = args.obs_weight * c.score - args.line_weight * line
    if args.support_penalty_weight > 0:
        excess_support = max(0.0, getattr(c, "attached_support", 0.0) - args.support_penalty_threshold)
        obs -= args.support_penalty_weight * excess_support
    if args.native_roi_score:
        native_excess = max(0.0, getattr(c, "native_dark_score", 0.0) - args.native_roi_neutral)
        obs += args.native_roi_weight * native_excess
    if (
        args.app_low_residual_penalty > 0
        and c.source == "appearance"
        and getattr(c, "map_score", 0.0) <= 0.0
        and c.mean_residual < args.app_low_residual_px
    ):
        deficit = (args.app_low_residual_px - c.mean_residual) / max(1e-3, args.app_low_residual_px)
        obs -= args.app_low_residual_penalty * deficit
    return obs


def attached_support_score(
    cur_g: np.ndarray,
    gx_abs: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """Score dark/edge structure continuing below a candidate, e.g. pole tops."""
    x, y, w, h = bbox
    h_img, w_img = cur_g.shape[:2]
    cx = int(round(x + 0.5 * w))
    half_col = max(2, int(round(0.5 * max(3, w + 2))))
    y0 = min(h_img, y + h + 1)
    y1 = min(h_img, y0 + max(24, 5 * max(w, h)))
    if y1 <= y0 + 4:
        return 0.0

    x0 = max(0, cx - half_col)
    x1 = min(w_img, cx + half_col + 1)
    below = cur_g[y0:y1, x0:x1].astype(np.float32)
    if below.size == 0:
        return 0.0

    side_pad = max(4, 2 * half_col)
    sx0 = max(0, x0 - side_pad)
    sx1 = min(w_img, x1 + side_pad)
    context = cur_g[y0:y1, sx0:sx1].astype(np.float32)
    mask = np.ones(context.shape, dtype=bool)
    mask[:, x0 - sx0 : x1 - sx0] = False
    side = context[mask]
    dark_column = max(0.0, float(np.median(side) - np.median(below))) if side.size else 0.0

    edge_column = float(np.mean(gx_abs[y0:y1, x0:x1]))
    return float(dark_column + 0.05 * edge_column)


def assign_attached_support(cands: list[base.Candidate], cur_g: np.ndarray) -> None:
    gx_abs = np.abs(cv2.Sobel(cur_g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    for cand in cands:
        cand.attached_support = attached_support_score(cur_g, gx_abs, cand.bbox)


def native_dark_score_for_bbox(
    full_g: np.ndarray,
    bbox: tuple[int, int, int, int],
    inv_downscale: float,
) -> float:
    x, y, w, h = bbox
    h_full, w_full = full_g.shape[:2]
    cx = int(round((x + 0.5 * w) * inv_downscale))
    cy = int(round((y + 0.5 * h) * inv_downscale))
    radius = max(1, int(round(0.55 * max(w, h) * inv_downscale)))
    outer = max(radius + 2, int(round(3.0 * radius)))
    x0 = max(0, cx - outer)
    y0 = max(0, cy - outer)
    x1 = min(w_full, cx + outer + 1)
    y1 = min(h_full, cy + outer + 1)
    patch = full_g[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    inner_mask = dist2 <= radius * radius
    ring_mask = (dist2 > radius * radius) & (dist2 <= outer * outer)
    if not np.any(inner_mask) or not np.any(ring_mask):
        return 0.0
    inner = patch[inner_mask]
    ring = patch[ring_mask]
    mad = float(np.median(np.abs(ring - np.median(ring))))
    return float(max(0.0, (np.median(ring) - np.mean(inner)) / (1.4826 * mad + 3.0)))


def assign_native_roi_scores(
    cands: list[base.Candidate],
    cur_full: np.ndarray | None,
    downscale: float,
) -> None:
    if cur_full is None or abs(downscale - 1.0) < 1e-6 or downscale <= 0:
        return
    full_g = base.ensure_gray(cur_full)
    inv_downscale = 1.0 / downscale
    for cand in cands:
        cand.native_dark_score = native_dark_score_for_bbox(full_g, cand.bbox, inv_downscale)


def sky_like_score(
    cur_g: np.ndarray,
    grad_mag: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    x, y, w, h = bbox
    h_img, w_img = cur_g.shape[:2]
    cx = int(round(x + 0.5 * w))
    cy = int(round(y + 0.5 * h))
    inner = max(2, int(round(0.8 * max(w, h))))
    outer = max(inner + 4, int(round(4.0 * max(w, h))))
    x0 = max(0, cx - outer)
    x1 = min(w_img, cx + outer + 1)
    y0 = max(0, cy - outer)
    y1 = min(h_img, cy + outer + 1)
    patch = cur_g[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    ring = (dist2 >= inner * inner) & (dist2 <= outer * outer)
    if not np.any(ring):
        ring = np.ones(patch.shape, dtype=bool)
    vals = patch[ring]
    mean_bg = float(np.mean(vals))
    std_bg = float(np.std(vals))
    grad_bg = float(np.mean(grad_mag[y0:y1, x0:x1][ring]))

    bright = np.clip((mean_bg - 95.0) / 75.0, 0.0, 1.0)
    smooth = np.clip((38.0 - std_bg) / 32.0, 0.0, 1.0)
    low_grad = np.clip((38.0 - grad_bg) / 38.0, 0.0, 1.0)
    return float(bright * smooth * low_grad)


def assign_sky_context(cands: list[base.Candidate], cur_g: np.ndarray) -> None:
    gray_f = cur_g.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    for cand in cands:
        cand.sky_like = sky_like_score(cur_g, grad_mag, cand.bbox)


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def tube_features(st: PathState) -> dict[str, float]:
    hist = st.candidate_history[-9:]
    cands = [c for c in hist if c is not None]
    n = max(1, len(hist))
    hits = len(cands)
    source_vals = [c.get("source", "") for c in cands]
    map_scores = [float(c.get("map_score", 0.0)) for c in cands]
    line_vals = [float(c.get("line_context", 0.0)) for c in cands]
    support_vals = [float(c.get("attached_support", 0.0)) for c in cands]
    native_vals = [float(c.get("native_dark_score", 0.0)) for c in cands]
    sky_vals = [float(c.get("sky_like", 0.0)) for c in cands]
    texture_vals = [float(c.get("texture", 0.0)) for c in cands]
    residual_vals = [float(c.get("mean_residual", 0.0)) for c in cands]
    appearance_vals = [float(c.get("mean_appearance", 0.0)) for c in cands]
    score_vals = [float(c.get("score", 0.0)) for c in cands]
    area_vals = [float(c.get("area", 0.0)) for c in cands]
    tiny_vals = [
        1.0
        for c in cands
        if c.get("bbox")
        and float(c["bbox"][2]) <= 4.0
        and float(c["bbox"][3]) <= 4.0
    ]
    pair_vals = [float(v) for v in st.pair_history[-9:]]
    pair_raw_vals = [float(v) for v in st.pair_raw_history[-9:]]
    pair_bg_vals = [float(v) for v in st.pair_bg_history[-9:]]
    pair_bg_local_vals = [float(v) for v in st.pair_bg_local_history[-9:]]
    align_vals = [float(v) for v in st.align_gain_history[-9:]]
    bg_dist_vals = [float(v) for v in st.bg_dist_history[-9:]]
    cv_resid_vals = [float(v) for v in st.cv_resid_history[-9:]]
    bg_minus_cv_vals = [float(v) for v in st.bg_minus_cv_history[-9:]]
    density_vals = [float(v) for v in st.cand_density_history[-9:]]
    speed_vals = [float(v) for v in st.speed_history[-9:]]
    accel_vals = [float(v) for v in st.accel_history[-9:]]
    app_only = [
        1.0
        for c in cands
        if c.get("source") == "appearance" and float(c.get("map_score", 0.0)) <= 0.0
    ]
    return {
        "tube_hits": float(hits),
        "tube_len": float(n),
        "hit_rate": hits / n,
        "miss_rate": 1.0 - hits / n,
        "mean_score": _mean(score_vals),
        "score_std": float(np.std(score_vals)) if len(score_vals) > 1 else 0.0,
        "mean_map_score": _mean(map_scores),
        "map_hit_rate": sum(1 for v in map_scores if v > 0.0) / max(1, hits),
        "appearance_only_rate": len(app_only) / max(1, hits),
        "tiny_rate": len(tiny_vals) / max(1, hits),
        "mean_line_context": _mean(line_vals),
        "max_line_context": max(line_vals) if line_vals else 0.0,
        "mean_attached_support": _mean(support_vals),
        "max_attached_support": max(support_vals) if support_vals else 0.0,
        "mean_native_dark_score": _mean(native_vals),
        "max_native_dark_score": max(native_vals) if native_vals else 0.0,
        "mean_sky_like": _mean(sky_vals),
        "max_sky_like": max(sky_vals) if sky_vals else 0.0,
        "sky_hit_rate": sum(1 for v in sky_vals if v >= 0.2) / max(1, hits),
        "mean_texture": _mean(texture_vals),
        "mean_residual": _mean(residual_vals),
        "mean_appearance": _mean(appearance_vals),
        "mean_area": _mean(area_vals),
        "mean_pair_score": _mean(pair_vals),
        "positive_pair_rate": sum(1 for v in pair_vals if v > 0.2) / max(1, len(pair_vals)),
        "mean_pair_raw": _mean(pair_raw_vals),
        "positive_pair_raw_rate": sum(1 for v in pair_raw_vals if v > 0.2) / max(1, len(pair_raw_vals)),
        "mean_pair_bg": _mean(pair_bg_vals),
        "positive_pair_bg_rate": sum(1 for v in pair_bg_vals if v > 0.2) / max(1, len(pair_bg_vals)),
        "mean_pair_bg_local": _mean(pair_bg_local_vals),
        "positive_pair_bg_local_rate": sum(1 for v in pair_bg_local_vals if v > 0.2) / max(1, len(pair_bg_local_vals)),
        "mean_align_gain": _mean(align_vals),
        "positive_align_rate": sum(1 for v in align_vals if v > 0.15) / max(1, len(align_vals)),
        "mean_bg_dist": _mean(bg_dist_vals),
        "mean_cv_resid": _mean(cv_resid_vals),
        "mean_bg_minus_cv": _mean(bg_minus_cv_vals),
        "mean_cand_density": _mean(density_vals),
        "log_cand_density": math.log1p(_mean(density_vals)),
        "mean_speed": _mean(speed_vals),
        "max_speed": max(speed_vals) if speed_vals else 0.0,
        "mean_accel": _mean(accel_vals),
        "max_accel": max(accel_vals) if accel_vals else 0.0,
    }


def heuristic_tube_verifier_score(features: dict[str, float]) -> float:
    # A lightweight hand-fit verifier until there are enough labeled tubes to
    # train logistic/GBM weights. Positive terms reward tiny map-backed tubes;
    # negative terms target the reviewed hard negatives: app-only clutter,
    # attached poles/stems, line context, and unstable high-texture blobs.
    score = 0.0
    score += 1.6 * features["map_hit_rate"]
    score += 0.45 * min(3.0, features["mean_map_score"])
    score += 0.85 * features["tiny_rate"]
    score += 0.55 * features["positive_pair_rate"]
    score += 0.25 * max(-2.0, min(2.0, features["mean_pair_score"]))
    score += 0.35 * features["hit_rate"]
    score += 0.18 * min(3.0, features["mean_speed"])
    score += 0.35 * max(0.0, features.get("mean_native_dark_score", 0.0) - 1.0)
    score += 0.20 * max(0.0, features.get("max_native_dark_score", 0.0) - 1.4)
    if (
        features["positive_pair_rate"] >= 0.45
        and features["mean_pair_score"] >= 0.8
        and features["mean_residual"] >= 20.0
        and features["mean_appearance"] >= 25.0
    ):
        score += 2.0

    score -= 2.4 * features["appearance_only_rate"]
    score -= 0.9 * max(0.0, features["mean_line_context"] - 0.35)
    score -= 0.65 * max(0.0, features["mean_attached_support"] - 2.4)
    score -= 0.45 * max(0.0, features["max_attached_support"] - 5.5)
    score -= 0.35 * max(0.0, (features["mean_texture"] - 85.0) / 35.0)
    if features["appearance_only_rate"] > 0.7 and features["mean_speed"] < 0.45:
        score -= 1.4
    if features["appearance_only_rate"] > 0.7 and features["mean_residual"] < 2.0:
        score -= 1.0
    if (
        features["map_hit_rate"] > 0.8
        and features["appearance_only_rate"] < 0.1
        and features["positive_pair_rate"] < 0.1
        and features["mean_residual"] < 2.5
        and features["mean_appearance"] < 20.0
    ):
        score -= 14.0
    return float(score)


def _logsumexp(vals: list[float]) -> float:
    if not vals:
        return 0.0
    m = max(vals)
    return float(m + math.log(sum(math.exp(v - m) for v in vals)))


def likelihood_tube_verifier_score(features: dict[str, float]) -> float:
    """Approximate tube log-likelihood ratio against explicit clutter modes.

    This is still hand-weighted, but it changes the structure from additive
    one-off penalties to "target tube must beat the best clutter explanation".
    The score scale is kept near the existing heuristic verifier so old
    selected_score values remain interpretable.
    """
    pair = max(-2.5, min(3.0, features["mean_pair_score"]))
    align = max(-2.5, min(3.0, features.get("mean_align_gain", 0.0)))
    low_pair = 1.0 - features["positive_pair_rate"]
    low_align = 1.0 - features.get("positive_align_rate", 0.0)
    low_residual = max(0.0, (5.0 - features["mean_residual"]) / 5.0)
    low_appearance = max(0.0, (20.0 - features["mean_appearance"]) / 20.0)
    low_speed = max(0.0, (0.75 - features["mean_speed"]) / 0.75)
    high_density = min(3.0, features.get("log_cand_density", 0.0))
    residual_strength = min(3.0, features["mean_residual"] / 18.0)
    appearance_strength = min(3.0, features["mean_appearance"] / 30.0)

    target = (
        0.95 * features["hit_rate"]
        + 0.95 * features["map_hit_rate"]
        + 0.35 * min(3.0, features["mean_map_score"])
        + 1.15 * features["positive_pair_rate"]
        + 0.55 * pair
        + 1.25 * features.get("positive_align_rate", 0.0)
        + 0.75 * align
        + 0.35 * residual_strength
        + 0.30 * appearance_strength
        + 0.42 * max(0.0, features.get("mean_native_dark_score", 0.0) - 1.0)
        + 0.30 * features["tiny_rate"]
    )

    static_hotspot = (
        -0.15
        + 1.40 * features["map_hit_rate"]
        + 1.10 * low_pair
        + 1.20 * low_align
        + 1.20 * low_residual
        + 0.85 * low_appearance
        + 0.65 * low_speed
    )
    line_attached = (
        -0.25
        + 2.00 * max(0.0, features["mean_line_context"] - 0.28)
        + 0.55 * max(0.0, features["max_line_context"] - 0.55)
        + 0.45 * max(0.0, features["mean_attached_support"] - 1.8)
        + 0.30 * max(0.0, features["max_attached_support"] - 4.5)
    )
    parallax_edge = (
        -0.55
        + 0.55 * max(0.0, (features["mean_texture"] - 45.0) / 35.0)
        + 0.55 * max(0.0, features["mean_line_context"] - 0.25)
        + 0.35 * residual_strength
        + 0.55 * low_pair
        + 0.35 * high_density
    )
    appearance_only_blob = (
        -0.10
        + 2.00 * features["appearance_only_rate"]
        + 0.75 * low_pair
        + 0.85 * low_align
        + 0.70 * low_residual
        + 0.20 * appearance_strength
    )
    boundary_artifact = (
        -0.45
        + 0.80 * features.get("max_sky_like", 0.0)
        + 0.65 * max(0.0, features["mean_line_context"] - 0.18)
        + 0.50 * low_pair
        + 0.40 * low_align
    )
    noise = (
        -0.35
        + 1.30 * features["miss_rate"]
        + 0.55 * max(0.0, 3.0 - features["tube_hits"])
        + 0.35 * high_density
        + 0.35 * max(0.0, features["score_std"] - 8.0) / 8.0
    )
    null_score = _logsumexp([
        static_hotspot,
        line_attached,
        parallax_edge,
        appearance_only_blob,
        boundary_artifact,
        noise,
    ])

    search_penalty = 0.18 * high_density + 0.12 * features["miss_rate"]
    score = 2.6 + target - 0.78 * null_score - search_penalty

    if (
        features["map_hit_rate"] > 0.8
        and features["positive_pair_rate"] < 0.1
        and features.get("positive_align_rate", 0.0) < 0.1
        and features["mean_residual"] < 2.5
        and features["mean_appearance"] < 20.0
    ):
        score -= 8.0
    if (
        features["positive_pair_rate"] >= 0.45
        and features["mean_pair_score"] >= 0.8
        and features.get("positive_align_rate", 0.0) >= 0.25
        and features["mean_residual"] >= 16.0
    ):
        score += 1.4
    return float(score)


def tube_verifier_score(features: dict[str, float], mode: str) -> float:
    if mode == "likelihood":
        return likelihood_tube_verifier_score(features)
    return heuristic_tube_verifier_score(features)


class BeamTBD:
    def __init__(self, args: argparse.Namespace, px_per_frame: float):
        self.args = args
        self.px_per_frame = px_per_frame
        self.states: list[PathState] = []
        self.next_sid = 1

    def verified_score(self, st: PathState) -> float:
        if self.args.tube_verifier == "off":
            return st.score()
        features = tube_features(st)
        tube_score = tube_verifier_score(features, self.args.tube_verifier)
        sky_bonus = self.args.sky_bonus_weight * (
            features.get("max_sky_like", 0.0) + 0.5 * features.get("sky_hit_rate", 0.0)
        )
        density_penalty = self.args.density_penalty_weight * features.get("log_cand_density", 0.0)
        return st.score() + self.args.tube_verifier_weight * tube_score + sky_bonus - density_penalty

    def _sky_rescue_best(self) -> PathState | None:
        if not self.args.sky_rescue:
            return None
        rescued: list[tuple[float, PathState]] = []
        for st in self.states:
            if st.misses > self.args.max_selected_misses:
                continue
            if st.hit_count() < self.args.sky_rescue_min_hits:
                continue
            features = tube_features(st)
            if features["max_sky_like"] < self.args.sky_rescue_min_sky:
                continue
            if features["mean_line_context"] > self.args.sky_rescue_max_line:
                continue
            if features["mean_attached_support"] > self.args.sky_rescue_max_support:
                continue
            if features["appearance_only_rate"] > 0.35:
                continue
            score = self.verified_score(st)
            if score >= self.args.sky_rescue_score:
                rescued.append((score, st))
        if not rescued:
            return None
        return max(rescued, key=lambda x: x[0])[1]

    def _trim(self, contribs: list[float], hits: list[bool]) -> tuple[list[float], list[bool]]:
        n = max(1, self.args.window)
        return contribs[-n:], hits[-n:]

    def _transition_cost(
        self,
        st: PathState,
        cand: base.Candidate,
        frame_no: int,
        prev_ref_bbox: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float, float]:
        dt = max(1, frame_no - st.last_frame)
        pcx, pcy = base.bbox_center(prev_ref_bbox)
        ccx, ccy = base.bbox_center(cand.bbox)
        vx = (ccx - pcx) / dt
        vy = (ccy - pcy) / dt
        speed = math.hypot(vx, vy)
        allowed = self.px_per_frame * dt
        excess = max(0.0, speed - allowed)
        accel = math.hypot(vx - st.vx, vy - st.vy)
        cost = self.args.speed_weight * excess * excess + self.args.accel_weight * accel * accel
        if self.args.static_penalty_weight > 0:
            static_deficit = max(0.0, self.args.static_penalty_px - speed)
            cost += self.args.static_penalty_weight * static_deficit * static_deficit
        return cost, vx, vy, speed, accel

    def update(
        self,
        frame_no: int,
        cands: list[base.Candidate],
        signed_diff: np.ndarray,
        signed_sigma: float,
        w_img: int,
        h_img: int,
        frame_h: np.ndarray | None,
        residual_blur: np.ndarray,
        app_resp: np.ndarray,
    ) -> list[PathState]:
        cands = sorted(cands, key=lambda c: c.score, reverse=True)[: self.args.top_k_candidates]
        new_states: list[PathState] = []
        cand_density = float(len(cands))

        for cand in cands:
            obs = candidate_obs(cand, self.args)
            birth_contrib = obs - self.args.birth_penalty
            birth_contribs, birth_hits = self._trim([birth_contrib], [True])
            birth = PathState(
                self.next_sid,
                cand.bbox,
                last_frame=frame_no,
                misses=0,
                age=1,
                hits=1,
                last_candidate=cand,
                contribs=birth_contribs,
                hit_flags=birth_hits,
                history=[(frame_no, cand.bbox, birth_contrib, True)],
                candidate_history=[cand.to_json()],
                pair_history=[0.0],
                pair_raw_history=[0.0],
                pair_bg_history=[0.0],
                pair_bg_local_history=[0.0],
                align_gain_history=[0.0],
                bg_dist_history=[0.0],
                cv_resid_history=[0.0],
                bg_minus_cv_history=[0.0],
                cand_density_history=[cand_density],
                speed_history=[0.0],
                accel_history=[0.0],
            )
            self.next_sid += 1
            best_state = birth
            best_score = birth.score()

            for st in self.states:
                dt = max(1, frame_no - st.last_frame)
                if dt > self.args.max_misses + 1:
                    continue
                transition_ref_bbox = (
                    warp_bbox(st.bbox, frame_h, w_img, h_img)
                    if self.args.bg_motion_transition
                    else st.bbox
                )
                pair_ref_bbox = (
                    warp_bbox(st.bbox, frame_h, w_img, h_img)
                    if self.args.bg_motion_transition or self.args.bg_pair_geometry
                    else st.bbox
                )
                bg_ref_bbox = warp_bbox(st.bbox, frame_h, w_img, h_img)
                dt_f = max(1, frame_no - st.last_frame)
                x0, y0, bw, bh = st.bbox
                cv_pred = (x0 + st.vx * dt_f, y0 + st.vy * dt_f, bw, bh)
                bg_dist = center_distance(cand.bbox, bg_ref_bbox)
                cv_resid = center_distance(cand.bbox, cv_pred)
                bg_minus_cv = bg_dist - cv_resid
                cost, vx, vy, speed, accel = self._transition_cost(st, cand, frame_no, transition_ref_bbox)
                if cost > 30.0:
                    continue
                pair_raw = dark_dipole_score(
                    signed_diff,
                    st.bbox,
                    cand.bbox,
                    signed_sigma,
                    self.args.pair_radius,
                    self.args.pair_min_step_px,
                )
                pair_bg = dark_dipole_score(
                    signed_diff,
                    bg_ref_bbox,
                    cand.bbox,
                    signed_sigma,
                    self.args.pair_radius,
                    self.args.pair_min_step_px,
                )
                pair_bg_local = dark_dipole_score(
                    signed_diff,
                    bg_ref_bbox,
                    cand.bbox,
                    signed_sigma,
                    self.args.pair_radius,
                    self.args.pair_min_step_px,
                    True,
                )
                pair = dark_dipole_score(
                    signed_diff,
                    pair_ref_bbox,
                    cand.bbox,
                    signed_sigma,
                    self.args.pair_radius,
                    self.args.pair_min_step_px,
                    self.args.local_pair_norm,
                    self.args.pair_search_penalty,
                )
                align_gain = alignment_gain_score(cand, bg_ref_bbox, residual_blur, app_resp)
                contrib = obs + self.args.pair_weight * pair - cost
                contribs, hits = self._trim(st.contribs + [contrib], st.hit_flags + [True])
                ns = PathState(
                    st.sid,
                    cand.bbox,
                    vx=0.65 * st.vx + 0.35 * vx,
                    vy=0.65 * st.vy + 0.35 * vy,
                    last_frame=frame_no,
                    misses=0,
                    age=st.age + 1,
                    hits=st.hits + 1,
                    last_candidate=cand,
                    contribs=contribs,
                    hit_flags=hits,
                    history=st.history[-self.args.window :] + [(frame_no, cand.bbox, contrib, True)],
                    candidate_history=(st.candidate_history + [cand.to_json()])[-self.args.window :],
                    pair_history=(st.pair_history + [pair])[-self.args.window :],
                    pair_raw_history=(st.pair_raw_history + [pair_raw])[-self.args.window :],
                    pair_bg_history=(st.pair_bg_history + [pair_bg])[-self.args.window :],
                    pair_bg_local_history=(st.pair_bg_local_history + [pair_bg_local])[-self.args.window :],
                    align_gain_history=(st.align_gain_history + [align_gain])[-self.args.window :],
                    bg_dist_history=(st.bg_dist_history + [bg_dist])[-self.args.window :],
                    cv_resid_history=(st.cv_resid_history + [cv_resid])[-self.args.window :],
                    bg_minus_cv_history=(st.bg_minus_cv_history + [bg_minus_cv])[-self.args.window :],
                    cand_density_history=(st.cand_density_history + [cand_density])[-self.args.window :],
                    speed_history=(st.speed_history + [speed])[-self.args.window :],
                    accel_history=(st.accel_history + [accel])[-self.args.window :],
                )
                if ns.score() > best_score:
                    best_score = ns.score()
                    best_state = ns
            new_states.append(best_state)

        for st in self.states[: self.args.beam_width]:
            if st.misses >= self.args.max_misses:
                continue
            if self.args.bg_motion_transition:
                bg_pred = warp_bbox(st.bbox, frame_h, w_img, h_img)
                bx, by, bw, bh = bg_pred
                pred = base.clip_bbox_float((bx + st.vx, by + st.vy, bw, bh), w_img, h_img)
            else:
                pred = st.predict_bbox(w_img, h_img)
            contribs, hits = self._trim(st.contribs + [-self.args.miss_penalty], st.hit_flags + [False])
            missed = PathState(
                st.sid,
                pred,
                vx=st.vx,
                vy=st.vy,
                last_frame=frame_no,
                misses=st.misses + 1,
                age=st.age + 1,
                hits=st.hits,
                last_candidate=st.last_candidate,
                contribs=contribs,
                hit_flags=hits,
                history=st.history[-self.args.window :] + [(frame_no, pred, -self.args.miss_penalty, False)],
                candidate_history=(st.candidate_history + [None])[-self.args.window :],
                pair_history=(st.pair_history + [0.0])[-self.args.window :],
                pair_raw_history=(st.pair_raw_history + [0.0])[-self.args.window :],
                pair_bg_history=(st.pair_bg_history + [0.0])[-self.args.window :],
                pair_bg_local_history=(st.pair_bg_local_history + [0.0])[-self.args.window :],
                align_gain_history=(st.align_gain_history + [0.0])[-self.args.window :],
                bg_dist_history=(st.bg_dist_history + [0.0])[-self.args.window :],
                cv_resid_history=(st.cv_resid_history + [0.0])[-self.args.window :],
                bg_minus_cv_history=(st.bg_minus_cv_history + [0.0])[-self.args.window :],
                cand_density_history=(st.cand_density_history + [cand_density])[-self.args.window :],
                speed_history=(st.speed_history + [math.hypot(st.vx, st.vy)])[-self.args.window :],
                accel_history=(st.accel_history + [0.0])[-self.args.window :],
            )
            new_states.append(missed)

        # Merge duplicate-ish states by keeping the stronger state near each center.
        deduped: list[PathState] = []
        for st in sorted(new_states, key=lambda s: s.score(), reverse=True):
            scx, scy = base.bbox_center(st.bbox)
            duplicate = False
            for kept in deduped:
                kcx, kcy = base.bbox_center(kept.bbox)
                if math.hypot(scx - kcx, scy - kcy) < 3.0 and base.bbox_iou(st.bbox, kept.bbox) > 0.1:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(st)
            if len(deduped) >= self.args.beam_width:
                break
        self.states = deduped
        return self.states

    def best(self) -> PathState | None:
        eligible = [
            st
            for st in self.states
            if st.misses <= self.args.max_selected_misses and st.hit_count() >= self.args.min_path_hits
        ]
        if not eligible:
            return None
        if self.args.tube_verifier == "off":
            best = max(eligible, key=lambda st: st.score())
            if best.score() >= self.args.selected_score:
                return best
            return self._sky_rescue_best()

        scored: list[tuple[float, PathState]] = []
        for st in eligible:
            tube_score = tube_verifier_score(tube_features(st), self.args.tube_verifier)
            if tube_score < self.args.tube_verifier_floor:
                continue
            scored.append((self.verified_score(st), st))
        if not scored:
            return self._sky_rescue_best()
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        if self.args.selection_margin > 0.0 and len(scored) > 1:
            if best_score - scored[1][0] < self.args.selection_margin:
                return self._sky_rescue_best()
        if best_score >= self.args.selected_score:
            return best
        return self._sky_rescue_best()


def draw_overlay(
    frame: np.ndarray,
    frame_no: int,
    model_name: str,
    inlier_ratio: float,
    threshold: float,
    cands: list[base.Candidate],
    states: list[PathState],
    selected: PathState | None,
    args: argparse.Namespace,
) -> np.ndarray:
    ov = frame.copy()
    if args.draw_debug:
        for cand in cands[: args.top_k_debug]:
            x, y, w, h = cand.bbox
            cv2.rectangle(ov, (x, y), (x + w, y + h), (80, 170, 80), 1)
        for st in states[:20]:
            if st.misses:
                continue
            x, y, w, h = st.bbox
            cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 180, 255), 1)
    if selected is not None:
        x, y, w, h = selected.bbox
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            ov,
            f"TBD T{selected.sid} {selected.score():.2f}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    text = f"f{frame_no:05d} {model_name} in={inlier_ratio:.2f} thr={threshold:.1f} cand={len(cands)} states={len(states)}"
    cv2.putText(ov, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ov, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return ov


def tube_state_payload(
    frame_no: int,
    rank: int,
    st: PathState,
    tbd: BeamTBD,
    args: argparse.Namespace,
    selected: PathState | None,
    competitor_margin: float | None = None,
) -> tuple[dict, dict]:
    features = tube_features(st)
    tube_score = tube_verifier_score(features, args.tube_verifier) if args.tube_verifier != "off" else 0.0
    verified = tbd.verified_score(st)
    raw = st.score()
    eligible = st.misses <= args.max_selected_misses and st.hit_count() >= args.min_path_hits
    passes_floor = (
        raw >= args.selected_score
        if args.tube_verifier == "off"
        else tube_score >= args.tube_verifier_floor and verified >= args.selected_score
    )
    is_selected = (
        selected is not None
        and st.sid == selected.sid
        and st.bbox == selected.bbox
        and st.last_frame == selected.last_frame
    )
    cand_json = st.last_candidate.to_json() if st.last_candidate is not None else None
    payload = {
        "rank": rank,
        "track_id": st.sid,
        "bbox": list(st.bbox),
        "score": round(raw, 3),
        "verified_score": round(verified, 3),
        "tube_verifier_score": round(tube_score, 3),
        "eligible": eligible,
        "passes_floor": passes_floor,
        "selected": is_selected,
        "hits": st.hit_count(),
        "misses": st.misses,
        "vx": round(st.vx, 3),
        "vy": round(st.vy, 3),
        "competitor_margin": round(float(competitor_margin), 3) if competitor_margin is not None else None,
        "candidate": cand_json,
        "tube_features": {k: round(float(v), 3) for k, v in features.items()},
    }
    row = {
        "frame": frame_no,
        "rank": rank,
        "track_id": st.sid,
        "x": st.bbox[0],
        "y": st.bbox[1],
        "w": st.bbox[2],
        "h": st.bbox[3],
        "score": round(raw, 6),
        "verified_score": round(verified, 6),
        "tube_verifier_score": round(tube_score, 6),
        "eligible": int(eligible),
        "passes_floor": int(passes_floor),
        "selected": int(is_selected),
        "hits": st.hit_count(),
        "misses": st.misses,
        "vx": round(st.vx, 6),
        "vy": round(st.vy, 6),
        "competitor_margin": round(float(competitor_margin), 6) if competitor_margin is not None else "",
    }
    if cand_json is not None:
        for key, value in cand_json.items():
            if isinstance(value, (int, float, str)):
                row[f"cand_{key}"] = value
    for key, value in features.items():
        row[f"tube_{key}"] = round(float(value), 6)
    return payload, row


def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit("no frames")
    prev_full = prev.copy()
    if args.downscale != 1.0:
        prev = cv2.resize(prev, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
    prev_g = base.ensure_gray(prev)
    h_img, w_img = prev_g.shape[:2]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = fps_src if fps_src > 1 else 30.0
        video_writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps_out, (w_img, h_img))

    px_per_frame = base.kinematic_px_per_frame(w_img, fps_src, args)
    tbd = BeamTBD(args, px_per_frame)
    prev_mask: np.ndarray | None = None
    report: list[dict] = []
    selected_rows: list[list] = []
    selected_feature_rows: list[dict] = []
    top_tube_rows: list[dict] = []
    model_counts: dict[str, int] = {}
    temporal_history: list[TemporalStackFrame] = []
    temporal_max_age = max(abs(v) for v in parse_int_offsets(args.temporal_stack_offsets))
    fno = 1

    while True:
        if args.max_frames is not None and fno >= args.max_frames:
            break
        ok, cur = cap.read()
        if not ok:
            break
        cur_full = cur
        if args.downscale != 1.0:
            cur = cv2.resize(cur, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
        cur_g = base.ensure_gray(cur)
        t0 = time.perf_counter()

        feature_mask = base.make_feature_mask(prev_g.shape[:2], [])
        g0, g1 = base.lk_tracks(prev_g, cur_g, feature_mask, args)
        if g0 is None:
            prev_g = cur_g
            prev_full = cur_full.copy()
            temporal_history = []
            fno += 1
            continue

        chosen = base.choose_model(prev_g, cur_g, g0, g1, args)
        if chosen is None:
            prev_g = cur_g
            prev_full = cur_full.copy()
            temporal_history = []
            fno += 1
            continue
        model_counts[chosen["name"]] = model_counts.get(chosen["name"], 0) + 1
        temporal_history = update_temporal_stack_history(
            temporal_history,
            base.ensure_gray(prev_full),
            chosen["h"],
            args.downscale,
            temporal_max_age,
        )

        warped = base.warp_prev(prev_g, chosen["h"], w_img, h_img)
        signed_diff = cur_g.astype(np.float32) - warped.astype(np.float32)
        signed_sigma = robust_sigma(signed_diff)
        residual = cv2.absdiff(warped, cur_g)
        residual_blur = cv2.GaussianBlur(residual, (3, 3), 0)
        threshold = base.robust_threshold(residual_blur, args.threshold_sigma, args.threshold_percentile, args.min_threshold)
        mask = (residual_blur >= threshold).astype(np.uint8) * 255
        app_resp, app_mask, app_threshold = base.appearance_response(cur_g, args)

        border = int(round(args.border_frac * min(w_img, h_img)))
        if border:
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0

        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

        if prev_mask is not None:
            dilate_k = 2 * args.temporal_radius + 1
            prev_dilated = cv2.dilate(prev_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
            confirmed_mask = cv2.bitwise_and(mask, prev_dilated)
            motion_mask = cv2.bitwise_or(confirmed_mask, cv2.erode(mask, k3))
        else:
            motion_mask = mask

        motion_cands = base.extract_candidates("motion", motion_mask, residual_blur, app_resp, cur_g, args)
        app_cands: list[base.Candidate] = []
        if args.appearance != "off":
            app_cands = base.extract_candidates("appearance", app_mask, residual_blur, app_resp, cur_g, args)
        map_cands = map_peak_candidates(cur_g, residual_blur, app_resp, args)
        native_cands = native_micro_candidates(
            cur_full,
            args.downscale,
            residual_blur,
            app_resp,
            cur_g,
            args,
        )
        large_dark_cands = large_dark_candidates(
            cur_full,
            args.downscale,
            residual_blur,
            app_resp,
            cur_g,
            args,
        )
        stack_cands = temporal_stack_candidates(
            cur_full,
            args.downscale,
            temporal_history,
            residual_blur,
            app_resp,
            cur_g,
            args,
        )
        hybrid_coast_cands = hybrid_coast_candidates(
            tbd.states,
            tbd,
            chosen["h"],
            w_img,
            h_img,
            residual_blur,
            app_resp,
            cur_g,
            args,
        )
        raw_cands = motion_cands + app_cands + map_cands + native_cands + large_dark_cands + stack_cands + hybrid_coast_cands
        if args.scenario_balance:
            pool_n = max(args.top_k_candidates, int(round(args.top_k_candidates * args.scenario_pool_factor)))
            cands = dedupe_candidates(raw_cands, pool_n)
        else:
            cands = dedupe_candidates(raw_cands, args.top_k_candidates)
        assign_attached_support(cands, cur_g)
        if args.native_roi_score:
            assign_native_roi_scores(cands, cur_full, args.downscale)
        assign_sky_context(cands, cur_g)
        if args.scenario_balance:
            cands = scenario_balanced_candidates(cands, args)

        states = tbd.update(fno, cands, signed_diff, signed_sigma, w_img, h_img, chosen["h"], residual_blur, app_resp)
        selected = tbd.best()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        selected_json = None
        if selected is not None:
            selected_tube_features = tube_features(selected)
            selected_tube_score = (
                tube_verifier_score(selected_tube_features, args.tube_verifier)
                if args.tube_verifier != "off"
                else 0.0
            )
            selected_json = {
                "track_id": selected.sid,
                "bbox": list(selected.bbox),
                "source": "tbd",
                "score": round(selected.score(), 3),
                "verified_score": round(tbd.verified_score(selected), 3),
                "tube_verifier_score": round(selected_tube_score, 3),
                "tube_features": {k: round(float(v), 3) for k, v in selected_tube_features.items()},
                "hits": selected.hit_count(),
                "misses": selected.misses,
                "vx": round(selected.vx, 3),
                "vy": round(selected.vy, 3),
                "candidate": selected.last_candidate.to_json() if selected.last_candidate is not None else None,
            }
            selected_rows.append([fno, selected.sid, *selected.bbox, selected.score(), selected.misses])
            row = {
                "frame": fno,
                "track_id": selected.sid,
                "x": selected.bbox[0],
                "y": selected.bbox[1],
                "w": selected.bbox[2],
                "h": selected.bbox[3],
                "score": round(selected.score(), 6),
                "verified_score": round(tbd.verified_score(selected), 6),
                "tube_verifier_score": round(selected_tube_score, 6),
                "misses": selected.misses,
                "vx": round(selected.vx, 6),
                "vy": round(selected.vy, 6),
            }
            if selected.last_candidate is not None:
                cand_json = selected.last_candidate.to_json()
                for key, value in cand_json.items():
                    if isinstance(value, (int, float, str)):
                        row[f"cand_{key}"] = value
            for key, value in selected_tube_features.items():
                row[f"tube_{key}"] = round(float(value), 6)
            selected_feature_rows.append(row)

        top_tubes_json: list[dict] = []
        if args.export_top_tubes > 0:
            ranked_scored = sorted(
                [(tbd.verified_score(st), st) for st in states],
                key=lambda item: item[0],
                reverse=True,
            )
            for rank, (score_cur, st) in enumerate(ranked_scored[: args.export_top_tubes], start=1):
                next_score = ranked_scored[rank][0] if rank < len(ranked_scored) else None
                margin = (score_cur - next_score) if next_score is not None else None
                payload, row = tube_state_payload(fno, rank, st, tbd, args, selected, margin)
                top_tubes_json.append(payload)
                top_tube_rows.append(row)

        frame_rec = {
            "frame": fno,
            "model": chosen["name"],
            "n_features": int(len(g0)),
            "inlier_ratio": round(chosen["inlier_ratio"], 3),
            "median_feature_error": round(chosen["median_feature_error"], 3),
            "threshold": round(threshold, 2),
            "appearance_threshold": round(app_threshold, 2),
            "n_candidates": len(cands),
            "n_map_candidates": len(map_cands),
            "n_native_candidates": len(native_cands),
            "n_large_dark_candidates": len(large_dark_cands),
            "n_temporal_stack_candidates": len(stack_cands),
            "n_hybrid_coast_candidates": len(hybrid_coast_cands),
            "n_tracks": len(states),
            "selected": selected_json,
            "kinematic_reject": None,
            "process_ms": round(dt_ms, 3),
            "top_candidates": [c.to_json() for c in cands[: args.top_k_debug]],
        }
        if args.export_top_tubes > 0:
            frame_rec["top_tubes"] = top_tubes_json
        report.append(frame_rec)

        if args.save_every and fno % args.save_every == 0:
            cv2.imwrite(str(out_dir / f"residual_{fno:05d}.png"), residual_blur)
            debug_mask = motion_mask if args.appearance == "off" else cv2.bitwise_or(motion_mask, app_mask)
            cv2.imwrite(str(out_dir / f"mask_{fno:05d}.png"), debug_mask)
            ov = draw_overlay(cur, fno, chosen["name"], chosen["inlier_ratio"], threshold, cands, states, selected, args)
            cv2.imwrite(str(out_dir / f"overlay_{fno:05d}.png"), ov)
        if video_writer is not None:
            ov = draw_overlay(cur, fno, chosen["name"], chosen["inlier_ratio"], threshold, cands, states, selected, args)
            video_writer.write(ov)

        prev_mask = mask
        prev_g = cur_g
        prev_full = cur_full.copy()
        fno += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if not report:
        raise SystemExit("no usable frame pairs")

    avg_ms = float(np.mean([r["process_ms"] for r in report]))
    avg_inlier = float(np.mean([r["inlier_ratio"] for r in report]))
    avg_candidates = float(np.mean([r["n_candidates"] for r in report]))
    med_candidates = float(np.median([r["n_candidates"] for r in report]))
    p90_candidates = float(np.percentile([r["n_candidates"] for r in report], 90))
    selected_frames = sum(1 for r in report if r["selected"] is not None)
    selected_rate = selected_frames / len(report)
    noisy_frames = sum(1 for r in report if r["n_candidates"] > 10)

    result = {
        "video": args.video,
        "source_frames": n_total,
        "source_fps": fps_src,
        "downscale": args.downscale,
        "args": vars(args),
        "summary": {
            "n_processed": len(report),
            "avg_ms_per_frame": round(avg_ms, 3),
            "fits_30hz": avg_ms <= 33.3,
            "fits_60hz_on_this_machine": avg_ms <= 16.7,
            "avg_inlier_ratio": round(avg_inlier, 3),
            "avg_candidates_per_frame": round(avg_candidates, 3),
            "median_candidates_per_frame": round(med_candidates, 3),
            "p90_candidates_per_frame": round(p90_candidates, 3),
            "selected_frames": selected_frames,
            "selected_frame_rate": round(selected_rate, 3),
            "kinematic_gate_px_per_frame": round(px_per_frame, 3),
            "kinematic_rejections": 0,
            "multi_candidate_frames": sum(1 for r in report if r["n_candidates"] > 1),
            "noisy_frames_gt10_candidates": noisy_frames,
            "model_counts": model_counts,
        },
        "frames": report,
    }
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "selected_tracks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x", "y", "w", "h", "score", "misses"])
        writer.writerows(selected_rows)
    if selected_feature_rows:
        fieldnames: list[str] = []
        for row in selected_feature_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "selected_tubes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_feature_rows)
    if top_tube_rows:
        fieldnames = []
        for row in top_tube_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "top_tubes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_tube_rows)
    (out_dir / "summary.md").write_text(
        f"""# candidate TBD summary

Video: `{args.video}`  
Source: {n_total} frames @ {fps_src:.2f} fps  
Processed: {len(report)} frame pairs at downscale {args.downscale}

| metric | value |
|---|---:|
| Avg time / frame | {avg_ms:.2f} ms |
| Avg RANSAC inlier ratio | {avg_inlier:.3f} |
| Avg candidates / frame | {avg_candidates:.2f} |
| Median candidates / frame | {med_candidates:.1f} |
| P90 candidates / frame | {p90_candidates:.1f} |
| Frames with selected box | {selected_frames}/{len(report)} ({selected_rate:.1%}) |
| Kinematic gate | {px_per_frame:.1f} px/frame |

This is candidate-level track-before-detect: selected_frame_rate is not accuracy.
"""
    )
    print(
        f"done -> {out_dir / 'summary.md'}\n"
        f"avg {avg_ms:.2f} ms/frame | inlier {avg_inlier:.1%} | "
        f"cand {avg_candidates:.1f}/frame p90 {p90_candidates:.1f} | selected {selected_rate:.1%}"
    )


if __name__ == "__main__":
    run(parse_args())
