#!/usr/bin/env python3
"""
General moving-camera motion detector.

This is a stricter successor to validate_motion_pipeline.py. It keeps the
same core idea, but adds the pieces that matter on cluttered footage:

  * forward/backward checked LK tracks for cleaner ego-motion features
  * affine/full-affine/homography model selection
  * adaptive residual thresholding
  * border rejection and component shape filtering
  * temporal confirmation and a lightweight tracker
  * one selected box per frame, plus diagnostic overlays and JSON

The detector is class-agnostic: it only reasons about independent image motion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--output_dir", default="results_v2")
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
    p.add_argument("--fusion", choices=("union", "fallback"), default="fallback")
    p.add_argument("--appearance_sigma", type=float, default=7.0)
    p.add_argument("--appearance_percentile", type=float, default=99.5)
    p.add_argument("--appearance_blur", type=float, default=6.0)
    p.add_argument("--min_app_residual", type=float, default=1.5)
    p.add_argument("--strong_appearance", type=float, default=45.0)
    p.add_argument("--min_appearance_mean", type=float, default=45.0)
    p.add_argument("--max_appearance_isolation", type=float, default=999.0)
    p.add_argument("--min_area", type=int, default=3)
    p.add_argument("--max_area", type=int, default=650)
    p.add_argument("--min_fill", type=float, default=0.08)
    p.add_argument("--max_aspect", type=float, default=5.0)
    p.add_argument("--border_frac", type=float, default=0.025)
    p.add_argument("--temporal_radius", type=int, default=7)
    p.add_argument("--min_hits", type=int, default=2)
    p.add_argument("--max_misses", type=int, default=5)
    p.add_argument("--max_selected_misses", type=int, default=1)
    p.add_argument("--max_match_dist", type=float, default=48.0)
    p.add_argument("--kinematic_gate", action="store_true")
    p.add_argument("--horizontal_fov_deg", type=float, default=120.0)
    p.add_argument("--max_relative_speed_mps", type=float, default=10.0)
    p.add_argument("--min_range_m", type=float, default=2.0)
    p.add_argument("--kinematic_slack_px", type=float, default=8.0)
    p.add_argument("--selection_gate_factor", type=float, default=1.2)
    p.add_argument("--selected_score", type=float, default=1.1)
    p.add_argument("--top_k_debug", type=int, default=12)
    p.add_argument("--draw_debug", action="store_true")
    p.add_argument("--local_flow", action="store_true")
    p.add_argument("--local_flow_tracks", choices=("sparse", "grid", "both"), default="grid")
    p.add_argument("--local_flow_grid_step", type=int, default=7)
    p.add_argument("--local_flow_min_grad", type=float, default=4.0)
    p.add_argument("--local_flow_fb_px", type=float, default=1.2)
    p.add_argument("--local_flow_inner_pad", type=int, default=3)
    p.add_argument("--local_flow_outer_scale", type=float, default=5.0)
    p.add_argument("--local_flow_min_outer_px", type=int, default=18)
    p.add_argument("--local_flow_min_inside", type=int, default=2)
    p.add_argument("--local_flow_min_annulus", type=int, default=10)
    p.add_argument("--local_flow_sigma_floor", type=float, default=0.45)
    p.add_argument("--local_flow_neutral_z", type=float, default=1.2)
    p.add_argument("--local_flow_z_scale", type=float, default=1.0)
    p.add_argument("--local_flow_weight", type=float, default=0.25)
    p.add_argument("--local_flow_score_cap", type=float, default=5.0)
    p.add_argument("--local_flow_comp_weight", type=float, default=0.75)
    p.add_argument("--local_flow_comp_min_residual", type=float, default=3.0)
    p.add_argument("--local_flow_comp_neutral_ratio", type=float, default=0.85)
    p.add_argument("--local_flow_comp_ratio_scale", type=float, default=0.35)
    p.add_argument("--stabilized_motion", action="store_true")
    p.add_argument("--stabilized_motion_weight", type=float, default=1.1)
    p.add_argument("--stabilized_motion_neutral_px", type=float, default=0.55)
    p.add_argument("--stabilized_motion_scale_px", type=float, default=1.1)
    p.add_argument("--temporal_bg", action="store_true")
    p.add_argument("--temporal_bg_window", type=int, default=5)
    p.add_argument("--temporal_bg_min_frames", type=int, default=3)
    p.add_argument("--temporal_bg_sigma", type=float, default=5.0)
    p.add_argument("--temporal_bg_percentile", type=float, default=97.0)
    p.add_argument("--temporal_bg_min_threshold", type=float, default=10.0)
    return p.parse_args()


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def robust_threshold(residual: np.ndarray, sigma: float, pct: float, floor: float) -> float:
    sample = residual.reshape(-1).astype(np.float32)
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    sigma_est = 1.4826 * mad
    percentile = float(np.percentile(sample, pct))
    return max(floor, med + sigma * sigma_est, percentile)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    iw = max(0, x1 - x0)
    ih = max(0, y1 - y0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter) / union if union else 0.0


def bbox_center(b: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = b
    return x + 0.5 * w, y + 0.5 * h


def kinematic_px_per_frame(w_img: int, fps: float, args: argparse.Namespace) -> float:
    if not args.kinematic_gate:
        return args.max_match_dist
    fps = max(1.0, fps)
    fov_deg = max(1.0, min(179.0, args.horizontal_fov_deg))
    f_px = (0.5 * w_img) / math.tan(0.5 * math.radians(fov_deg))
    range_m = max(0.1, args.min_range_m)
    physical_step = f_px * args.max_relative_speed_mps / range_m / fps
    return min(args.max_match_dist, physical_step + args.kinematic_slack_px)


def kinematic_reject_reason(
    previous_bbox: tuple[int, int, int, int] | None,
    previous_frame: int | None,
    current_bbox: tuple[int, int, int, int],
    current_frame: int,
    px_per_frame: float,
    factor: float,
) -> str | None:
    if previous_bbox is None or previous_frame is None:
        return None
    dt = max(1, current_frame - previous_frame)
    px_allowed = px_per_frame * factor * dt
    pcx, pcy = bbox_center(previous_bbox)
    ccx, ccy = bbox_center(current_bbox)
    dist = math.hypot(ccx - pcx, ccy - pcy)
    if dist > px_allowed:
        return f"jump_px={dist:.1f} allowed_px={px_allowed:.1f} dt={dt}"
    return None


def expanded_bbox(b: tuple[int, int, int, int], pad: int, w_img: int, h_img: int) -> tuple[int, int, int, int]:
    x, y, w, h = b
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def clip_bbox_float(b: tuple[float, float, float, float], w_img: int, h_img: int) -> tuple[int, int, int, int]:
    x, y, w, h = b
    x = max(0, min(w_img - 1, x))
    y = max(0, min(h_img - 1, y))
    w = max(1, min(w_img - x, w))
    h = max(1, min(h_img - y, h))
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


@dataclass
class Candidate:
    source: str
    bbox: tuple[int, int, int, int]
    area: int
    fill: float
    aspect: float
    mean_residual: float
    mean_appearance: float
    local_contrast: float
    texture: float
    line_context: float
    isolation: float
    score: float
    local_flow_score: float = 0.0
    local_flow_delta_px: float = 0.0
    local_flow_bg_sigma: float = 0.0
    local_flow_inside_n: int = 0
    local_flow_annulus_n: int = 0
    local_bg_residual: float = 0.0
    local_residual_ratio: float = 0.0
    local_comp_score: float = 0.0
    stab_x: float | None = None
    stab_y: float | None = None
    map_score: float = 0.0
    attached_support: float = 0.0
    native_dark_score: float = 0.0
    sky_like: float = 0.0

    def to_json(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "source": self.source,
            "area": self.area,
            "fill": round(self.fill, 3),
            "aspect": round(self.aspect, 3),
            "mean_residual": round(self.mean_residual, 2),
            "mean_appearance": round(self.mean_appearance, 2),
            "local_contrast": round(self.local_contrast, 2),
            "texture": round(self.texture, 2),
            "line_context": round(self.line_context, 3),
            "isolation": round(self.isolation, 2),
            "score": round(self.score, 3),
            "local_flow_score": round(self.local_flow_score, 3),
            "local_flow_delta_px": round(self.local_flow_delta_px, 3),
            "local_flow_bg_sigma": round(self.local_flow_bg_sigma, 3),
            "local_flow_inside_n": self.local_flow_inside_n,
            "local_flow_annulus_n": self.local_flow_annulus_n,
            "local_bg_residual": round(self.local_bg_residual, 2),
            "local_residual_ratio": round(self.local_residual_ratio, 3),
            "local_comp_score": round(self.local_comp_score, 3),
            "stabilized_center": [round(self.stab_x, 2), round(self.stab_y, 2)]
            if self.stab_x is not None and self.stab_y is not None
            else None,
            "map_score": round(self.map_score, 3),
            "attached_support": round(self.attached_support, 3),
            "native_dark_score": round(self.native_dark_score, 3),
            "sky_like": round(self.sky_like, 3),
        }


@dataclass
class Track:
    tid: int
    bbox: tuple[int, int, int, int]
    score_ema: float
    hits: int = 1
    age: int = 1
    misses: int = 0
    vx: float = 0.0
    vy: float = 0.0
    last_frame: int = 0
    last_candidate: Candidate | None = None
    stab_x: float | None = None
    stab_y: float | None = None
    stab_speed_ema: float = 0.0
    motion_score_ema: float = 0.0
    history: list[tuple[int, tuple[int, int, int, int], float, bool]] = field(default_factory=list)

    def predict_bbox(self, w_img: int, h_img: int) -> tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return clip_bbox_float((x + self.vx, y + self.vy, w, h), w_img, h_img)

    def update(
        self,
        frame_no: int,
        cand: Candidate,
        stabilized_weight: float,
        stabilized_neutral_px: float,
        stabilized_scale_px: float,
    ) -> None:
        old_cx, old_cy = bbox_center(self.bbox)
        new_cx, new_cy = bbox_center(cand.bbox)
        self.vx = 0.65 * self.vx + 0.35 * (new_cx - old_cx)
        self.vy = 0.65 * self.vy + 0.35 * (new_cy - old_cy)
        if self.stab_x is not None and self.stab_y is not None and cand.stab_x is not None and cand.stab_y is not None:
            dt_frames = max(1, frame_no - self.last_frame)
            stab_speed = math.hypot(cand.stab_x - self.stab_x, cand.stab_y - self.stab_y) / dt_frames
            motion_like = math.tanh((stab_speed - stabilized_neutral_px) / max(0.05, stabilized_scale_px))
            self.stab_speed_ema = 0.72 * self.stab_speed_ema + 0.28 * stab_speed
            self.motion_score_ema = 0.78 * self.motion_score_ema + 0.22 * stabilized_weight * motion_like
        if cand.stab_x is not None and cand.stab_y is not None:
            self.stab_x = cand.stab_x
            self.stab_y = cand.stab_y
        self.bbox = cand.bbox
        self.score_ema = 0.72 * self.score_ema + 0.28 * cand.score
        self.last_candidate = cand
        self.hits += 1
        self.age += 1
        self.misses = 0
        self.last_frame = frame_no
        self.history.append((frame_no, cand.bbox, cand.score, True))

    def mark_missed(self, frame_no: int, w_img: int, h_img: int) -> None:
        self.bbox = self.predict_bbox(w_img, h_img)
        self.age += 1
        self.misses += 1
        self.score_ema *= 0.82
        self.motion_score_ema *= 0.86
        self.last_frame = frame_no
        self.history.append((frame_no, self.bbox, self.score_ema, False))

    def selection_score(self) -> float:
        persistence = min(1.0, self.hits / 8.0)
        miss_penalty = 0.28 * self.misses
        return self.score_ema + 1.4 * persistence + self.motion_score_ema - miss_penalty

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits


class Tracker:
    def __init__(
        self,
        max_match_dist: float,
        max_misses: int,
        min_hits: int,
        stabilized_weight: float,
        stabilized_neutral_px: float,
        stabilized_scale_px: float,
    ):
        self.max_match_dist = max_match_dist
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.stabilized_weight = stabilized_weight
        self.stabilized_neutral_px = stabilized_neutral_px
        self.stabilized_scale_px = stabilized_scale_px
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, frame_no: int, cands: list[Candidate], w_img: int, h_img: int) -> list[Track]:
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_cands = set(range(len(cands)))
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            pred = tr.predict_bbox(w_img, h_img)
            pcx, pcy = bbox_center(pred)
            match_limit = self.max_match_dist * max(1, tr.misses + 1)
            for ci, cand in enumerate(cands):
                ccx, ccy = bbox_center(cand.bbox)
                dist = math.hypot(ccx - pcx, ccy - pcy)
                iou = bbox_iou(pred, cand.bbox)
                if dist <= match_limit or (iou > 0.15 and dist <= 2.0 * match_limit):
                    cost = dist - 35.0 * iou - 8.0 * cand.score
                    pairs.append((cost, ti, ci))
        for _, ti, ci in sorted(pairs):
            if ti not in unmatched_tracks or ci not in unmatched_cands:
                continue
            self.tracks[ti].update(
                frame_no,
                cands[ci],
                self.stabilized_weight,
                self.stabilized_neutral_px,
                self.stabilized_scale_px,
            )
            unmatched_tracks.remove(ti)
            unmatched_cands.remove(ci)
        for ti in sorted(unmatched_tracks):
            self.tracks[ti].mark_missed(frame_no, w_img, h_img)
        for ci in sorted(unmatched_cands):
            cand = cands[ci]
            tr = Track(self.next_id, cand.bbox, cand.score, last_frame=frame_no)
            tr.last_candidate = cand
            if cand.stab_x is not None and cand.stab_y is not None:
                tr.stab_x = cand.stab_x
                tr.stab_y = cand.stab_y
            tr.history.append((frame_no, cand.bbox, cand.score, True))
            self.next_id += 1
            self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return self.tracks

    def best(self, score_floor: float, max_selected_misses: int = 0) -> Track | None:
        confirmed = [
            tr for tr in self.tracks
            if tr.is_confirmed(self.min_hits) and tr.misses <= max_selected_misses
        ]
        if not confirmed:
            return None
        best = max(confirmed, key=lambda tr: tr.selection_score())
        return best if best.selection_score() >= score_floor else None


def lk_tracks(prev_g: np.ndarray, cur_g: np.ndarray, mask: np.ndarray | None, args: argparse.Namespace):
    pts0 = cv2.goodFeaturesToTrack(
        prev_g,
        maxCorners=args.max_corners,
        qualityLevel=args.quality,
        minDistance=args.min_distance,
        mask=mask,
    )
    if pts0 is None or len(pts0) < 20:
        return None, None
    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev_g,
        cur_g,
        pts0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    pts0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
        cur_g,
        prev_g,
        pts1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    ok = (st1.flatten() == 1) & (st2.flatten() == 1)
    fb_err = np.linalg.norm(pts0.reshape(-1, 2) - pts0_back.reshape(-1, 2), axis=1)
    ok &= fb_err < 1.5
    g0 = pts0.reshape(-1, 2)[ok]
    g1 = pts1.reshape(-1, 2)[ok]
    if len(g0) < 20:
        return None, None
    return g0.astype(np.float32), g1.astype(np.float32)


def residual_flow_from_model(h: np.ndarray, g0: np.ndarray, g1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    warped = cv2.perspectiveTransform(g0.reshape(-1, 1, 2), h).reshape(-1, 2)
    residual_flow = g1 - warped
    ok = np.isfinite(g1).all(axis=1) & np.isfinite(residual_flow).all(axis=1)
    return g1[ok].astype(np.float32), residual_flow[ok].astype(np.float32)


def grid_lk_tracks(prev_g: np.ndarray, cur_g: np.ndarray, args: argparse.Namespace):
    h_img, w_img = prev_g.shape[:2]
    step = max(3, int(args.local_flow_grid_step))
    margin = max(4, step)
    xs = np.arange(margin, max(margin + 1, w_img - margin), step, dtype=np.float32)
    ys = np.arange(margin, max(margin + 1, h_img - margin), step, dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        return None, None
    xx, yy = np.meshgrid(xs, ys)
    pts0 = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32).reshape(-1, 1, 2)

    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev_g,
        cur_g,
        pts0,
        None,
        winSize=(17, 17),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.03),
    )
    pts0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
        cur_g,
        prev_g,
        pts1,
        None,
        winSize=(17, 17),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.03),
    )
    ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    fb_err = np.linalg.norm(pts0.reshape(-1, 2) - pts0_back.reshape(-1, 2), axis=1)
    ok &= fb_err <= args.local_flow_fb_px

    if args.local_flow_min_grad > 0:
        gx = cv2.Sobel(prev_g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(prev_g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        p = pts0.reshape(-1, 2)
        xi = np.clip(np.rint(p[:, 0]).astype(np.int32), 0, w_img - 1)
        yi = np.clip(np.rint(p[:, 1]).astype(np.int32), 0, h_img - 1)
        ok &= mag[yi, xi] >= args.local_flow_min_grad

    g0 = pts0.reshape(-1, 2)[ok]
    g1 = pts1.reshape(-1, 2)[ok]
    if len(g0) < 20:
        return None, None
    return g0.astype(np.float32), g1.astype(np.float32)


def local_flow_points(
    prev_g: np.ndarray,
    cur_g: np.ndarray,
    h: np.ndarray,
    sparse_g0: np.ndarray,
    sparse_g1: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    point_sets: list[np.ndarray] = []
    flow_sets: list[np.ndarray] = []

    if args.local_flow_tracks in ("sparse", "both"):
        pts, flows = residual_flow_from_model(h, sparse_g0, sparse_g1)
        point_sets.append(pts)
        flow_sets.append(flows)

    if args.local_flow_tracks in ("grid", "both"):
        grid_g0, grid_g1 = grid_lk_tracks(prev_g, cur_g, args)
        if grid_g0 is not None and grid_g1 is not None:
            pts, flows = residual_flow_from_model(h, grid_g0, grid_g1)
            point_sets.append(pts)
            flow_sets.append(flows)

    if not point_sets:
        return None, None
    return np.vstack(point_sets).astype(np.float32), np.vstack(flow_sets).astype(np.float32)


def robust_vector_scale(vectors: np.ndarray, floor: float) -> float:
    if len(vectors) == 0:
        return floor
    med = np.median(vectors, axis=0)
    dev = np.abs(vectors - med)
    component_mad = np.median(dev, axis=0)
    sigma_vec = 1.4826 * component_mad
    sigma = float(np.sqrt(np.mean(np.square(sigma_vec))))
    return max(floor, sigma)


def points_in_bbox(points: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return (
        (points[:, 0] >= x)
        & (points[:, 0] < x + w)
        & (points[:, 1] >= y)
        & (points[:, 1] < y + h)
    )


def shifted_residual_mean(
    bbox: tuple[int, int, int, int],
    warped: np.ndarray,
    cur_g: np.ndarray,
    shift_xy: np.ndarray,
) -> float:
    h_img, w_img = cur_g.shape[:2]
    dx = float(np.clip(shift_xy[0], -16.0, 16.0))
    dy = float(np.clip(shift_xy[1], -16.0, 16.0))
    pad = int(math.ceil(max(abs(dx), abs(dy)))) + 4
    x, y, w, h = bbox
    cx, cy, cw, ch = expanded_bbox(bbox, pad, w_img, h_img)
    warped_crop = warped[cy : cy + ch, cx : cx + cw]
    cur_crop = cur_g[cy : cy + ch, cx : cx + cw]
    if warped_crop.size == 0 or cur_crop.size == 0:
        return 0.0
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(
        warped_crop,
        m,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    ix = x - cx
    iy = y - cy
    roi = cv2.absdiff(shifted, cur_crop)[iy : iy + h, ix : ix + w]
    return float(roi.mean()) if roi.size else 0.0


def apply_local_flow_scores(
    cands: list[Candidate],
    points: np.ndarray | None,
    residual_flows: np.ndarray | None,
    warped: np.ndarray,
    cur_g: np.ndarray,
    w_img: int,
    h_img: int,
    args: argparse.Namespace,
) -> list[Candidate]:
    if not args.local_flow or points is None or residual_flows is None or len(points) == 0:
        return cands

    for cand in cands:
        x, y, w, h = cand.bbox
        inner_pad = max(args.local_flow_inner_pad, int(round(0.35 * max(w, h))))
        outer_pad = max(args.local_flow_min_outer_px, int(round(args.local_flow_outer_scale * max(w, h))))
        inner_bbox = expanded_bbox(cand.bbox, inner_pad, w_img, h_img)
        outer_bbox = expanded_bbox(cand.bbox, outer_pad, w_img, h_img)

        inside_mask = points_in_bbox(points, inner_bbox)
        outer_mask = points_in_bbox(points, outer_bbox)
        annulus_mask = outer_mask & ~inside_mask
        inside_n = int(np.count_nonzero(inside_mask))
        annulus_n = int(np.count_nonzero(annulus_mask))
        cand.local_flow_inside_n = inside_n
        cand.local_flow_annulus_n = annulus_n

        if annulus_n < args.local_flow_min_annulus:
            continue

        annulus = residual_flows[annulus_mask]
        annulus_med = np.median(annulus, axis=0)
        bg_sigma = robust_vector_scale(annulus, args.local_flow_sigma_floor)
        cand.local_flow_bg_sigma = bg_sigma

        if cand.mean_residual >= args.local_flow_comp_min_residual:
            local_bg_res = shifted_residual_mean(cand.bbox, warped, cur_g, annulus_med)
            ratio = (local_bg_res + 1.0) / (cand.mean_residual + 1.0)
            comp_like = math.tanh(
                (ratio - args.local_flow_comp_neutral_ratio)
                / max(0.05, args.local_flow_comp_ratio_scale)
            )
            cand.local_bg_residual = local_bg_res
            cand.local_residual_ratio = float(ratio)
            cand.local_comp_score = float(comp_like)
            cand.score += args.local_flow_comp_weight * comp_like

        if inside_n >= args.local_flow_min_inside:
            inside = residual_flows[inside_mask]
            inside_med = np.median(inside, axis=0)
            delta = float(np.linalg.norm(inside_med - annulus_med))
            z = delta / bg_sigma
            capped_z = min(args.local_flow_score_cap, z)
            likelihood = math.tanh((capped_z - args.local_flow_neutral_z) / max(0.1, args.local_flow_z_scale))

            cand.local_flow_delta_px = delta
            cand.local_flow_score = float(likelihood)
            cand.score += args.local_flow_weight * likelihood

    cands = [c for c in cands if c.score > 0.1]
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def assign_stabilized_centers(cands: list[Candidate], cur_to_ref_h: np.ndarray | None) -> None:
    if cur_to_ref_h is None or not cands:
        return
    centers = np.array([bbox_center(c.bbox) for c in cands], dtype=np.float32).reshape(-1, 1, 2)
    try:
        stabilized = cv2.perspectiveTransform(centers, cur_to_ref_h).reshape(-1, 2)
    except cv2.error:
        return
    for cand, point in zip(cands, stabilized):
        if np.isfinite(point).all():
            cand.stab_x = float(point[0])
            cand.stab_y = float(point[1])


def temporal_background_residual(
    cur_g: np.ndarray,
    cur_to_ref_h: np.ndarray | None,
    frame_buffer: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, float | None]:
    if not args.temporal_bg or cur_to_ref_h is None or len(frame_buffer) < args.temporal_bg_min_frames:
        return None, None
    h_img, w_img = cur_g.shape[:2]
    try:
        ref_to_cur_h = np.linalg.inv(cur_to_ref_h)
    except np.linalg.LinAlgError:
        return None, None

    warped_frames = []
    for img, img_to_ref_h in frame_buffer[-max(1, args.temporal_bg_window) :]:
        h_to_cur = ref_to_cur_h @ img_to_ref_h
        if abs(float(h_to_cur[2, 2])) < 1e-6:
            continue
        h_to_cur = (h_to_cur / h_to_cur[2, 2]).astype(np.float32)
        warped = cv2.warpPerspective(
            img,
            h_to_cur,
            (w_img, h_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        warped_frames.append(warped)

    if len(warped_frames) < args.temporal_bg_min_frames:
        return None, None
    median_bg = np.median(np.stack(warped_frames, axis=0), axis=0).astype(np.uint8)
    residual = cv2.absdiff(median_bg, cur_g)
    residual = cv2.GaussianBlur(residual, (3, 3), 0)
    threshold = robust_threshold(
        residual,
        args.temporal_bg_sigma,
        args.temporal_bg_percentile,
        args.temporal_bg_min_threshold,
    )
    return residual, threshold


def model_candidates(g0: np.ndarray, g1: np.ndarray, args: argparse.Namespace) -> Iterable[tuple[str, np.ndarray, np.ndarray | None]]:
    if args.model in ("partial_affine", "auto"):
        m, inliers = cv2.estimateAffinePartial2D(
            g0,
            g1,
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac_px,
            maxIters=2500,
            confidence=0.995,
            refineIters=10,
        )
        if m is not None:
            h = np.eye(3, dtype=np.float32)
            h[:2, :] = m
            yield "partial_affine", h, inliers
    if args.model in ("full_affine", "auto"):
        m, inliers = cv2.estimateAffine2D(
            g0,
            g1,
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac_px,
            maxIters=2500,
            confidence=0.995,
            refineIters=10,
        )
        if m is not None:
            h = np.eye(3, dtype=np.float32)
            h[:2, :] = m
            yield "full_affine", h, inliers
    if args.model in ("homography", "auto") and len(g0) >= 35:
        h, inliers = cv2.findHomography(
            g0,
            g1,
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac_px,
            maxIters=2500,
            confidence=0.995,
        )
        if h is not None:
            yield "homography", h.astype(np.float32), inliers


def reprojection_stats(h: np.ndarray, g0: np.ndarray, g1: np.ndarray, inliers: np.ndarray | None) -> tuple[float, float]:
    p0 = cv2.perspectiveTransform(g0.reshape(-1, 1, 2), h).reshape(-1, 2)
    err = np.linalg.norm(p0 - g1, axis=1)
    if inliers is not None:
        mask = inliers.reshape(-1).astype(bool)
        if mask.any():
            err_eval = err[mask]
            inlier_ratio = float(mask.mean())
        else:
            err_eval = err
            inlier_ratio = 0.0
    else:
        err_eval = err
        inlier_ratio = 1.0
    return float(np.median(err_eval)), inlier_ratio


def choose_model(prev_g: np.ndarray, cur_g: np.ndarray, g0: np.ndarray, g1: np.ndarray, args: argparse.Namespace):
    best = None
    for name, h, inliers in model_candidates(g0, g1, args):
        med_err, inlier_ratio = reprojection_stats(h, g0, g1, inliers)
        complexity_penalty = {"partial_affine": 0.0, "full_affine": 0.08, "homography": 0.16}[name]
        score = med_err + complexity_penalty - 0.55 * inlier_ratio
        if best is None or score < best["score"]:
            best = {
                "name": name,
                "h": h,
                "inliers": inliers,
                "median_feature_error": med_err,
                "inlier_ratio": inlier_ratio,
                "score": score,
            }
    return best


def warp_prev(prev_g: np.ndarray, h: np.ndarray, w_img: int, h_img: int) -> np.ndarray:
    return cv2.warpPerspective(prev_g, h, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def candidate_score(
    source: str,
    bbox: tuple[int, int, int, int],
    area: int,
    residual: np.ndarray,
    appearance: np.ndarray,
    candidate_mask: np.ndarray,
    cur_g: np.ndarray,
) -> Candidate:
    h_img, w_img = residual.shape[:2]
    x, y, w, h = bbox
    rect_area = max(1, w * h)
    fill = area / rect_area
    aspect = max(w / max(1, h), h / max(1, w))
    roi_res = residual[y : y + h, x : x + w]
    mean_res = float(roi_res.mean()) if roi_res.size else 0.0
    roi_app = appearance[y : y + h, x : x + w]
    mean_app = float(roi_app.mean()) if roi_app.size else 0.0

    ex, ey, ew, eh = expanded_bbox(bbox, max(8, int(2.2 * max(w, h))), w_img, h_img)
    outer = cur_g[ey : ey + eh, ex : ex + ew].astype(np.float32)
    inner = cur_g[y : y + h, x : x + w].astype(np.float32)
    ring_mask = np.ones((eh, ew), dtype=np.uint8)
    ix0 = x - ex
    iy0 = y - ey
    ring_mask[iy0 : iy0 + h, ix0 : ix0 + w] = 0
    ring_vals = outer[ring_mask.astype(bool)] if outer.size else np.array([], dtype=np.float32)
    if ring_vals.size and inner.size:
        local_contrast = abs(float(inner.mean()) - float(ring_vals.mean()))
    else:
        local_contrast = 0.0
    if ring_vals.size:
        gx = cv2.Sobel(outer, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(outer, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ring_bool = ring_mask.astype(bool)
        texture = float(np.mean(mag[ring_bool]))
        rgx = gx[ring_bool].reshape(-1)
        rgy = gy[ring_bool].reshape(-1)
        gxx = float(np.mean(rgx * rgx))
        gyy = float(np.mean(rgy * rgy))
        gxy = float(np.mean(rgx * rgy))
        trace = gxx + gyy
        disc = max(0.0, (gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy)
        lam1 = 0.5 * (trace + math.sqrt(disc))
        lam2 = 0.5 * (trace - math.sqrt(disc))
        linearity = (lam1 - lam2) / max(1e-6, lam1 + lam2)
        line_context = float(linearity * min(1.4, texture / 35.0))
    else:
        texture = 0.0
        line_context = 0.0

    expanded_mask = candidate_mask[ey : ey + eh, ex : ex + ew]
    inner_mask = candidate_mask[y : y + h, x : x + w]
    neighbor_pixels = max(0, int(np.count_nonzero(expanded_mask)) - int(np.count_nonzero(inner_mask)))
    isolation = neighbor_pixels / max(1.0, float(area))

    area_pref = math.exp(-abs(math.log(max(area, 1) / 32.0)) / 2.2)
    compact_pref = min(1.0, fill / 0.32)
    aspect_pref = max(0.0, 1.0 - (aspect - 1.0) / 6.0)
    residual_pref = min(1.8, mean_res / 28.0)
    appearance_pref = min(2.0, mean_app / 20.0)
    contrast_pref = min(1.2, local_contrast / 22.0)
    texture_penalty = min(1.35, texture / 28.0)
    line_penalty = min(1.25, line_context)
    isolation_penalty = min(1.6, isolation / 2.5)
    score = (
        0.85 * area_pref
        + 0.7 * compact_pref
        + 0.45 * aspect_pref
        + 0.55 * residual_pref
        + 0.85 * appearance_pref
        + 0.35 * contrast_pref
        - 0.55 * texture_penalty
        - 0.45 * line_penalty
        - 0.65 * isolation_penalty
    )
    return Candidate(
        source,
        bbox,
        int(area),
        float(fill),
        float(aspect),
        mean_res,
        mean_app,
        local_contrast,
        texture,
        line_context,
        isolation,
        float(score),
    )


def extract_candidates(
    source: str,
    mask: np.ndarray,
    residual: np.ndarray,
    appearance: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[Candidate]:
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h_img, w_img = mask.shape[:2]
    border = int(round(args.border_frac * min(w_img, h_img)))
    cands: list[Candidate] = []
    for i in range(1, nlab):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < args.min_area or area > args.max_area:
            continue
        if border and (x <= border or y <= border or x + w >= w_img - border or y + h >= h_img - border):
            continue
        fill = area / max(1, w * h)
        aspect = max(w / max(1, h), h / max(1, w))
        if fill < args.min_fill or aspect > args.max_aspect:
            continue
        cand = candidate_score(source, (x, y, w, h), area, residual, appearance, mask, cur_g)
        if (
            source == "appearance"
            and cand.mean_residual < args.min_app_residual
            and cand.mean_appearance < args.strong_appearance
        ):
            continue
        if source == "appearance" and cand.mean_appearance < args.min_appearance_mean:
            continue
        if source == "appearance" and cand.isolation > args.max_appearance_isolation:
            continue
        if cand.score <= 0.1:
            continue
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def appearance_response(cur_g: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    if args.appearance == "off":
        z = np.zeros_like(cur_g, dtype=np.uint8)
        return z, z, 255.0
    small = cv2.GaussianBlur(cur_g, (3, 3), 0).astype(np.float32)
    bg = cv2.GaussianBlur(cur_g, (0, 0), args.appearance_blur).astype(np.float32)
    responses = []
    if args.appearance in ("dark", "both"):
        responses.append(np.maximum(0.0, bg - small))
    if args.appearance in ("bright", "both"):
        responses.append(np.maximum(0.0, small - bg))
    resp = np.maximum.reduce(responses) if len(responses) > 1 else responses[0]
    resp = np.clip(resp, 0, 255).astype(np.uint8)
    threshold = robust_threshold(resp, args.appearance_sigma, args.appearance_percentile, 8.0)
    mask = (resp >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return resp, mask, threshold


def make_feature_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    h_img, w_img = shape
    mask = np.full((h_img, w_img), 255, dtype=np.uint8)
    margin = int(round(0.02 * min(w_img, h_img)))
    if margin:
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0
    for b in boxes:
        x, y, w, h = expanded_bbox(b, 8, w_img, h_img)
        mask[y : y + h, x : x + w] = 0
    return mask


def draw_overlay(
    frame: np.ndarray,
    frame_no: int,
    model_name: str,
    inlier_ratio: float,
    threshold: float,
    cands: list[Candidate],
    tracks: list[Track],
    selected: Track | None,
    args: argparse.Namespace,
) -> np.ndarray:
    ov = frame.copy()
    if args.draw_debug:
        for cand in cands[: args.top_k_debug]:
            x, y, w, h = cand.bbox
            cv2.rectangle(ov, (x, y), (x + w, y + h), (80, 170, 80), 1)
        for tr in tracks:
            if tr.misses:
                continue
            x, y, w, h = tr.bbox
            cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 180, 255), 1)
            cv2.putText(ov, f"T{tr.tid}", (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)
    if selected is not None:
        x, y, w, h = selected.bbox
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            ov,
            f"SELECT T{selected.tid} {selected.selection_score():.2f}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        ov,
        f"f{frame_no:05d} {model_name} in={inlier_ratio:.2f} thr={threshold:.1f} cand={len(cands)}",
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        ov,
        f"f{frame_no:05d} {model_name} in={inlier_ratio:.2f} thr={threshold:.1f} cand={len(cands)}",
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return ov


def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit("no frames")
    if args.downscale != 1.0:
        prev = cv2.resize(prev, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
    prev_g = ensure_gray(prev)
    h_img, w_img = prev_g.shape[:2]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = fps_src if fps_src > 1 else 30.0
        video_writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps_out, (w_img, h_img))

    kinematic_step_px = kinematic_px_per_frame(w_img, fps_src, args)
    stabilized_weight = args.stabilized_motion_weight if args.stabilized_motion else 0.0
    motion_tracker = Tracker(
        kinematic_step_px,
        args.max_misses,
        args.min_hits,
        stabilized_weight,
        args.stabilized_motion_neutral_px,
        args.stabilized_motion_scale_px,
    )
    appearance_tracker = Tracker(
        kinematic_step_px,
        args.max_misses,
        args.min_hits,
        stabilized_weight,
        args.stabilized_motion_neutral_px,
        args.stabilized_motion_scale_px,
    )
    prev_mask: np.ndarray | None = None
    prev_to_ref_h = np.eye(3, dtype=np.float32)
    frame_buffer: list[tuple[np.ndarray, np.ndarray]] = [(prev_g.copy(), prev_to_ref_h.copy())]
    last_selected_boxes: list[tuple[int, int, int, int]] = []
    last_accepted_bbox: tuple[int, int, int, int] | None = None
    last_accepted_frame: int | None = None
    report: list[dict] = []
    selected_rows: list[list] = []
    model_counts: dict[str, int] = {}
    kinematic_rejections = 0
    fno = 1

    while True:
        if args.max_frames is not None and fno >= args.max_frames:
            break
        ok, cur = cap.read()
        if not ok:
            break
        if args.downscale != 1.0:
            cur = cv2.resize(cur, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
        cur_g = ensure_gray(cur)
        t0 = time.perf_counter()

        feature_mask = make_feature_mask(prev_g.shape[:2], last_selected_boxes)
        g0, g1 = lk_tracks(prev_g, cur_g, feature_mask, args)
        if g0 is None:
            prev_g = cur_g
            fno += 1
            continue

        chosen = choose_model(prev_g, cur_g, g0, g1, args)
        if chosen is None:
            prev_g = cur_g
            fno += 1
            continue
        model_counts[chosen["name"]] = model_counts.get(chosen["name"], 0) + 1
        local_points = None
        local_residual_flows = None
        if args.local_flow:
            local_points, local_residual_flows = local_flow_points(prev_g, cur_g, chosen["h"], g0, g1, args)
        try:
            cur_to_ref_h = prev_to_ref_h @ np.linalg.inv(chosen["h"])
            cur_to_ref_h = (cur_to_ref_h / cur_to_ref_h[2, 2]).astype(np.float32)
        except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
            cur_to_ref_h = None

        warped = warp_prev(prev_g, chosen["h"], w_img, h_img)
        residual = cv2.absdiff(warped, cur_g)
        residual_blur = cv2.GaussianBlur(residual, (3, 3), 0)
        threshold = robust_threshold(residual_blur, args.threshold_sigma, args.threshold_percentile, args.min_threshold)
        mask = (residual_blur >= threshold).astype(np.uint8) * 255
        residual_for_candidates = residual_blur
        temporal_threshold = None
        temporal_residual, temporal_threshold = temporal_background_residual(cur_g, cur_to_ref_h, frame_buffer, args)
        if temporal_residual is not None and temporal_threshold is not None:
            temporal_mask = (temporal_residual >= temporal_threshold).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, temporal_mask)
            residual_for_candidates = np.maximum(residual_blur, temporal_residual)
        app_resp, app_mask, app_threshold = appearance_response(cur_g, args)

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
            # Keep a small amount of current evidence so new movers are not delayed forever.
            motion_mask_for_components = cv2.bitwise_or(confirmed_mask, cv2.erode(mask, k3))
        else:
            motion_mask_for_components = mask

        motion_cands = extract_candidates("motion", motion_mask_for_components, residual_for_candidates, app_resp, cur_g, args)
        app_cands: list[Candidate] = []
        if args.appearance != "off":
            app_cands = extract_candidates("appearance", app_mask, residual_blur, app_resp, cur_g, args)
        motion_cands = apply_local_flow_scores(motion_cands, local_points, local_residual_flows, warped, cur_g, w_img, h_img, args)
        app_cands = apply_local_flow_scores(app_cands, local_points, local_residual_flows, warped, cur_g, w_img, h_img, args)
        if args.stabilized_motion:
            assign_stabilized_centers(motion_cands, cur_to_ref_h)
            assign_stabilized_centers(app_cands, cur_to_ref_h)

        if args.fusion == "union":
            cands = sorted(motion_cands + app_cands, key=lambda c: c.score, reverse=True)
            tracks = motion_tracker.update(fno, cands, w_img, h_img)
            selected = motion_tracker.best(args.selected_score, args.max_selected_misses)
            selected_source = "union" if selected is not None else None
        else:
            motion_tracks = motion_tracker.update(fno, motion_cands, w_img, h_img)
            motion_selected = motion_tracker.best(args.selected_score, args.max_selected_misses)
            app_tracks = appearance_tracker.update(fno, app_cands, w_img, h_img)
            app_selected = appearance_tracker.best(args.selected_score + 0.15, args.max_selected_misses)
            if motion_selected is not None and app_selected is not None:
                if app_selected.selection_score() > motion_selected.selection_score() + 0.45:
                    selected = app_selected
                    selected_source = "appearance"
                else:
                    selected = motion_selected
                    selected_source = "motion"
            elif motion_selected is not None:
                selected = motion_selected
                selected_source = "motion"
            else:
                selected = app_selected
                selected_source = "appearance" if app_selected is not None else None
            cands = sorted(motion_cands + app_cands, key=lambda c: c.score, reverse=True)
            tracks = motion_tracks + app_tracks
        kinematic_reject = None
        if args.kinematic_gate and selected is not None:
            kinematic_reject = kinematic_reject_reason(
                last_accepted_bbox,
                last_accepted_frame,
                selected.bbox,
                fno,
                kinematic_step_px,
                args.selection_gate_factor,
            )
            if kinematic_reject is not None:
                kinematic_rejections += 1
                selected = None
                selected_source = None
        last_selected_boxes = [selected.bbox] if selected is not None else []

        dt_ms = (time.perf_counter() - t0) * 1000.0
        selected_json = None
        if selected is not None:
            last_accepted_bbox = selected.bbox
            last_accepted_frame = fno
            selected_json = {
                "track_id": selected.tid,
                "bbox": list(selected.bbox),
                "source": selected_source,
                "score": round(selected.selection_score(), 3),
                "hits": selected.hits,
                "misses": selected.misses,
                "stab_speed_ema": round(selected.stab_speed_ema, 3),
                "motion_score_ema": round(selected.motion_score_ema, 3),
                "candidate": selected.last_candidate.to_json() if selected.last_candidate is not None else None,
            }
            selected_rows.append([fno, selected.tid, *selected.bbox, selected.selection_score(), selected.misses])

        frame_rec = {
            "frame": fno,
            "model": chosen["name"],
            "n_features": int(len(g0)),
            "inlier_ratio": round(chosen["inlier_ratio"], 3),
            "median_feature_error": round(chosen["median_feature_error"], 3),
            "threshold": round(threshold, 2),
            "temporal_threshold": round(temporal_threshold, 2) if temporal_threshold is not None else None,
            "appearance_threshold": round(app_threshold, 2),
            "n_candidates": len(cands),
            "n_tracks": len(tracks),
            "selected": selected_json,
            "kinematic_reject": kinematic_reject,
            "process_ms": round(dt_ms, 3),
            "top_candidates": [c.to_json() for c in cands[: args.top_k_debug]],
        }
        report.append(frame_rec)

        if args.save_every and fno % args.save_every == 0:
            cv2.imwrite(str(out_dir / f"residual_{fno:05d}.png"), residual_for_candidates)
            debug_mask = motion_mask_for_components if args.appearance == "off" else cv2.bitwise_or(motion_mask_for_components, app_mask)
            cv2.imwrite(str(out_dir / f"mask_{fno:05d}.png"), debug_mask)
            ov = draw_overlay(
                cur,
                fno,
                chosen["name"],
                chosen["inlier_ratio"],
                threshold,
                cands,
                tracks,
                selected,
                args,
            )
            cv2.imwrite(str(out_dir / f"overlay_{fno:05d}.png"), ov)
        if video_writer is not None:
            ov = draw_overlay(
                cur,
                fno,
                chosen["name"],
                chosen["inlier_ratio"],
                threshold,
                cands,
                tracks,
                selected,
                args,
            )
            video_writer.write(ov)

        prev_mask = mask
        prev_g = cur_g
        if cur_to_ref_h is not None:
            prev_to_ref_h = cur_to_ref_h
            frame_buffer.append((cur_g.copy(), prev_to_ref_h.copy()))
            max_buffer = max(args.temporal_bg_window + 2, args.temporal_bg_min_frames + 2, 4)
            if len(frame_buffer) > max_buffer:
                frame_buffer = frame_buffer[-max_buffer:]
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
    multi_candidate_frames = sum(1 for r in report if r["n_candidates"] > 1)
    noisy_frames = sum(1 for r in report if r["n_candidates"] > 10)
    fits30 = avg_ms <= 33.3
    fits60 = avg_ms <= 16.7

    result = {
        "video": args.video,
        "source_frames": n_total,
        "source_fps": fps_src,
        "downscale": args.downscale,
        "args": vars(args),
        "summary": {
            "n_processed": len(report),
            "avg_ms_per_frame": round(avg_ms, 3),
            "fits_30hz": fits30,
            "fits_60hz_on_this_machine": fits60,
            "avg_inlier_ratio": round(avg_inlier, 3),
            "avg_candidates_per_frame": round(avg_candidates, 3),
            "median_candidates_per_frame": round(med_candidates, 3),
            "p90_candidates_per_frame": round(p90_candidates, 3),
            "selected_frames": selected_frames,
            "selected_frame_rate": round(selected_rate, 3),
            "kinematic_gate_px_per_frame": round(kinematic_step_px, 3),
            "kinematic_rejections": kinematic_rejections,
            "multi_candidate_frames": multi_candidate_frames,
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

    (out_dir / "summary.md").write_text(
        f"""# motion detector v2 summary

Video: `{args.video}`  
Source: {n_total} frames @ {fps_src:.2f} fps  
Processed: {len(report)} frame pairs at downscale {args.downscale}

## Runtime

| metric | value |
|---|---:|
| Avg time / frame | {avg_ms:.2f} ms |
| Fits 30 Hz on this machine | {"yes" if fits30 else "no"} |
| Fits 60 Hz on this machine | {"yes" if fits60 else "no"} |

## Detection Diagnostics

| metric | value |
|---|---:|
| Avg RANSAC inlier ratio | {avg_inlier:.3f} |
| Avg candidates / frame | {avg_candidates:.2f} |
| Median candidates / frame | {med_candidates:.1f} |
| P90 candidates / frame | {p90_candidates:.1f} |
| Frames with selected box | {selected_frames}/{len(report)} ({selected_rate:.1%}) |
| Kinematic gate | {kinematic_step_px:.1f} px/frame |
| Kinematic rejections | {kinematic_rejections} |
| Frames with >1 candidates | {multi_candidate_frames}/{len(report)} |
| Frames with >10 candidates | {noisy_frames}/{len(report)} |

Model counts: `{json.dumps(model_counts, sort_keys=True)}`

## Notes

`selected_frame_rate` is not retrieval accuracy. It only says whether the tracker
emitted one selected box. Without frame-level ground truth, accuracy must be
judged from overlays or annotated labels.
"""
    )

    print(
        f"done -> {out_dir / 'summary.md'}\n"
        f"avg {avg_ms:.2f} ms/frame | inlier {avg_inlier:.1%} | "
        f"cand {avg_candidates:.1f}/frame p90 {p90_candidates:.1f} | "
        f"selected {selected_rate:.1%}"
    )


if __name__ == "__main__":
    run(parse_args())
