#!/usr/bin/env python3
"""Profile target-aligned vs background-aligned tube evidence.

This is an offline harness for the current hard surface failure mode. It asks
whether a candidate tube looks more like a compact dark object when crops are
aligned along the candidate path than when the same anchor point is propagated
by the dominant background transform.

The script is intentionally diagnostic. It does not change detector behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples", nargs="+", required=True, help="Hard-example CSVs with hard_label and candidate rows.")
    p.add_argument("--tube_csv", nargs="+", required=True, help="scored_top_tubes/top_tubes CSVs used to recover per-track paths.")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--window_radius", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=31)
    p.add_argument("--detector_scale", type=float, default=0.5, help="Detector coordinate scale relative to source video.")
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--orb_features", type=int, default=900)
    p.add_argument("--min_matches", type=int, default=18)
    p.add_argument("--random_seed", type=int, default=13)
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


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def center(row: dict[str, Any]) -> tuple[float, float]:
    return (
        safe_float(row.get("x")) + 0.5 * max(1.0, safe_float(row.get("w"), 1.0)),
        safe_float(row.get("y")) + 0.5 * max(1.0, safe_float(row.get("h"), 1.0)),
    )


def clip_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem in {"top_tubes", "scored_top_tubes"}:
        return path.parent.name
    return stem


def clip_matches(a: str, b: str) -> bool:
    return a == b or a.startswith(b) or b.startswith(a)


class FrameCache:
    def __init__(self, video_dir: Path, detector_scale: float) -> None:
        self.video_dir = video_dir
        self.detector_scale = detector_scale
        self.captures: dict[str, cv2.VideoCapture] = {}
        self.frames: dict[tuple[str, int], np.ndarray] = {}
        self.video_paths: dict[str, Path] = {}

    def video_path(self, clip: str) -> Path:
        if clip in self.video_paths:
            return self.video_paths[clip]
        for path in sorted(self.video_dir.glob("*.MP4")) + sorted(self.video_dir.glob("*.mp4")):
            if clip_matches(clip, path.stem):
                self.video_paths[clip] = path
                return path
        raise FileNotFoundError(f"no video found for clip {clip} under {self.video_dir}")

    def capture(self, clip: str) -> cv2.VideoCapture:
        if clip not in self.captures:
            cap = cv2.VideoCapture(str(self.video_path(clip)))
            if not cap.isOpened():
                raise RuntimeError(f"failed to open video for {clip}")
            self.captures[clip] = cap
        return self.captures[clip]

    def gray(self, clip: str, frame: int) -> np.ndarray | None:
        key = (clip, frame)
        if key in self.frames:
            return self.frames[key]
        cap = self.capture(clip)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame < 0 or frame >= total:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if abs(self.detector_scale - 1.0) > 1e-6:
            gray = cv2.resize(gray, None, fx=self.detector_scale, fy=self.detector_scale, interpolation=cv2.INTER_AREA)
        self.frames[key] = gray
        return gray

    def close(self) -> None:
        for cap in self.captures.values():
            cap.release()


class TransformCache:
    def __init__(self, frame_cache: FrameCache, orb_features: int, min_matches: int) -> None:
        self.frame_cache = frame_cache
        self.orb = cv2.ORB_create(nfeatures=orb_features, fastThreshold=7)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        self.transforms: dict[tuple[str, int, int], np.ndarray] = {}

    def transform(self, clip: str, anchor_frame: int, target_frame: int) -> np.ndarray:
        key = (clip, anchor_frame, target_frame)
        if key in self.transforms:
            return self.transforms[key]
        if anchor_frame == target_frame:
            mat = np.eye(3, dtype=np.float32)
            self.transforms[key] = mat
            return mat
        a = self.frame_cache.gray(clip, anchor_frame)
        b = self.frame_cache.gray(clip, target_frame)
        if a is None or b is None:
            mat = np.eye(3, dtype=np.float32)
            self.transforms[key] = mat
            return mat
        kp_a, des_a = self.orb.detectAndCompute(a, None)
        kp_b, des_b = self.orb.detectAndCompute(b, None)
        if des_a is None or des_b is None or len(kp_a) < self.min_matches or len(kp_b) < self.min_matches:
            mat = np.eye(3, dtype=np.float32)
            self.transforms[key] = mat
            return mat
        matches = sorted(self.matcher.match(des_a, des_b), key=lambda m: m.distance)
        matches = matches[: min(len(matches), 180)]
        if len(matches) < self.min_matches:
            mat = np.eye(3, dtype=np.float32)
            self.transforms[key] = mat
            return mat
        src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        aff, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=1500)
        if aff is None or inliers is None or int(inliers.sum()) < self.min_matches:
            mat = np.eye(3, dtype=np.float32)
        else:
            mat = np.eye(3, dtype=np.float32)
            mat[:2, :] = aff.astype(np.float32)
        self.transforms[key] = mat
        return mat


def project(mat: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    vec = np.asarray([pt[0], pt[1], 1.0], dtype=np.float32)
    out = mat @ vec
    if abs(float(out[2])) < 1e-6:
        return pt
    return float(out[0] / out[2]), float(out[1] / out[2])


def extract_crop(gray: np.ndarray, pt: tuple[float, float], size: int) -> np.ndarray:
    return cv2.getRectSubPix(gray.astype(np.float32), (size, size), pt)


def masks(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    rr = np.hypot(xx - cx, yy - cy)
    center_mask = rr <= max(2.0, size * 0.11)
    ring_mask = (rr >= size * 0.22) & (rr <= size * 0.42)
    outer_mask = rr >= size * 0.35
    return center_mask, ring_mask, outer_mask


def robust_sigma(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(1.4826 * mad, 2.0)


def anisotropy(img: np.ndarray, mask: np.ndarray) -> float:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    xs = gx[mask].reshape(-1)
    ys = gy[mask].reshape(-1)
    if xs.size < 6:
        return 0.0
    cov = np.cov(np.vstack([xs, ys]))
    vals = np.linalg.eigvalsh(cov)
    denom = float(vals[0] + vals[1] + 1e-6)
    return float((vals[1] - vals[0]) / denom)


def crop_dark_z(crop: np.ndarray, center_mask: np.ndarray, ring_mask: np.ndarray) -> float:
    center_val = float(np.mean(crop[center_mask]))
    ring = crop[ring_mask]
    ring_med = float(np.median(ring)) if ring.size else float(np.mean(crop))
    sigma = robust_sigma(ring)
    return (ring_med - center_val) / sigma


def stack_quality(crops: list[np.ndarray], size: int) -> dict[str, float]:
    if not crops:
        return {
            "q": 0.0,
            "mean_dark_z": 0.0,
            "stack_dark_z": 0.0,
            "center_std": 0.0,
            "ring_std": 0.0,
            "anisotropy": 0.0,
        }
    center_mask, ring_mask, outer_mask = masks(size)
    arr = np.stack(crops).astype(np.float32)
    mean_img = np.mean(arr, axis=0)
    dark_vals = [crop_dark_z(c, center_mask, ring_mask) for c in arr]
    mean_dark_z = float(np.mean(dark_vals))
    stack_dark_z = crop_dark_z(mean_img, center_mask, ring_mask)
    center_std = float(np.mean(np.std(arr[:, center_mask], axis=0)))
    ring_std = float(np.mean(np.std(arr[:, ring_mask], axis=0))) if np.any(ring_mask) else 0.0
    line = anisotropy(mean_img, outer_mask)
    q = 0.55 * mean_dark_z + 0.75 * stack_dark_z - 0.18 * line - 0.01 * max(0.0, center_std - ring_std)
    return {
        "q": float(q),
        "mean_dark_z": mean_dark_z,
        "stack_dark_z": float(stack_dark_z),
        "center_std": center_std,
        "ring_std": ring_std,
        "anisotropy": float(line),
    }


def load_tube_rows(paths: list[Path]) -> dict[tuple[str, str], dict[int, dict[str, str]]]:
    out: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for path in paths:
        if not path.exists():
            continue
        rows = read_csv(path)
        inferred_clip = clip_id_from_path(path)
        for row in rows:
            clip = row.get("clip") or inferred_clip
            tid = row.get("track_id", "")
            if not tid:
                continue
            frame = safe_int(row.get("frame"), -1)
            if frame < 0:
                continue
            existing = out[(clip, tid)].get(frame)
            if existing is None or safe_int(row.get("rank"), 999999) < safe_int(existing.get("rank"), 999999):
                out[(clip, tid)][frame] = row
    return out


def find_tube_row(
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    clip: str,
    tid: str,
    frame: int,
) -> dict[str, str] | None:
    for (row_clip, row_tid), by_frame in tube_rows.items():
        if row_tid == tid and clip_matches(clip, row_clip):
            return by_frame.get(frame)
    return None


def target_point(
    row: dict[str, str],
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    clip: str,
    frame: int,
    target_frame: int,
) -> tuple[float, float]:
    tid = row.get("track_id", "")
    match = find_tube_row(tube_rows, clip, tid, target_frame)
    if match is not None:
        return center(match)
    cx, cy = center(row)
    vx = safe_float(row.get("vx"))
    vy = safe_float(row.get("vy"))
    dt = target_frame - frame
    return cx + vx * dt, cy + vy * dt


def score_example(
    row: dict[str, str],
    frames: FrameCache,
    transforms: TransformCache,
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    radius: int,
    crop_size: int,
) -> dict[str, Any]:
    clip = row.get("clip", "")
    frame = safe_int(row.get("frame"), -1)
    anchor = center(row)
    target_crops: list[np.ndarray] = []
    bg_crops: list[np.ndarray] = []
    path_bg_dist: list[float] = []
    used_frames = 0
    transform_failures = 0
    for target_frame in range(frame - radius, frame + radius + 1):
        gray = frames.gray(clip, target_frame)
        if gray is None:
            continue
        tpt = target_point(row, tube_rows, clip, frame, target_frame)
        mat = transforms.transform(clip, frame, target_frame)
        bpt = project(mat, anchor)
        if np.allclose(mat, np.eye(3), atol=1e-6) and target_frame != frame:
            transform_failures += 1
        target_crops.append(extract_crop(gray, tpt, crop_size))
        bg_crops.append(extract_crop(gray, bpt, crop_size))
        path_bg_dist.append(float(math.hypot(tpt[0] - bpt[0], tpt[1] - bpt[1])))
        used_frames += 1
    tq = stack_quality(target_crops, crop_size)
    bq = stack_quality(bg_crops, crop_size)
    out: dict[str, Any] = dict(row)
    out.update(
        {
            "align_target_q": tq["q"],
            "align_bg_q": bq["q"],
            "align_gain": tq["q"] - bq["q"],
            "target_mean_dark_z": tq["mean_dark_z"],
            "bg_mean_dark_z": bq["mean_dark_z"],
            "target_stack_dark_z": tq["stack_dark_z"],
            "bg_stack_dark_z": bq["stack_dark_z"],
            "target_anisotropy": tq["anisotropy"],
            "bg_anisotropy": bq["anisotropy"],
            "target_center_std": tq["center_std"],
            "bg_center_std": bq["center_std"],
            "path_bg_dist_mean": float(np.mean(path_bg_dist)) if path_bg_dist else 0.0,
            "path_bg_dist_max": float(np.max(path_bg_dist)) if path_bg_dist else 0.0,
            "align_used_frames": used_frames,
            "align_transform_failures": transform_failures,
        }
    )
    return out


def roc_auc(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [(int(safe_int(r.get("hard_label"), 0)), safe_float(r.get(key))) for r in rows]
    pos = [v for y, v in vals if y == 1]
    neg = [v for y, v in vals if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else None


def pairwise_summary(rows: list[dict[str, Any]], key: str) -> tuple[int, int, float]:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[(row.get("clip", ""), safe_int(row.get("frame"), -1))].append(row)
    wins = 0
    total = 0
    for group in by_frame.values():
        pos = [r for r in group if safe_int(r.get("hard_label"), 0) == 1]
        neg = [r for r in group if safe_int(r.get("hard_label"), 0) == 0]
        for p in pos:
            for n in neg:
                total += 1
                if safe_float(p.get(key)) > safe_float(n.get(key)):
                    wins += 1
    return wins, total, wins / total if total else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "align_gain",
        "align_target_q",
        "align_bg_q",
        "path_bg_dist_mean",
        "target_stack_dark_z",
        "bg_stack_dark_z",
        "target_anisotropy",
        "bg_anisotropy",
        "learned_score",
        "verified_score",
    ]
    out: list[dict[str, Any]] = []
    clips = sorted({r.get("clip", "") for r in rows})
    for clip in ["ALL", *clips]:
        subset = rows if clip == "ALL" else [r for r in rows if r.get("clip", "") == clip]
        if not subset:
            continue
        pos = [r for r in subset if safe_int(r.get("hard_label"), 0) == 1]
        neg = [r for r in subset if safe_int(r.get("hard_label"), 0) == 0]
        for key in keys:
            pw, pt, pr = pairwise_summary(subset, key)
            out.append(
                {
                    "clip": clip,
                    "feature": key,
                    "rows": len(subset),
                    "positives": len(pos),
                    "negatives": len(neg),
                    "pos_mean": round(float(np.mean([safe_float(r.get(key)) for r in pos])) if pos else 0.0, 6),
                    "neg_mean": round(float(np.mean([safe_float(r.get(key)) for r in neg])) if neg else 0.0, 6),
                    "auc": "" if roc_auc(subset, key) is None else round(float(roc_auc(subset, key)), 6),
                    "pairwise_wins": pw,
                    "pairwise_total": pt,
                    "pairwise_win_rate": round(pr, 6),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, str]] = []
    for path_str in args.examples:
        for row in read_csv(Path(path_str)):
            if row.get("hard_label", "") in {"0", "1"} and row.get("clip") and row.get("frame"):
                examples.append(row)
    if args.max_examples and len(examples) > args.max_examples:
        idx = rng.choice(len(examples), size=args.max_examples, replace=False)
        examples = [examples[int(i)] for i in sorted(idx)]
    tube_rows = load_tube_rows([Path(p) for p in args.tube_csv])
    frame_cache = FrameCache(Path(args.video_dir), args.detector_scale)
    transform_cache = TransformCache(frame_cache, args.orb_features, args.min_matches)
    try:
        rows = [
            score_example(row, frame_cache, transform_cache, tube_rows, args.window_radius, args.crop_size)
            for row in examples
        ]
    finally:
        frame_cache.close()
    write_csv(out_dir / "alignment_features.csv", rows)
    summary = summarize(rows)
    write_csv(out_dir / "alignment_feature_summary.csv", summary)
    meta = {
        "examples": len(rows),
        "window_radius": args.window_radius,
        "crop_size": args.crop_size,
        "detector_scale": args.detector_scale,
        "tube_csv": args.tube_csv,
        "example_csv": args.examples,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    best = sorted(
        [r for r in summary if r["clip"] == "ALL"],
        key=lambda r: float(r["auc"] or 0.0),
        reverse=True,
    )[:8]
    lines = ["# Tube Alignment Feature Profile", "", "Top global features by AUC:", ""]
    for row in best:
        lines.append(
            f"- `{row['feature']}`: auc={row['auc']} pos_mean={row['pos_mean']} "
            f"neg_mean={row['neg_mean']} pairwise={row['pairwise_wins']}/{row['pairwise_total']}"
        )
    lines.append("")
    lines.append("Interpretation note: this is a hard-example separability test, not full detector accuracy.")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "examples": len(rows), "top_features": best[:3]}, indent=2))


if __name__ == "__main__":
    main()
