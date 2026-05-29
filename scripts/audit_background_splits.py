#!/usr/bin/env python3
"""Audit target-tube performance split by local background type.

This is meant for dense XY labels. It classifies the area around the labeled
target as clean-sky, boundary/mixed, or textured/non-sky, then reports proposal
oracle and selected-box accuracy per split.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--video", action="append", default=[], help="CLIP=/absolute/path/video.mp4")
    p.add_argument("--max_rank", type=int, default=100)
    p.add_argument("--center_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--score_column", default="verified_score")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_video_map(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--video must be CLIP=/path, got {item!r}")
        clip, path = item.split("=", 1)
        out[clip] = Path(path)
    return out


def clip_matches(row_clip: str, video_clip: str) -> bool:
    return row_clip == video_clip or row_clip.startswith(video_clip) or video_clip.startswith(row_clip)


def video_for_clip(clip: str, videos: dict[str, Path]) -> Path | None:
    for key, path in videos.items():
        if clip_matches(clip, key):
            return path
    return None


def row_bbox(row: dict[str, str], prefix: str = "") -> tuple[float, float, float, float]:
    keys = (f"{prefix}x", f"{prefix}y", f"{prefix}w", f"{prefix}h")
    if all(k in row and row[k] not in ("", None) for k in keys):
        return tuple(safe_float(row[k], 0.0) for k in keys)  # type: ignore[return-value]
    return (
        safe_float(row.get("det_x", row.get("x", 0.0))),
        safe_float(row.get("det_y", row.get("y", 0.0))),
        safe_float(row.get("det_w", row.get("w", 1.0)), 1.0),
        safe_float(row.get("det_h", row.get("h", 1.0)), 1.0),
    )


def center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + 0.5 * w, y + 0.5 * h


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def load_top_tubes(results_dir: Path, clip: str, max_rank: int) -> dict[int, list[dict[str, str]]]:
    candidates = [results_dir / clip / "top_tubes.csv"]
    if not candidates[0].exists():
        for path in results_dir.glob("*/top_tubes.csv"):
            if clip_matches(clip, path.parent.name):
                candidates.append(path)
                break
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        rank = int(safe_float(row.get("rank"), 999))
        if rank > max_rank:
            continue
        frame = int(safe_float(row.get("frame"), -1))
        if frame >= 0:
            by_frame[frame].append(row)
    return by_frame


class FrameReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        self.last_frame_no = -1
        self.last_frame: np.ndarray | None = None

    def read(self, frame_no: int) -> np.ndarray | None:
        if self.last_frame_no == frame_no and self.last_frame is not None:
            return self.last_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_frame_no = frame_no
        self.last_frame = frame
        return frame

    def close(self) -> None:
        self.cap.release()


def background_metrics(frame: np.ndarray | None, label: dict[str, str]) -> dict[str, Any]:
    if frame is None:
        return {
            "bg_split": "unknown",
            "bg_sky_like": "",
            "bg_mean": "",
            "bg_std": "",
            "bg_grad": "",
            "bg_texture": "",
        }
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray_f = gray.astype(np.float32)
    h_img, w_img = gray.shape[:2]

    if all(label.get(k, "") != "" for k in ("orig_x", "orig_y", "orig_w", "orig_h")):
        x, y, w, h = row_bbox(label, "orig_")
    else:
        x, y, w, h = row_bbox(label)
        # Most detector labels are half-resolution. If no original bbox exists,
        # use a conservative 2x conversion for background inspection.
        x, y, w, h = 2.0 * x, 2.0 * y, 2.0 * w, 2.0 * h

    cx = int(round(x + 0.5 * w))
    cy = int(round(y + 0.5 * h))
    inner = max(4, int(round(1.2 * max(w, h))))
    outer = max(inner + 10, int(round(5.5 * max(w, h))))
    x0 = max(0, cx - outer)
    x1 = min(w_img, cx + outer + 1)
    y0 = max(0, cy - outer)
    y1 = min(h_img, cy + outer + 1)
    if x1 <= x0 or y1 <= y0:
        return {"bg_split": "unknown", "bg_sky_like": "", "bg_mean": "", "bg_std": "", "bg_grad": "", "bg_texture": ""}

    patch = gray_f[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    ring = (dist2 >= inner * inner) & (dist2 <= outer * outer)
    if not np.any(ring):
        ring = np.ones(patch.shape, dtype=bool)

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    vals = patch[ring]
    grad_vals = grad[y0:y1, x0:x1][ring]
    mean_bg = float(np.mean(vals))
    std_bg = float(np.std(vals))
    grad_bg = float(np.mean(grad_vals))
    texture = 0.55 * std_bg + 0.45 * grad_bg

    bright = np.clip((mean_bg - 95.0) / 75.0, 0.0, 1.0)
    smooth = np.clip((38.0 - std_bg) / 32.0, 0.0, 1.0)
    low_grad = np.clip((38.0 - grad_bg) / 38.0, 0.0, 1.0)
    sky_like = float(bright * smooth * low_grad)
    if sky_like >= 0.25 and texture < 35.0:
        split = "clean_sky"
    elif sky_like < 0.10 or texture >= 45.0:
        split = "textured_non_sky"
    else:
        split = "boundary_mixed"
    return {
        "bg_split": split,
        "bg_sky_like": round(sky_like, 4),
        "bg_mean": round(mean_bg, 3),
        "bg_std": round(std_bg, 3),
        "bg_grad": round(grad_bg, 3),
        "bg_texture": round(texture, 3),
    }


def summarize(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    groups["ALL"] = rows
    out: list[dict[str, Any]] = []
    for name, group in sorted(groups.items()):
        n = len(group)
        if n == 0:
            continue
        oracle = sum(bool(r["oracle_hit"]) for r in group)
        strict = sum(bool(r["strict_hit"]) for r in group)
        loose = sum(bool(r["loose_hit"]) for r in group)
        selected = sum(bool(r["has_selection"]) for r in group)
        out.append(
            {
                key: name,
                "frames": n,
                "selected_frames": selected,
                "oracle_hit": oracle,
                "oracle_recall": round(oracle / n, 4),
                "strict_hit": strict,
                "strict_recall": round(strict / n, 4),
                "loose_hit": loose,
                "loose_recall": round(loose / n, 4),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    labels = [r for r in read_csv(Path(args.labels)) if r.get("visible", "1") != "0"]
    labels.sort(key=lambda r: (r.get("clip", ""), int(safe_float(r.get("frame"), 0))))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = parse_video_map(args.video)
    readers: dict[Path, FrameReader] = {}
    top_cache: dict[str, dict[int, list[dict[str, str]]]] = {}

    rows_out: list[dict[str, Any]] = []
    for lab in labels:
        clip = lab.get("clip", "")
        frame_no = int(safe_float(lab.get("frame"), -1))
        if not clip or frame_no < 0:
            continue
        if clip not in top_cache:
            top_cache[clip] = load_top_tubes(Path(args.results_dir), clip, args.max_rank)
        tubes = top_cache[clip].get(frame_no, [])
        label_bbox = row_bbox(lab)
        lx, ly, lw, lh = label_bbox

        best = None
        if tubes:
            best = max(tubes, key=lambda r: safe_float(r.get(args.score_column), -1e9))
        best_bbox = row_bbox(best or {}) if best else (0.0, 0.0, 0.0, 0.0)
        best_dist = center_dist(best_bbox, label_bbox) if best else float("inf")
        oracle_dist = min((center_dist(row_bbox(r), label_bbox) for r in tubes), default=float("inf"))
        oracle_row = min(tubes, key=lambda r: center_dist(row_bbox(r), label_bbox)) if tubes else None

        path = video_for_clip(clip, videos)
        frame = None
        if path is not None:
            if path not in readers:
                readers[path] = FrameReader(path)
            frame = readers[path].read(frame_no)
        bg = background_metrics(frame, lab)
        rows_out.append(
            {
                "clip": clip,
                "frame": frame_no,
                "confidence": lab.get("confidence", ""),
                "label_x": round(lx, 3),
                "label_y": round(ly, 3),
                "label_w": round(lw, 3),
                "label_h": round(lh, 3),
                "label_cx": round(lx + 0.5 * lw, 3),
                "label_cy": round(ly + 0.5 * lh, 3),
                **bg,
                "has_selection": best is not None,
                "selected_rank": best.get("rank", "") if best else "",
                "selected_source": best.get("cand_source", "") if best else "",
                "selected_score": best.get(args.score_column, "") if best else "",
                "selected_dist_px": round(best_dist, 3) if math.isfinite(best_dist) else "",
                "strict_hit": best_dist <= args.center_tol_px,
                "loose_hit": best_dist <= args.loose_tol_px,
                "oracle_hit": oracle_dist <= args.center_tol_px,
                "oracle_dist_px": round(oracle_dist, 3) if math.isfinite(oracle_dist) else "",
                "oracle_rank": oracle_row.get("rank", "") if oracle_row else "",
                "oracle_source": oracle_row.get("cand_source", "") if oracle_row else "",
            }
        )

    for reader in readers.values():
        reader.close()

    write_csv(out_dir / "frame_background_audit.csv", rows_out)
    write_csv(out_dir / "split_summary.csv", summarize(rows_out, "bg_split"))
    write_csv(out_dir / "confidence_summary.csv", summarize(rows_out, "confidence"))
    hard = [
        r
        for r in rows_out
        if r["bg_split"] == "textured_non_sky" and (not r["strict_hit"] or not r["oracle_hit"])
    ]
    write_csv(out_dir / "textured_non_sky_failures.csv", hard)
    (out_dir / "README.md").write_text(
        "# Background Split Audit\n\n"
        f"Labels: `{args.labels}`\n\n"
        f"Results dir: `{args.results_dir}`\n\n"
        "See `split_summary.csv`, `frame_background_audit.csv`, and "
        "`textured_non_sky_failures.csv`.\n"
    )
    print(out_dir / "split_summary.csv")


if __name__ == "__main__":
    main()
