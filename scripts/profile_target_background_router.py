#!/usr/bin/env python3
"""Profile target labels by immediate and surrounding background context.

The first surface split was intentionally simple and over-classified many
skyline-adjacent targets as textured/non-sky. This router looks at a close
annulus around the target separately from a wider ring so labels can be split
into cleaner states:

- clean_sky
- sky_target_near_surface
- boundary_mixed
- surface_backed
- unknown
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
    p.add_argument("--out_dir", required=True)
    p.add_argument("--video", action="append", default=[], help="CLIP=/absolute/path/video.mp4")
    p.add_argument("--confidence", nargs="*", default=["high", "medium_high"])
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


class FrameReader:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        self.last_frame = -1
        self.frame: np.ndarray | None = None

    def read(self, frame_no: int) -> np.ndarray | None:
        if frame_no == self.last_frame and self.frame is not None:
            return self.frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no))
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_frame = frame_no
        self.frame = frame
        return frame

    def close(self) -> None:
        self.cap.release()


def label_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    if all(row.get(k, "") not in ("", None) for k in ("orig_x", "orig_y", "orig_w", "orig_h")):
        return tuple(safe_float(row[k]) for k in ("orig_x", "orig_y", "orig_w", "orig_h"))  # type: ignore[return-value]
    return (
        2.0 * safe_float(row.get("det_x", row.get("label_x", row.get("x", 0.0)))),
        2.0 * safe_float(row.get("det_y", row.get("label_y", row.get("y", 0.0)))),
        2.0 * safe_float(row.get("det_w", row.get("label_w", row.get("w", 1.0))), 1.0),
        2.0 * safe_float(row.get("det_h", row.get("label_h", row.get("h", 1.0))), 1.0),
    )


def ring_stats(gray: np.ndarray, grad: np.ndarray, cx: int, cy: int, inner: float, outer: float) -> dict[str, float]:
    h_img, w_img = gray.shape[:2]
    x0 = max(0, int(math.floor(cx - outer)))
    x1 = min(w_img, int(math.ceil(cx + outer + 1)))
    y0 = max(0, int(math.floor(cy - outer)))
    y1 = min(h_img, int(math.ceil(cy + outer + 1)))
    if x1 <= x0 or y1 <= y0:
        return {"mean": 0.0, "std": 0.0, "grad": 0.0, "texture": 0.0, "sky_like": 0.0}
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    mask = (dist2 >= inner * inner) & (dist2 <= outer * outer)
    if not np.any(mask):
        return {"mean": 0.0, "std": 0.0, "grad": 0.0, "texture": 0.0, "sky_like": 0.0}
    patch = gray[y0:y1, x0:x1]
    gpatch = grad[y0:y1, x0:x1]
    vals = patch[mask]
    gvals = gpatch[mask]
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    grad_mean = float(np.mean(gvals))
    texture = 0.55 * std + 0.45 * grad_mean
    bright = np.clip((mean - 95.0) / 75.0, 0.0, 1.0)
    smooth = np.clip((38.0 - std) / 32.0, 0.0, 1.0)
    low_grad = np.clip((38.0 - grad_mean) / 38.0, 0.0, 1.0)
    sky_like = float(bright * smooth * low_grad)
    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "grad": round(grad_mean, 3),
        "texture": round(texture, 3),
        "sky_like": round(sky_like, 4),
    }


def classify(close: dict[str, float], far: dict[str, float]) -> str:
    close_sky = close["sky_like"] >= 0.22 and close["texture"] < 32.0
    far_sky = far["sky_like"] >= 0.18 and far["texture"] < 38.0
    close_surface = close["sky_like"] < 0.08 and close["texture"] >= 38.0
    far_surface = far["sky_like"] < 0.12 and far["texture"] >= 42.0
    if close_sky and far_sky:
        return "clean_sky"
    if close_sky and far_surface:
        return "sky_target_near_surface"
    if close_surface:
        return "surface_backed"
    if far_surface or abs(close["texture"] - far["texture"]) >= 18.0:
        return "boundary_mixed"
    return "unknown"


def profile(frame: np.ndarray | None, row: dict[str, str]) -> dict[str, Any]:
    if frame is None:
        return {"router_state": "unknown"}
    gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = gray_u8.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    x, y, w, h = label_bbox(row)
    cx = int(round(x + 0.5 * w))
    cy = int(round(y + 0.5 * h))
    radius = max(4.0, max(w, h))
    close = ring_stats(gray, grad, cx, cy, 1.4 * radius, 3.2 * radius)
    far = ring_stats(gray, grad, cx, cy, 3.8 * radius, 7.0 * radius)
    state = classify(close, far)
    return {
        "router_state": state,
        "close_sky_like": close["sky_like"],
        "close_texture": close["texture"],
        "close_mean": close["mean"],
        "close_std": close["std"],
        "close_grad": close["grad"],
        "far_sky_like": far["sky_like"],
        "far_texture": far["texture"],
        "far_mean": far["mean"],
        "far_std": far["std"],
        "far_grad": far["grad"],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = parse_video_map(args.video)
    readers: dict[Path, FrameReader] = {}
    allowed_conf = set(args.confidence)
    rows = [r for r in read_csv(Path(args.labels)) if r.get("visible", "1") != "0"]
    if allowed_conf:
        rows = [r for r in rows if r.get("confidence") in allowed_conf]

    profiled: list[dict[str, Any]] = []
    for row in rows:
        clip = row.get("clip", "")
        frame_no = int(safe_float(row.get("frame"), -1))
        path = video_for_clip(clip, videos)
        frame = None
        if path is not None and frame_no >= 0:
            if path not in readers:
                readers[path] = FrameReader(path)
            frame = readers[path].read(frame_no)
        profiled.append({**row, **profile(frame, row)})
    for reader in readers.values():
        reader.close()

    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_clip_state: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profiled:
        state = row.get("router_state", "unknown")
        by_state[state].append(row)
        by_clip_state[(row.get("clip", ""), state)].append(row)
    summary = [
        {"router_state": state, "frames": len(state_rows)}
        for state, state_rows in sorted(by_state.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    clip_summary = [
        {"clip": clip, "router_state": state, "frames": len(state_rows)}
        for (clip, state), state_rows in sorted(by_clip_state.items())
    ]
    write_csv(out_dir / "background_router_profile.csv", profiled)
    write_csv(out_dir / "router_summary.csv", summary)
    write_csv(out_dir / "router_clip_summary.csv", clip_summary)
    print(out_dir / "router_summary.csv")


if __name__ == "__main__":
    main()
