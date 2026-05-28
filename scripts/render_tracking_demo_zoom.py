#!/usr/bin/env python3
"""Render a tracking demo with a full-frame view and zoomed crop panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--selections", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start_sec", type=float, default=0.0)
    p.add_argument("--duration_sec", type=float, default=None)
    p.add_argument("--selection_scale", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--title", default="Current supported lock")
    p.add_argument("--trail", type=int, default=45)
    return p.parse_args()


def put_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55, color=(255, 255, 255)) -> None:
    x, y = xy
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def crop_zoom(frame: np.ndarray, cx: int, cy: int, size: int, out_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    args = parse_args()
    sel = pd.read_csv(args.selections)
    sel = sel[sel["clip"].eq(args.clip)].copy()
    by_frame = {int(r.frame): r for r in sel.itertuples(index=False)}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    out_fps = float(args.fps or src_fps)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    start_frame = max(0, int(round(args.start_sec * src_fps)))
    end_frame = None if args.duration_sec is None else start_frame + int(round(args.duration_sec * src_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    panel_w = 360
    out_w = src_w + panel_w
    out_h = src_h
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"cannot write {out_path}")

    trail: list[tuple[int, int]] = []
    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if end_frame is not None and pos >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        row = by_frame.get(frame_no)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, :src_w] = frame
        panel = canvas[:, src_w:]
        panel[:] = (24, 24, 24)

        selected = row is not None and int(row.selected) == 1
        if selected:
            x = int(round(float(row.x) * args.selection_scale))
            y = int(round(float(row.y) * args.selection_scale))
            bw = max(2, int(round(float(row.w) * args.selection_scale)))
            bh = max(2, int(round(float(row.h) * args.selection_scale)))
            cx = x + bw // 2
            cy = y + bh // 2
            trail.append((cx, cy))
            trail = trail[-args.trail :]
            for a, b in zip(trail, trail[1:]):
                cv2.line(canvas[:, :src_w], a, b, (0, 190, 255), 2, cv2.LINE_AA)
            pad = max(10, 3 * max(bw, bh))
            cv2.rectangle(canvas, (max(0, x - pad), max(0, y - pad)), (min(src_w - 1, x + bw + pad), min(src_h - 1, y + bh + pad)), (0, 255, 255), 2)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            put_text(canvas, f"rank {int(row.rank)} score {float(row.learned_score):.1f}", (x, max(48, y - 10)), 0.52, (0, 255, 0))

            zoom = crop_zoom(frame, cx, cy, 72, 320)
            cv2.rectangle(zoom, (150, 150), (170, 170), (0, 255, 0), 2)
            panel[56:376, 20:340] = zoom
            put_text(panel, "zoomed candidate crop", (20, 34), 0.55, (255, 255, 255))
            put_text(panel, f"frame {frame_no}  t={frame_no / src_fps:.2f}s", (20, 408), 0.52, (220, 220, 220))
            put_text(panel, "green = selected box", (20, 438), 0.52, (0, 255, 0))
            put_text(panel, "yellow = search context", (20, 468), 0.52, (0, 255, 255))
        else:
            put_text(panel, "no selected box", (20, 220), 0.7, (210, 210, 210))
            put_text(panel, f"frame {frame_no}  t={frame_no / src_fps:.2f}s", (20, 256), 0.52, (180, 180, 180))

        cv2.rectangle(canvas, (0, 0), (out_w, 38), (0, 0, 0), -1)
        put_text(canvas, f"{args.title} | {args.clip[:8]} | source {src_fps:.1f} fps", (10, 25), 0.58, (255, 255, 255))
        writer.write(canvas)

    cap.release()
    writer.release()
    print(out_path)


if __name__ == "__main__":
    main()

