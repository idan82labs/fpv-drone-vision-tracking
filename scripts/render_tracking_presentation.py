#!/usr/bin/env python3
"""Render a cleaner tracking presentation video from selection CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--selections", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start_sec", type=float, default=0.0)
    p.add_argument("--duration_sec", type=float, default=None)
    p.add_argument("--selection_scale", type=float, default=2.0)
    p.add_argument("--max_fill_gap", type=int, default=8)
    p.add_argument("--smooth_window", type=int, default=7)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--title", default="Current tracking status")
    return p.parse_args()


def draw_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    x, y = xy
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def selected_rows(path: Path, clip: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["clip"].eq(clip)].copy()
    for col in ["selected", "x", "y", "w", "h", "rank", "learned_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    df = df[df["frame"].notna()].copy()
    df["frame"] = df["frame"].astype(int)
    return df.sort_values("frame")


def build_smoothed_boxes(df: pd.DataFrame, scale: float, max_fill_gap: int, smooth_window: int) -> dict[int, dict[str, float | str]]:
    selected = df[df["selected"].fillna(0).astype(int).eq(1)].copy()
    selected = selected.dropna(subset=["x", "y", "w", "h"])
    if selected.empty:
        return {}

    boxes: dict[int, dict[str, float | str]] = {}
    for row in selected.itertuples(index=False):
        boxes[int(row.frame)] = {
            "x": float(row.x) * scale,
            "y": float(row.y) * scale,
            "w": float(row.w) * scale,
            "h": float(row.h) * scale,
            "source": str(getattr(row, "source", "")),
            "score": float(getattr(row, "learned_score", 0.0) or 0.0),
            "rank": float(getattr(row, "rank", 0.0) or 0.0),
            "mode": "measured",
        }

    frames = sorted(boxes)
    for left, right in zip(frames, frames[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_fill_gap:
            continue
        left_box = boxes[left]
        right_box = boxes[right]
        for frame in range(left + 1, right):
            alpha = (frame - left) / (right - left)
            boxes[frame] = {
                key: (1.0 - alpha) * float(left_box[key]) + alpha * float(right_box[key])
                for key in ["x", "y", "w", "h"]
            }
            boxes[frame].update({"source": "interpolated", "score": 0.0, "rank": 0.0, "mode": "interpolated"})

    if smooth_window > 1:
        half = smooth_window // 2
        all_frames = sorted(boxes)
        raw = {f: dict(boxes[f]) for f in all_frames}
        for frame in all_frames:
            local = [f for f in range(frame - half, frame + half + 1) if f in raw]
            if len(local) < 3:
                continue
            # Only smooth inside dense local runs; do not drag boxes across long losses.
            if max(local) - min(local) > smooth_window + 2:
                continue
            for key in ["x", "y", "w", "h"]:
                boxes[frame][key] = float(np.median([float(raw[f][key]) for f in local]))

    return boxes


def crop_panel(frame: np.ndarray, box: dict[str, float | str], out_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    cx = int(round(float(box["x"]) + 0.5 * float(box["w"])))
    cy = int(round(float(box["y"]) + 0.5 * float(box["h"])))
    size = 84
    x0 = max(0, cx - size // 2)
    y0 = max(0, cy - size // 2)
    x1 = min(w, cx + size // 2)
    y1 = min(h, cy + size // 2)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    zoom = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    center = out_size // 2
    cv2.rectangle(zoom, (center - 12, center - 12), (center + 12, center + 12), (0, 255, 0), 2)
    return zoom


def main() -> None:
    args = parse_args()
    df = selected_rows(Path(args.selections), args.clip)
    boxes = build_smoothed_boxes(df, args.selection_scale, args.max_fill_gap, args.smooth_window)

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

    panel_w = 340
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
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t = frame_no / src_fps
        box = boxes.get(frame_no)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, :src_w] = frame
        panel = canvas[:, src_w:]
        panel[:] = (18, 18, 18)

        cv2.rectangle(canvas, (0, 0), (out_w, 42), (0, 0, 0), -1)
        draw_text(canvas, f"{args.title} | {t:05.2f}s | {src_fps:.1f} fps source", (12, 28), 0.58)

        if box is not None:
            x = int(round(float(box["x"])))
            y = int(round(float(box["y"])))
            bw = max(4, int(round(float(box["w"]))))
            bh = max(4, int(round(float(box["h"]))))
            cx = x + bw // 2
            cy = y + bh // 2
            trail.append((cx, cy))
            trail = trail[-45:]
            for i, (a, b) in enumerate(zip(trail, trail[1:])):
                alpha = (i + 1) / max(1, len(trail))
                color = (0, int(120 + 110 * alpha), int(150 + 80 * alpha))
                cv2.line(canvas[:, :src_w], a, b, color, 2, cv2.LINE_AA)

            pad = 18
            cv2.rectangle(canvas, (max(0, x - pad), max(43, y - pad)), (min(src_w - 1, x + bw + pad), min(src_h - 1, y + bh + pad)), (0, 230, 255), 2)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            mode = str(box.get("mode", "measured"))
            label = "TRACK" if mode == "measured" else "FILL"
            draw_text(canvas, label, (max(6, x - 2), max(64, y - 10)), 0.55, (0, 255, 0), 1)

            zoom = crop_panel(frame, box, 292)
            panel[62:354, 24:316] = zoom
            draw_text(panel, "zoom crop", (24, 36), 0.62)
            draw_text(panel, f"frame {frame_no}", (24, 392), 0.55, (220, 220, 220))
            draw_text(panel, "green box = selected target", (24, 424), 0.52, (0, 255, 0))
            if mode == "interpolated":
                draw_text(panel, "short gap interpolated", (24, 454), 0.52, (0, 220, 255))
            else:
                draw_text(panel, "measured candidate", (24, 454), 0.52, (220, 220, 220))
        else:
            trail = []
            draw_text(panel, "NO CONFIDENT TRACK", (28, 214), 0.72, (200, 200, 200), 2)
            draw_text(panel, f"frame {frame_no}", (28, 250), 0.55, (170, 170, 170))
            draw_text(panel, "suppressed instead of", (28, 292), 0.52, (170, 170, 170))
            draw_text(panel, "following clutter", (28, 322), 0.52, (170, 170, 170))

        writer.write(canvas)

    cap.release()
    writer.release()
    print(out_path)


if __name__ == "__main__":
    main()
