#!/usr/bin/env python3
"""Render a video overlay of current tube-verifier selections."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--selections", required=True)
    p.add_argument("--clip", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--trail", type=int, default=20)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--title", default="Current learned tube verifier")
    return p.parse_args()


def put_text(img, text: str, xy: tuple[int, int], scale: float = 0.5, color=(255, 255, 255), thick: int = 1) -> None:
    x, y = xy
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_status_bar(frame, lines: list[str]) -> None:
    h, w = frame.shape[:2]
    bar_h = 48
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for i, line in enumerate(lines[:2]):
        put_text(frame, line, (8, 18 + i * 20), scale=0.48, color=(255, 255, 255))


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    clip = args.clip or video_path.stem
    selections = pd.read_csv(args.selections)
    selections = selections[selections["clip"].eq(clip)].copy()
    if selections.empty:
        raise SystemExit(f"no selections for clip {clip}")
    by_frame = {int(row.frame): row for row in selections.itertuples(index=False)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"failed to open {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    out_fps = float(args.fps or src_fps)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_w = int(round(src_w * args.downscale))
    out_h = int(round(src_h * args.downscale))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"failed to write {out_path}")

    trail: list[tuple[int, int]] = []
    frames = 0
    selected_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if args.downscale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        row = by_frame.get(frame_no)
        if row is not None:
            score = float(row.learned_score)
            threshold = float(row.threshold)
            selected = int(row.selected) == 1
        else:
            score = 0.0
            threshold = 0.0
            selected = False

        if selected and row is not None:
            selected_frames += 1
            x = int(round(float(row.x)))
            y = int(round(float(row.y)))
            bw = int(round(float(row.w)))
            bh = int(round(float(row.h)))
            cx = x + bw // 2
            cy = y + bh // 2
            trail.append((cx, cy))
            if len(trail) > args.trail:
                trail = trail[-args.trail :]
            for a, b in zip(trail, trail[1:]):
                cv2.line(frame, a, b, (0, 180, 255), 1, cv2.LINE_AA)
            pad = max(3, int(max(bw, bh) * 1.4))
            cv2.rectangle(frame, (max(0, x - pad), max(0, y - pad)), (min(out_w - 1, x + bw + pad), min(out_h - 1, y + bh + pad)), (0, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            put_text(frame, f"r{int(row.rank)} {score:.2f}", (x, max(62, y - 7)), scale=0.42, color=(0, 255, 0), thick=1)
            status = f"DETECTION rank {int(row.rank)} score {score:.2f} source {row.source}"
        else:
            status = f"no box score {score:.2f} threshold {threshold:.2f}"

        draw_status_bar(
            frame,
            [
                f"{args.title} | {clip[:8]} | {src_fps:.1f} fps source | frame {frame_no}",
                status,
            ],
        )
        writer.write(frame)
        frames += 1

    cap.release()
    writer.release()
    print(out_path)
    print(f"frames={frames} selected_frames={selected_frames} fps={src_fps:.3f} output_fps={out_fps:.3f}")


if __name__ == "__main__":
    main()
