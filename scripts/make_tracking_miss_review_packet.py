#!/usr/bin/env python3
"""Build review sheets from visible detector misses.

Inputs are the output of scripts/evaluate_tracking_run.py plus the source
video. The generated images use detector coordinates, so any reviewed box can
be copied back into training labels without another scale conversion.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--misses", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--max_frames", type=int, default=96)
    p.add_argument("--stride", type=int, default=1, help="Keep every Nth miss after sorting by frame.")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--crop_pad", type=int, default=42)
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


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def bbox(row: dict[str, str], prefix: str) -> tuple[float, float, float, float] | None:
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_w", f"{prefix}_h")
    if not all(str(row.get(k, "")).strip() for k in keys):
        return None
    return tuple(fnum(row[k]) for k in keys)  # type: ignore[return-value]


def draw_box(
    img: np.ndarray,
    box: tuple[float, float, float, float] | None,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 1,
) -> None:
    if box is None:
        return
    x, y, w, h = box
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, p1, p2, color, thickness)
    cv2.putText(
        img,
        label,
        (p1[0], max(12, p1[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def extract_frame(cap: cv2.VideoCapture, frame_no: int, downscale: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        return None
    if downscale != 1.0:
        frame = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return frame


def crop_around(
    img: np.ndarray,
    primary: tuple[float, float, float, float] | None,
    fallback: tuple[float, float, float, float] | None,
    pad: int,
) -> np.ndarray:
    box = primary or fallback
    if box is None:
        return img
    x, y, w, h = box
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    half = max(pad, int(round(max(w, h) * 4)))
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(img.shape[1], int(round(cx + half)))
    y1 = min(img.shape[0], int(round(cy + half)))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def label_image(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def sheet(paths: list[Path], out_path: Path, cols: int, thumb_w: int) -> None:
    imgs: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = thumb_w / max(1, img.shape[1])
        imgs.append(cv2.resize(img, (thumb_w, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    if not imgs:
        return
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    rows = int(math.ceil(len(imgs) / cols))
    canvas = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * h : r * h + img.shape[0], c * w : c * w + img.shape[1]] = img
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "frames_raw"
    diag_dir = out_dir / "frames_diagnostic"
    crop_dir = out_dir / "crops_diagnostic"
    raw_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(read_csv(Path(args.misses)), key=lambda r: int(fnum(r.get("frame"), 0)))
    rows = rows[:: max(1, args.stride)][: args.max_frames]
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    packet_rows: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    diag_paths: list[Path] = []
    crop_paths: list[Path] = []
    for idx, row in enumerate(rows, start=1):
        frame_no = int(fnum(row.get("frame"), -1))
        frame = extract_frame(cap, frame_no, args.downscale)
        if frame is None:
            continue
        label_box = bbox(row, "det")
        selected_box = bbox(row, "selected")
        t = frame_no / fps if fps else 0.0
        title = f"{idx:03d} f{frame_no} t={t:.2f}s dist={row.get('dist_px','')}"

        raw = label_image(frame, title)
        diag = raw.copy()
        draw_box(diag, label_box, (255, 255, 0), "label", 1)
        draw_box(diag, selected_box, (0, 0, 255), "selected", 1)
        crop = crop_around(diag, label_box, selected_box, args.crop_pad)
        crop = cv2.resize(crop, (220, 220), interpolation=cv2.INTER_NEAREST)

        stem = f"{idx:03d}_f{frame_no:05d}"
        raw_path = raw_dir / f"{stem}.jpg"
        diag_path = diag_dir / f"{stem}_diag.jpg"
        crop_path = crop_dir / f"{stem}_crop.jpg"
        cv2.imwrite(str(raw_path), raw)
        cv2.imwrite(str(diag_path), diag)
        cv2.imwrite(str(crop_path), crop)
        raw_paths.append(raw_path)
        diag_paths.append(diag_path)
        crop_paths.append(crop_path)
        packet_rows.append(
            {
                "packet_id": idx,
                "frame": frame_no,
                "time_s": round(t, 3) if fps else "",
                "det_x": row.get("det_x", ""),
                "det_y": row.get("det_y", ""),
                "det_w": row.get("det_w", ""),
                "det_h": row.get("det_h", ""),
                "selected": row.get("selected", ""),
                "selected_x": row.get("selected_x", ""),
                "selected_y": row.get("selected_y", ""),
                "selected_w": row.get("selected_w", ""),
                "selected_h": row.get("selected_h", ""),
                "dist_px": row.get("dist_px", ""),
                "raw_image": str(raw_path),
                "diagnostic_image": str(diag_path),
                "crop_image": str(crop_path),
                "vision_visible": "",
                "vision_x": "",
                "vision_y": "",
                "vision_w": "",
                "vision_h": "",
                "vision_confidence": "",
                "vision_notes": "",
            }
        )

    cap.release()
    write_csv(out_dir / "miss_review_index.csv", packet_rows)
    sheet(raw_paths, out_dir / "miss_sheet_raw.jpg", args.cols, 320)
    sheet(diag_paths, out_dir / "miss_sheet_diagnostic.jpg", args.cols, 320)
    sheet(crop_paths, out_dir / "miss_sheet_crops.jpg", args.cols, 220)
    print(out_dir / "miss_review_index.csv")
    print(out_dir / "miss_sheet_raw.jpg")
    print(out_dir / "miss_sheet_diagnostic.jpg")
    print(out_dir / "miss_sheet_crops.jpg")


if __name__ == "__main__":
    main()
