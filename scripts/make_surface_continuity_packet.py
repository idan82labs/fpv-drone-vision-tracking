#!/usr/bin/env python3
"""Create continuous non-sky visual review packets from dense XY labels.

Unlike make_surface_vision_packet.py, this keeps neighboring frames together so
motion continuity can be inspected before adding labels to the surface ranker.
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
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--video", action="append", default=[], help="CLIP=/absolute/path/video.mp4")
    p.add_argument("--clip", default="", help="Optional clip id/prefix to select from multi-clip CSVs.")
    p.add_argument("--split", default="textured_non_sky")
    p.add_argument("--start_frame", type=int, required=True)
    p.add_argument("--end_frame", type=int, required=True)
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=90)
    p.add_argument("--crop_half_w", type=int, default=70, help="full-resolution crop half width")
    p.add_argument("--crop_half_h", type=int, default=55, help="full-resolution crop half height")
    p.add_argument("--tile_cols", type=int, default=5)
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


def first_value(row: dict[str, str], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in (None, ""):
            return str(value)
    return default


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
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    def read(self, frame_no: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


def label_box_fullres(row: dict[str, str]) -> tuple[int, int, int, int]:
    # Dense ground labels are detector-space coordinates. Older packets used
    # label_*; newer audit/eval CSVs use det_*; raw candidate tables use x/y.
    x = 2.0 * safe_float(first_value(row, ("label_x", "det_x", "x")), 0.0)
    y = 2.0 * safe_float(first_value(row, ("label_y", "det_y", "y")), 0.0)
    w = 2.0 * safe_float(first_value(row, ("label_w", "det_w", "w"), "1.0"), 1.0)
    h = 2.0 * safe_float(first_value(row, ("label_h", "det_h", "h"), "1.0"), 1.0)
    return int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))


def draw_label(img: np.ndarray, row: dict[str, str], color: tuple[int, int, int]) -> None:
    x, y, w, h = label_box_fullres(row)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cx, cy = x + w // 2, y + h // 2
    cv2.drawMarker(img, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)


def crop_around_label(frame: np.ndarray, row: dict[str, str], half_w: int, half_h: int) -> np.ndarray:
    x, y, w, h = label_box_fullres(row)
    cx, cy = x + w // 2, y + h // 2
    h_img, w_img = frame.shape[:2]
    x0 = max(0, cx - half_w)
    x1 = min(w_img, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(h_img, cy + half_h)
    crop = frame[y0:y1, x0:x1].copy()
    local = dict(row)
    local["label_x"] = str((x - x0) / 2.0)
    local["label_y"] = str((y - y0) / 2.0)
    local["label_w"] = str(w / 2.0)
    local["label_h"] = str(h / 2.0)
    draw_label(crop, local, (255, 255, 0))
    return crop


def put_title(img: np.ndarray, title: str, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def contact_sheet(paths: list[Path], out_path: Path, cols: int, thumb_w: int) -> None:
    imgs: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = thumb_w / max(1, img.shape[1])
        thumb = cv2.resize(img, (thumb_w, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        imgs.append(thumb)
    if not imgs:
        return
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    rows = int(math.ceil(len(imgs) / max(1, cols)))
    canvas = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        rr, cc = divmod(idx, cols)
        canvas[rr * h : rr * h + img.shape[0], cc * w : cc * w + img.shape[1]] = img
    cv2.imwrite(str(out_path), canvas)


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        frame = int(safe_float(row.get("frame"), -1))
        if args.clip and not clip_matches(str(row.get("clip", "")), args.clip):
            continue
        if row.get("bg_split") != args.split:
            continue
        if frame < args.start_frame or frame > args.end_frame:
            continue
        if (frame - args.start_frame) % max(1, args.frame_step) != 0:
            continue
        selected.append(row)
    selected.sort(key=lambda r: (r.get("clip", ""), int(safe_float(r.get("frame"), -1))))
    return selected[: args.max_frames]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "frames_clean"
    diag_dir = out_dir / "frames_diagnostic"
    crop_dir = out_dir / "crops_diagnostic"
    clean_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    videos = parse_video_map(args.video)
    rows = select_rows(read_csv(Path(args.audit_csv)), args)
    readers: dict[Path, FrameReader] = {}
    clean_paths: list[Path] = []
    diag_paths: list[Path] = []
    crop_paths: list[Path] = []
    packet: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        clip = row.get("clip", "")
        frame_no = int(safe_float(row.get("frame"), -1))
        video_path = video_for_clip(clip, videos)
        if video_path is None or frame_no < 0:
            continue
        if video_path not in readers:
            readers[video_path] = FrameReader(video_path)
        reader = readers[video_path]
        frame = reader.read(frame_no)
        if frame is None:
            continue

        short = clip.split("-")[0]
        title = (
            f"{idx:03d} {short} f{frame_no} t={frame_no / reader.fps:.2f}s "
            f"{row.get('bg_split','')} strict={row.get('strict_hit','')}"
        )
        stem = f"{idx:03d}_{short}_f{frame_no:05d}"

        clean = frame.copy()
        put_title(clean, title)
        diag = frame.copy()
        draw_label(diag, row, (255, 255, 0))
        put_title(diag, title, (0, 255, 255))
        crop = crop_around_label(frame, row, args.crop_half_w, args.crop_half_h)
        put_title(crop, title, (0, 255, 255))

        clean_path = clean_dir / f"{stem}.jpg"
        diag_path = diag_dir / f"{stem}_diag.jpg"
        crop_path = crop_dir / f"{stem}_crop.jpg"
        cv2.imwrite(str(clean_path), clean)
        cv2.imwrite(str(diag_path), diag)
        cv2.imwrite(str(crop_path), crop)
        clean_paths.append(clean_path)
        diag_paths.append(diag_path)
        crop_paths.append(crop_path)

        packet.append(
            {
                "clip": clip,
                "frame": frame_no,
                "time_s": round(frame_no / reader.fps, 3) if reader.fps else "",
                "bg_split": row.get("bg_split", ""),
                "source_confidence": row.get("confidence", ""),
                "det_x": first_value(row, ("label_x", "det_x", "x")),
                "det_y": first_value(row, ("label_y", "det_y", "y")),
                "det_w": first_value(row, ("label_w", "det_w", "w")),
                "det_h": first_value(row, ("label_h", "det_h", "h")),
                "visible": "true",
                "vision_confidence": "",
                "status": "needs_vision_review",
                "strict_hit": row.get("strict_hit", ""),
                "loose_hit": row.get("loose_hit", ""),
                "oracle_hit": row.get("oracle_hit", ""),
                "oracle_rank": row.get("oracle_rank", ""),
                "selected_dist_px": row.get("selected_dist_px", ""),
                "clean_image": str(clean_path),
                "diagnostic_image": str(diag_path),
                "crop_image": str(crop_path),
                "notes": "",
            }
        )

    for reader in readers.values():
        reader.close()

    write_csv(out_dir / "surface_continuity_label_template.csv", packet)
    contact_sheet(clean_paths, out_dir / "contact_clean.jpg", max(1, args.tile_cols), 330)
    contact_sheet(diag_paths, out_dir / "contact_diagnostic.jpg", max(1, args.tile_cols), 330)
    contact_sheet(crop_paths, out_dir / "contact_crops_diagnostic.jpg", max(1, args.tile_cols), 240)
    (out_dir / "README.md").write_text(
        "# Surface Continuity Packet\n\n"
        "This packet is for continuity review of non-sky target labels. The CSV "
        "`surface_continuity_label_template.csv` contains detector-space label "
        "coordinates copied from the dense seed labels. Only set "
        "`vision_confidence` after visual review of the clean frame, diagnostic "
        "frame, and crop sheet.\n"
    )
    print(out_dir / "surface_continuity_label_template.csv")
    print(out_dir / "contact_diagnostic.jpg")
    print(out_dir / "contact_crops_diagnostic.jpg")


if __name__ == "__main__":
    main()
