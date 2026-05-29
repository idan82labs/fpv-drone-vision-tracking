#!/usr/bin/env python3
"""Create a vision-label packet for hard non-sky drone frames.

The packet contains clean frames for unbiased XY labeling, diagnostic frames
with current detector boxes, contact sheets, and a CSV template that can be
filled by either the model or a human reviewer.
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
    p.add_argument("--split", default="textured_non_sky")
    p.add_argument("--require_failure", action="store_true")
    p.add_argument("--max_frames", type=int, default=80)
    p.add_argument("--min_frame_gap", type=int, default=5)
    p.add_argument("--include_low_confidence", action="store_true")
    p.add_argument("--tile_cols", type=int, default=4)
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
        self.path = path
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


def pick_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    filtered = [r for r in rows if r.get("bg_split") == args.split]
    if args.require_failure:
        filtered = [r for r in filtered if r.get("strict_hit") != "True" or r.get("oracle_hit") != "True"]
    if not args.include_low_confidence:
        filtered = [r for r in filtered if r.get("confidence") != "low_review_required"]

    def priority(row: dict[str, str]) -> tuple[int, float, int]:
        # Prefer frames where the proposal exists but ranking failed, then true
        # proposal misses. These are most useful for non-sky ranker training.
        oracle = row.get("oracle_hit") == "True"
        strict = row.get("strict_hit") == "True"
        pri = 0 if (oracle and not strict) else 1 if not oracle else 2
        dist = safe_float(row.get("selected_dist_px"), 999.0)
        return (pri, -dist, int(safe_float(row.get("frame"), 0)))

    filtered.sort(key=priority)
    picked: list[dict[str, str]] = []
    last_by_clip: dict[str, int] = {}
    for row in filtered:
        clip = row.get("clip", "")
        frame = int(safe_float(row.get("frame"), -1))
        if frame < 0:
            continue
        if clip in last_by_clip and abs(frame - last_by_clip[clip]) < args.min_frame_gap:
            continue
        picked.append(row)
        last_by_clip[clip] = frame
        if len(picked) >= args.max_frames:
            break
    return picked


def draw_box(img: np.ndarray, row: dict[str, str], prefix: str, color: tuple[int, int, int], label: str) -> None:
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_w", f"{prefix}_h")
    if not all(row.get(k, "") not in ("", None) for k in keys):
        return
    x, y, w, h = [safe_float(row.get(k), 0.0) for k in keys]
    # Audit rows are in detector coordinates. Draw on full-res frame by 2x.
    x, y, w, h = 2.0 * x, 2.0 * y, 2.0 * w, 2.0 * h
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, p1, p2, color, 2)
    cv2.putText(img, label, (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def make_contact_sheet(paths: list[Path], out_path: Path, cols: int, thumb_w: int = 360) -> None:
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
    rows = int(math.ceil(len(imgs) / cols))
    canvas = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * h : r * h + img.shape[0], c * w : c * w + img.shape[1]] = img
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "frames_clean"
    diag_dir = out_dir / "frames_diagnostic"
    clean_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    videos = parse_video_map(args.video)
    rows = pick_rows(read_csv(Path(args.audit_csv)), args)

    readers: dict[Path, FrameReader] = {}
    packet: list[dict[str, Any]] = []
    clean_paths: list[Path] = []
    diag_paths: list[Path] = []
    for idx, row in enumerate(rows, start=1):
        clip = row.get("clip", "")
        frame_no = int(safe_float(row.get("frame"), -1))
        path = video_for_clip(clip, videos)
        if path is None or frame_no < 0:
            continue
        if path not in readers:
            readers[path] = FrameReader(path)
        reader = readers[path]
        frame = reader.read(frame_no)
        if frame is None:
            continue
        short = clip.split("-")[0]
        stem = f"{idx:03d}_{short}_f{frame_no:05d}"
        clean = frame.copy()
        diag = frame.copy()
        title = f"{idx:03d} {short} f{frame_no} t={frame_no / reader.fps:.2f}s {row.get('confidence','')}"
        cv2.putText(clean, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(clean, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        draw_box(diag, row, "label", (255, 255, 0), "known/seed")
        # selected/oracle centers are not boxes in the audit CSV, but the row
        # metadata is still useful for prioritization.
        cv2.putText(diag, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        clean_path = clean_dir / f"{stem}.jpg"
        diag_path = diag_dir / f"{stem}_diag.jpg"
        cv2.imwrite(str(clean_path), clean)
        cv2.imwrite(str(diag_path), diag)
        clean_paths.append(clean_path)
        diag_paths.append(diag_path)
        packet.append(
            {
                "packet_id": idx,
                "clip": clip,
                "frame": frame_no,
                "time_s": round(frame_no / reader.fps, 3) if reader.fps else "",
                "bg_split": row.get("bg_split", ""),
                "confidence": row.get("confidence", ""),
                "clean_image": str(clean_path),
                "diagnostic_image": str(diag_path),
                "status": "unlabeled",
                "det_x": "",
                "det_y": "",
                "det_w": "",
                "det_h": "",
                "visible": "",
                "vision_confidence": "",
                "notes": "",
                "seed_x": row.get("label_x", ""),
                "seed_y": row.get("label_y", ""),
                "seed_w": row.get("label_w", ""),
                "seed_h": row.get("label_h", ""),
                "selected_dist_px": row.get("selected_dist_px", ""),
                "oracle_hit": row.get("oracle_hit", ""),
                "oracle_rank": row.get("oracle_rank", ""),
                "oracle_source": row.get("oracle_source", ""),
            }
        )
    for reader in readers.values():
        reader.close()

    write_csv(out_dir / "vision_label_template.csv", packet)
    make_contact_sheet(clean_paths, out_dir / "contact_clean.jpg", max(1, args.tile_cols))
    make_contact_sheet(diag_paths, out_dir / "contact_diagnostic.jpg", max(1, args.tile_cols))
    (out_dir / "README.md").write_text(
        "# Surface Vision Label Packet\n\n"
        "Label from `frames_clean/` first. Use `frames_diagnostic/` only to audit "
        "current detector behavior after choosing the target box.\n\n"
        "Fill `vision_label_template.csv` columns `det_x`, `det_y`, `det_w`, "
        "`det_h`, `visible`, `vision_confidence`, and `notes`. Coordinates are "
        "detector-space half-resolution, matching the ranker labels.\n"
    )
    print(out_dir / "vision_label_template.csv")
    print(out_dir / "contact_clean.jpg")


if __name__ == "__main__":
    main()
