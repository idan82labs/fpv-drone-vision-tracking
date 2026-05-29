#!/usr/bin/env python3
"""Build visual review packets from selector-disagreement rows.

This is a router-training packet builder. It renders the frames where two
selector families disagree, so the failure can be labeled as either:

* continuous visible target where the conservative branch should not suppress;
* null/hard clutter where the permissive branch should be rejected;
* both selectors wrong, requiring a new label or feature.
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


BOX_COLORS = {
    "label": (255, 255, 0),
    "a": (0, 0, 255),
    "b": (0, 180, 255),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--disagreements", required=True)
    p.add_argument("--video_dir", default="/Users/idant/Downloads")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--selector_a_name", default="crop_score")
    p.add_argument("--selector_b_name", default="hmm_s9b70")
    p.add_argument("--categories", default="", help="Comma-separated category allowlist.")
    p.add_argument("--clips", default="", help="Comma-separated clip allowlist.")
    p.add_argument("--max_per_bucket", type=int, default=80)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--crop_pad", type=int, default=46)
    p.add_argument("--cols", type=int, default=4)
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


def fnum(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def bbox(row: dict[str, str], prefix: str, scale: float) -> tuple[float, float, float, float] | None:
    keys = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_w", f"{prefix}_h")
    vals = [fnum(row.get(key)) for key in keys]
    if any(v is None for v in vals):
        return None
    x, y, w, h = [float(v) * scale for v in vals if v is not None]
    return x, y, w, h


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
        (p1[0], max(14, p1[1] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def label_bar(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text[:150], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_frame(cap: cv2.VideoCapture, frame_no: int, downscale: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no))
    ok, frame = cap.read()
    if not ok:
        return None
    if downscale != 1.0:
        frame = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return frame


def crop_around(
    img: np.ndarray,
    boxes: list[tuple[float, float, float, float] | None],
    pad: int,
) -> np.ndarray:
    valid = [b for b in boxes if b is not None]
    if not valid:
        return img
    cx = sum(b[0] + 0.5 * b[2] for b in valid) / len(valid)
    cy = sum(b[1] + 0.5 * b[3] for b in valid) / len(valid)
    max_span = max(max(b[2], b[3]) for b in valid)
    half = max(pad, int(round(max_span * 5)))
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(img.shape[1], int(round(cx + half)))
    y1 = min(img.shape[0], int(round(cy + half)))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def make_sheet(paths: list[Path], out_path: Path, cols: int, thumb_w: int) -> None:
    imgs: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = thumb_w / max(1, img.shape[1])
        imgs.append(cv2.resize(img, (thumb_w, max(1, int(round(img.shape[0] * scale)))), interpolation=cv2.INTER_AREA))
    if not imgs:
        return
    h = max(img.shape[0] for img in imgs)
    w = max(img.shape[1] for img in imgs)
    rows = int(math.ceil(len(imgs) / max(1, cols)))
    canvas = np.full((rows * h, max(1, cols) * w, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, max(1, cols))
        canvas[r * h : r * h + img.shape[0], c * w : c * w + img.shape[1]] = img
    cv2.imwrite(str(out_path), canvas)


def allowed_set(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    diag_dir = out_dir / "frames_diagnostic"
    crop_dir = out_dir / "crops_diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    categories = allowed_set(args.categories)
    clips = allowed_set(args.clips)
    rows = read_csv(Path(args.disagreements))
    rows = [
        row
        for row in rows
        if (not categories or row.get("category", "") in categories)
        and (not clips or row.get("clip", "") in clips)
    ]
    rows.sort(key=lambda r: (r.get("clip", ""), r.get("category", ""), int(fnum(r.get("frame"), 0) or 0)))

    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("clip", ""), row.get("category", ""))].append(row)

    selected_rows: list[dict[str, str]] = []
    for bucket_rows in buckets.values():
        sampled = bucket_rows[:: max(1, args.stride)][: args.max_per_bucket]
        selected_rows.extend(sampled)

    caps: dict[str, cv2.VideoCapture] = {}
    packet_rows: list[dict[str, Any]] = []
    diag_paths_by_bucket: dict[str, list[Path]] = defaultdict(list)
    crop_paths_by_bucket: dict[str, list[Path]] = defaultdict(list)
    for idx, row in enumerate(selected_rows, start=1):
        clip = row.get("clip", "")
        frame_no = int(fnum(row.get("frame"), -1) or -1)
        video_path = Path(args.video_dir) / f"{clip}.MP4"
        if clip not in caps:
            caps[clip] = cv2.VideoCapture(str(video_path))
        cap = caps[clip]
        if not cap.isOpened():
            continue
        frame = read_frame(cap, frame_no, args.downscale)
        if frame is None:
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        t = frame_no / fps if fps else fnum(row.get("time_s"), 0.0) or 0.0

        label_box = bbox(row, "det", args.downscale)
        a_box = bbox(row, args.selector_a_name, args.downscale)
        b_box = bbox(row, args.selector_b_name, args.downscale)
        title = (
            f"{idx:03d} {clip[:8]} f{frame_no} t={t:.2f}s {row.get('category','')} "
            f"A={row.get(args.selector_a_name + '_status','')} B={row.get(args.selector_b_name + '_status','')}"
        )
        diag = label_bar(frame, title)
        draw_box(diag, label_box, BOX_COLORS["label"], "label", 2)
        draw_box(diag, a_box, BOX_COLORS["a"], "A", 1)
        draw_box(diag, b_box, BOX_COLORS["b"], "B", 1)
        crop = crop_around(diag, [label_box, a_box, b_box], args.crop_pad)
        crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)

        safe_bucket = f"{clip[:8]}_{row.get('category','unknown')}"
        stem = f"{idx:03d}_{clip[:8]}_f{frame_no:05d}_{row.get('category','unknown')}"
        diag_path = diag_dir / f"{stem}.jpg"
        crop_path = crop_dir / f"{stem}_crop.jpg"
        cv2.imwrite(str(diag_path), diag)
        cv2.imwrite(str(crop_path), crop)
        diag_paths_by_bucket[safe_bucket].append(diag_path)
        crop_paths_by_bucket[safe_bucket].append(crop_path)
        packet_rows.append(
            {
                "packet_id": idx,
                "clip": clip,
                "frame": frame_no,
                "time_s": round(t, 3),
                "category": row.get("category", ""),
                "visible": row.get("visible", ""),
                "selector_a_status": row.get(args.selector_a_name + "_status", ""),
                "selector_b_status": row.get(args.selector_b_name + "_status", ""),
                "det_x": row.get("det_x", ""),
                "det_y": row.get("det_y", ""),
                "det_w": row.get("det_w", ""),
                "det_h": row.get("det_h", ""),
                "selector_a_x": row.get(args.selector_a_name + "_x", ""),
                "selector_a_y": row.get(args.selector_a_name + "_y", ""),
                "selector_a_w": row.get(args.selector_a_name + "_w", ""),
                "selector_a_h": row.get(args.selector_a_name + "_h", ""),
                "selector_b_x": row.get(args.selector_b_name + "_x", ""),
                "selector_b_y": row.get(args.selector_b_name + "_y", ""),
                "selector_b_w": row.get(args.selector_b_name + "_w", ""),
                "selector_b_h": row.get(args.selector_b_name + "_h", ""),
                "confidence": row.get("confidence", ""),
                "source_notes": row.get("notes", ""),
                "diagnostic_image": str(diag_path),
                "crop_image": str(crop_path),
                "target_visible": "",
                "target_x": "",
                "target_y": "",
                "target_w": "",
                "target_h": "",
                "false_lock_kind": "",
                "router_label": "",
                "review_confidence": "",
                "review_notes": "",
            }
        )

    for cap in caps.values():
        cap.release()

    write_csv(out_dir / "selector_disagreement_review_index.csv", packet_rows)
    sheet_rows: list[dict[str, Any]] = []
    sheet_dir = out_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for bucket, paths in sorted(diag_paths_by_bucket.items()):
        diag_sheet = sheet_dir / f"{bucket}_diagnostic.jpg"
        crop_sheet = sheet_dir / f"{bucket}_crops.jpg"
        make_sheet(paths, diag_sheet, args.cols, 360)
        make_sheet(crop_paths_by_bucket[bucket], crop_sheet, args.cols, 240)
        sheet_rows.append({"bucket": bucket, "diagnostic_sheet": str(diag_sheet), "crop_sheet": str(crop_sheet)})
    write_csv(out_dir / "contact_sheet_index.csv", sheet_rows)
    (out_dir / "README.md").write_text(
        f"""# Selector Disagreement Review Packet

Source: `{args.disagreements}`

Legend:

- cyan/yellow label box: reviewed target box when visible.
- red A box: `{args.selector_a_name}` selection.
- orange B box: `{args.selector_b_name}` selection.

Review goal:

- For visible rows, confirm whether the label is correct and whether suppression
  was wrong.
- For null rows, classify the false lock kind: static hot spot, branch/tree,
  terrain texture, skyline boundary, road/field edge, cloud/sky speck, or other.
- Fill `target_*`, `false_lock_kind`, `router_label`, and `review_confidence`
  in `selector_disagreement_review_index.csv`.

Rows rendered: {len(packet_rows)}
"""
    )
    print(out_dir / "selector_disagreement_review_index.csv")
    print(out_dir / "contact_sheet_index.csv")


if __name__ == "__main__":
    main()
