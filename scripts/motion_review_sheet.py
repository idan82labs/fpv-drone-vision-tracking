#!/usr/bin/env python3
"""
Build human-review sheets from motion_detector_v2 outputs.

This does not change detection. It creates evenly sampled overlay/crop sheets
and a CSV where a reviewer can mark each sampled selected box as:

  gold    selected box is on the independently moving object
  clutter selected box is on background/texture/cloud/edge clutter
  miss    there is a visible object but no useful selected box
  empty   no relevant object visible in that frame
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("result_dir")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--crop_pad", type=int, default=36)
    return p.parse_args()


def safe_read(path: Path):
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"cannot read {path}")
    return im


def main() -> None:
    args = parse_args()
    root = Path(args.result_dir)
    rows = []
    crop_rows = []
    review_rows = []

    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        report_path = d / "report.json"
        if not report_path.exists():
            continue
        data = json.loads(report_path.read_text())
        frames_by_no = {r["frame"]: r for r in data["frames"]}
        overlays = sorted(d.glob("overlay_*.png"))
        if not overlays:
            continue
        idxs = np.linspace(0, len(overlays) - 1, min(args.samples, len(overlays))).round().astype(int)

        thumbs = []
        crops = []
        for idx in idxs:
            overlay_path = overlays[int(idx)]
            im = safe_read(overlay_path)
            fno = int(overlay_path.stem.split("_")[-1])
            rec = frames_by_no.get(fno, {})
            sel = rec.get("selected")
            review_rows.append(
                {
                    "clip": d.name,
                    "frame": fno,
                    "selected_bbox": json.dumps(sel["bbox"]) if sel else "",
                    "selected_source": sel.get("source", "") if sel else "",
                    "selected_score": sel.get("score", "") if sel else "",
                    "label": "",
                    "notes": "",
                    "overlay": str(overlay_path),
                }
            )
            thumbs.append(cv2.resize(im, (150, 112), interpolation=cv2.INTER_AREA))
            if sel:
                x, y, w, h = sel["bbox"]
                x0 = max(0, x - args.crop_pad)
                y0 = max(0, y - args.crop_pad)
                x1 = min(im.shape[1], x + w + args.crop_pad)
                y1 = min(im.shape[0], y + h + args.crop_pad)
                crop = im[y0:y1, x0:x1]
                crops.append(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST))
            else:
                crops.append(np.full((128, 128, 3), 245, dtype=np.uint8))

        while len(thumbs) < args.samples:
            thumbs.append(np.full((112, 150, 3), 245, dtype=np.uint8))
        while len(crops) < args.samples:
            crops.append(np.full((128, 128, 3), 245, dtype=np.uint8))

        label = np.full((112, 280, 3), 255, dtype=np.uint8)
        cv2.putText(label, d.name[:21], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
        summary = data.get("summary", {})
        cv2.putText(
            label,
            f"sel {100 * summary.get('selected_frame_rate', 0):.0f}% cand {summary.get('avg_candidates_per_frame', 0):.1f}",
            (6, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.hstack([label] + thumbs))

        crop_label = np.full((128, 280, 3), 255, dtype=np.uint8)
        cv2.putText(crop_label, d.name[:21], (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(crop_label, "review crops", (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)
        crop_rows.append(np.hstack([crop_label] + crops))

    if rows:
        cv2.imwrite(str(root / "review_overlays.jpg"), np.vstack(rows))
    if crop_rows:
        cv2.imwrite(str(root / "review_crops.jpg"), np.vstack(crop_rows))
    with (root / "review_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip", "frame", "selected_bbox", "selected_source", "selected_score", "label", "notes", "overlay"],
        )
        writer.writeheader()
        writer.writerows(review_rows)
    print(root / "review_overlays.jpg")
    print(root / "review_crops.jpg")
    print(root / "review_labels.csv")


if __name__ == "__main__":
    main()
