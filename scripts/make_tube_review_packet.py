#!/usr/bin/env python3
"""Build human-review sheets for top tube alternatives at checkpoint frames."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (0, 0, 255),
    (0, 160, 255),
    (0, 220, 0),
    (255, 80, 0),
    (255, 0, 180),
    (180, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (120, 220, 255),
    (255, 180, 120),
    (180, 255, 120),
    (220, 120, 255),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--video_dir", default="/Users/idant/Downloads")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top_n", type=int, default=16)
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--pad", type=int, default=18)
    return p.parse_args()


def bbox_from_text(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    try:
        vals = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
        return None
    return tuple(int(round(float(v))) for v in vals)  # type: ignore[return-value]


def read_frame(video_path: Path, frame_no: int, downscale: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if downscale != 1.0:
        frame = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return frame


def row_bbox(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        int(round(float(row.get("x", "0") or 0))),
        int(round(float(row.get("y", "0") or 0))),
        int(round(float(row.get("w", "1") or 1))),
        int(round(float(row.get("h", "1") or 1))),
    )


def load_top_rows(path: Path, frame_no: int, top_n: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(float(row.get("frame", "0") or 0)) != frame_no:
                continue
            if int(float(row.get("rank", "999") or 999)) > top_n:
                continue
            if int(float(row.get("eligible", "1") or 1)) != 1:
                continue
            rows.append(row)
    return sorted(rows, key=lambda r: int(float(r.get("rank", "999") or 999)))


def draw_overview(
    frame: np.ndarray,
    rows: list[dict[str, str]],
    reviewed_bbox: tuple[int, int, int, int] | None,
    title: str,
) -> np.ndarray:
    img = frame.copy()
    if reviewed_bbox is not None:
        x, y, w, h = reviewed_bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.putText(img, "reviewed", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    for i, row in enumerate(rows):
        x, y, w, h = row_bbox(row)
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, str(row.get("rank", "?")), (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, title[:110], (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def crop_tile(frame: np.ndarray, row: dict[str, str], pad: int) -> np.ndarray:
    h_img, w_img = frame.shape[:2]
    x, y, w, h = row_bbox(row)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
    sx = x - x0
    sy = y - y0
    cv2.rectangle(crop, (sx, sy), (sx + w, sy + h), (0, 0, 255), 1)
    crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_NEAREST)
    label = f"r{row.get('rank')} s{float(row.get('verified_score', 0) or 0):.1f}"
    cv2.rectangle(crop, (0, 0), (112, 16), (0, 0, 0), -1)
    cv2.putText(crop, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def make_crop_sheet(frame: np.ndarray, rows: list[dict[str, str]], pad: int, title: str) -> np.ndarray:
    cols = 4
    rows_n = max(1, int(np.ceil(len(rows) / cols)))
    tile_w = 112
    tile_h = 112
    header_h = 28
    sheet = np.zeros((header_h + rows_n * tile_h, cols * tile_w, 3), dtype=np.uint8)
    cv2.putText(sheet, title[:90], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    for i, row in enumerate(rows):
        tile = crop_tile(frame, row, pad)
        y = header_h + (i // cols) * tile_h
        x = (i % cols) * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = tile
    return sheet


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    results_dir = Path(args.results_dir)
    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    overview_dir = out_dir / "overviews"
    crops_dir = out_dir / "crops"
    overview_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    with labels_path.open() as f:
        labels = list(csv.DictReader(f))

    review_rows: list[dict[str, str]] = []
    for label in labels:
        clip = label.get("clip", "")
        frame_no = int(label.get("frame", "0") or 0)
        video_path = video_dir / f"{clip}.MP4"
        frame = read_frame(video_path, frame_no, args.downscale)
        if frame is None:
            continue
        reviewed_bbox = bbox_from_text(label.get("selected_bbox"))
        top_rows = load_top_rows(results_dir / clip / "top_tubes.csv", frame_no, args.top_n)
        title = f"{clip} frame {frame_no} label={label.get('label')} notes={label.get('notes','')}"
        overview = draw_overview(frame, top_rows, reviewed_bbox, title)
        crop_sheet = make_crop_sheet(frame, top_rows, args.pad, title)
        stem = f"{clip}_f{frame_no:05d}"
        overview_path = overview_dir / f"{stem}_overview.jpg"
        crops_path = crops_dir / f"{stem}_crops.jpg"
        cv2.imwrite(str(overview_path), overview)
        cv2.imwrite(str(crops_path), crop_sheet)

        for row in top_rows:
            bbox = row_bbox(row)
            review_rows.append(
                {
                    "clip": clip,
                    "frame": str(frame_no),
                    "checkpoint_label": label.get("label", ""),
                    "rank": row.get("rank", ""),
                    "bbox": str(list(bbox)),
                    "verified_score": row.get("verified_score", ""),
                    "raw_score": row.get("score", ""),
                    "tube_verifier_score": row.get("tube_verifier_score", ""),
                    "candidate_source": row.get("cand_source", ""),
                    "notes": label.get("notes", ""),
                    "overview_image": str(overview_path),
                    "crop_sheet": str(crops_path),
                    "human_label": "",
                    "human_notes": "",
                    "crop_t_prob": row.get("crop_t_prob", ""),
                    "crop_s_prob": row.get("crop_s_prob", ""),
                    "crop_e_prob": row.get("crop_e_prob", ""),
                    "crop_h_prob": row.get("crop_h_prob", ""),
                    "crop_g_prob": row.get("crop_g_prob", ""),
                    "crop_pred_class": row.get("crop_pred_class", ""),
                    "crop_t_logit": row.get("crop_t_logit", ""),
                    "crop_s_logit": row.get("crop_s_logit", ""),
                    "crop_e_logit": row.get("crop_e_logit", ""),
                    "crop_h_logit": row.get("crop_h_logit", ""),
                    "crop_g_logit": row.get("crop_g_logit", ""),
                    "cand_router_state": row.get("cand_router_state", ""),
                    "cand_line_context": row.get("cand_line_context", ""),
                    "cand_attached_support": row.get("cand_attached_support", ""),
                    "cand_texture": row.get("cand_texture", ""),
                    "clba_bg_static_likelihood": row.get("clba_bg_static_likelihood", ""),
                    "clba_attached_likelihood": row.get("clba_attached_likelihood", ""),
                    "tube_router_boundary_rate": row.get("tube_router_boundary_rate", ""),
                    "tube_router_line_attached_rate": row.get("tube_router_line_attached_rate", ""),
                    "taxonomy_label": "",
                    "taxonomy_label_options": "target|static_hotspot|attached_tree_branch_terrain|skyline_boundary_parallax|terrain_texture|noise|near_target_wrong_center|uncertain",
                }
            )

    with (out_dir / "tube_alternatives_to_label.csv").open("w", newline="") as f:
        fieldnames = [
            "clip",
            "frame",
            "checkpoint_label",
            "rank",
            "bbox",
            "verified_score",
            "raw_score",
            "tube_verifier_score",
            "candidate_source",
            "notes",
            "overview_image",
            "crop_sheet",
            "human_label",
            "human_notes",
            "crop_t_prob",
            "crop_s_prob",
            "crop_e_prob",
            "crop_h_prob",
            "crop_g_prob",
            "crop_pred_class",
            "crop_t_logit",
            "crop_s_logit",
            "crop_e_logit",
            "crop_h_logit",
            "crop_g_logit",
            "cand_router_state",
            "cand_line_context",
            "cand_attached_support",
            "cand_texture",
            "clba_bg_static_likelihood",
            "clba_attached_likelihood",
            "tube_router_boundary_rate",
            "tube_router_line_attached_rate",
            "taxonomy_label",
            "taxonomy_label_options",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    (out_dir / "README.md").write_text(
        f"""# Tube Alternative Review Packet

This packet shows top tube alternatives for reviewed checkpoint frames.

Use `tube_alternatives_to_label.csv` to mark rows in `human_label`.

Suggested labels:

- `target`
- `near_target_wrong_center`
- `static_hotspot`
- `line_attached`
- `parallax_edge`
- `boundary_artifact`
- `appearance_blob`
- `terrain_texture`
- `noise`
- `uncertain`

Images:

- `overviews/`: full downscaled frame with reviewed box in cyan and top alternatives numbered.
- `crops/`: crop sheet for the same top alternatives.

Top alternatives per checkpoint: {args.top_n}
"""
    )
    print(out_dir / "tube_alternatives_to_label.csv")


if __name__ == "__main__":
    main()
