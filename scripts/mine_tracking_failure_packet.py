#!/usr/bin/env python3
"""Mine continuity gaps and jumpy locks from rendered tracking selections.

This creates a review/training packet from full-video selector output. It is
meant to answer: where does the current tracker lose continuity, and which
frames should be reviewed or labeled next?
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


RISK_ROUTERS = {"surface", "line", "line_attached", "boundary", "unknown"}


@dataclass(frozen=True)
class Event:
    clip: str
    issue_type: str
    start_frame: int
    end_frame: int
    score: float
    detail: str
    router_bucket: str = ""

    @property
    def length(self) -> int:
        return max(1, self.end_frame - self.start_frame + 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", required=True, help="CSV with clip/frame/selected/x/y/w/h rows.")
    p.add_argument("--video_dir", required=True, help="Directory containing source MP4 files.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--selection_scale", type=float, default=2.0)
    p.add_argument("--min_gap_frames", type=int, default=12)
    p.add_argument("--jump_px", type=float, default=28.0, help="Detector-coordinate jump threshold.")
    p.add_argument("--max_events_per_clip", type=int, default=14)
    p.add_argument("--max_review_frames", type=int, default=180)
    p.add_argument("--crop_half", type=int, default=74, help="Full-resolution crop half-size.")
    p.add_argument("--cols", type=int, default=4)
    return p.parse_args()


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_video_meta(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return w, h, fps, frames


def video_for_clip(video_dir: Path, clip: str) -> Path:
    exact = video_dir / f"{clip}.MP4"
    if exact.exists():
        return exact
    short = clip.split("-")[0]
    matches = sorted(video_dir.glob(f"{short}*.MP4"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no video found for {clip} in {video_dir}")


def row_box(row: pd.Series) -> tuple[float, float, float, float] | None:
    if int(fnum(row.get("selected"), 0)) != 1:
        return None
    vals = [fnum(row.get(k), math.nan) for k in ("x", "y", "w", "h")]
    if any(not math.isfinite(v) for v in vals):
        return None
    return vals[0], vals[1], vals[2], vals[3]


def selected_center(row: pd.Series) -> tuple[float, float] | None:
    box = row_box(row)
    if box is None:
        return None
    x, y, w, h = box
    return x + 0.5 * w, y + 0.5 * h


def selected_mask(rows: pd.DataFrame, frame_count: int) -> dict[int, bool]:
    out = {frame: False for frame in range(frame_count)}
    for row in rows.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        if 0 <= frame < frame_count:
            out[frame] = int(fnum(getattr(row, "selected", 0), 0)) == 1
    return out


def gap_events(clip: str, rows: pd.DataFrame, frame_count: int, min_gap_frames: int) -> list[Event]:
    selected = selected_mask(rows, frame_count)
    events: list[Event] = []
    start: int | None = None
    for frame in range(frame_count + 1):
        is_gap = frame < frame_count and not selected.get(frame, False)
        if is_gap and start is None:
            start = frame
        if (not is_gap or frame == frame_count) and start is not None:
            end = frame - 1
            length = end - start + 1
            if length >= min_gap_frames:
                events.append(Event(clip, "no_selection_gap", start, end, float(length), f"{length} no-box frames"))
            start = None
    return events


def jump_events(clip: str, rows: pd.DataFrame, jump_px: float) -> list[Event]:
    selected_rows = rows[rows["selected"].astype(int).eq(1)].sort_values("frame")
    events: list[Event] = []
    prev_row: pd.Series | None = None
    prev_center: tuple[float, float] | None = None
    for _, row in selected_rows.iterrows():
        center = selected_center(row)
        if center is None:
            continue
        if prev_row is not None and prev_center is not None:
            frame = int(row["frame"])
            prev_frame = int(prev_row["frame"])
            frame_delta = max(1, frame - prev_frame)
            dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            per_frame = dist / frame_delta
            router = str(row.get("router_bucket", "") or "")
            risk_bonus = 1.35 if router in RISK_ROUTERS else 1.0
            if dist >= jump_px or per_frame >= max(8.0, jump_px / 2.0):
                detail = f"jump={dist:.1f}px over {frame_delta}f ({per_frame:.1f}px/f)"
                events.append(Event(clip, "selected_jump", prev_frame, frame, dist * risk_bonus, detail, router))
        prev_row = row
        prev_center = center
    return events


def low_margin_events(clip: str, rows: pd.DataFrame) -> list[Event]:
    if "target_margin" not in rows.columns:
        return []
    selected = rows[rows["selected"].astype(int).eq(1)].copy()
    if selected.empty:
        return []
    selected["target_margin_num"] = selected["target_margin"].map(lambda v: fnum(v, 99.0))
    selected["rank_num"] = selected["rank"].map(lambda v: fnum(v, 99.0))
    router = selected["router_bucket"].astype(str) if "router_bucket" in selected.columns else pd.Series("", index=selected.index)
    risky = selected[
        (selected["target_margin_num"] < 0.6)
        | (selected["rank_num"] >= 10)
        | (router.isin(RISK_ROUTERS))
    ].copy()
    events: list[Event] = []
    for _, row in risky.iterrows():
        frame = int(row["frame"])
        rank = fnum(row.get("rank"), 99.0)
        margin = fnum(row.get("target_margin"), 0.0)
        router = str(row.get("router_bucket", "") or "")
        score = max(0.0, 12.0 - margin) + max(0.0, rank - 4.0) * 0.5
        if router in RISK_ROUTERS:
            score += 4.0
        events.append(
            Event(
                clip,
                "risky_selected_lock",
                frame,
                frame,
                score,
                f"rank={rank:.0f} margin={margin:.2f} router={router}",
                router,
            )
        )
    return events


def choose_review_frames(event: Event) -> list[int]:
    if event.issue_type == "no_selection_gap":
        mid = (event.start_frame + event.end_frame) // 2
        return sorted({event.start_frame, mid, event.end_frame})
    if event.issue_type == "selected_jump":
        return sorted({event.start_frame, event.end_frame})
    return [event.start_frame]


def put_text(img: np.ndarray, text: str, xy: tuple[int, int], color=(255, 255, 255), scale: float = 0.52) -> None:
    x, y = xy
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_detector_box(
    img: np.ndarray,
    row: pd.Series | None,
    label: str,
    color: tuple[int, int, int],
    selection_scale: float,
) -> tuple[int, int] | None:
    if row is None:
        return None
    box = row_box(row)
    if box is None:
        return None
    x, y, w, h = box
    x = int(round(x * selection_scale))
    y = int(round(y * selection_scale))
    w = max(2, int(round(w * selection_scale)))
    h = max(2, int(round(h * selection_scale)))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    put_text(img, label, (x, max(18, y - 5)), color, 0.45)
    return x + w // 2, y + h // 2


def crop_context(img: np.ndarray, centers: list[tuple[int, int]], half: int) -> np.ndarray:
    if not centers:
        return cv2.resize(img, (260, 180), interpolation=cv2.INTER_AREA)
    cx = int(round(sum(c[0] for c in centers) / len(centers)))
    cy = int(round(sum(c[1] for c in centers) / len(centers)))
    h, w = img.shape[:2]
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return cv2.resize(img, (260, 180), interpolation=cv2.INTER_AREA)
    return cv2.resize(crop, (260, 260), interpolation=cv2.INTER_NEAREST)


def contact_sheet(paths: list[Path], out_path: Path, cols: int, thumb_w: int) -> None:
    imgs: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = thumb_w / max(1, img.shape[1])
        imgs.append(cv2.resize(img, (thumb_w, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    if not imgs:
        return
    h = max(img.shape[0] for img in imgs)
    w = max(img.shape[1] for img in imgs)
    rows = int(math.ceil(len(imgs) / max(1, cols)))
    canvas = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        rr, cc = divmod(idx, cols)
        canvas[rr * h : rr * h + img.shape[0], cc * w : cc * w + img.shape[1]] = img
    cv2.imwrite(str(out_path), canvas)


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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "frames_clean"
    diag_dir = out_dir / "frames_diagnostic"
    crop_dir = out_dir / "crops_context"
    clean_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    selections = pd.read_csv(args.selections)
    selections["clip"] = selections["clip"].astype(str)
    selections["selected"] = selections["selected"].fillna(0).astype(int)
    video_dir = Path(args.video_dir)

    all_events: list[Event] = []
    review_rows: list[dict[str, Any]] = []
    clean_paths: list[Path] = []
    diag_paths: list[Path] = []
    crop_paths: list[Path] = []

    for clip, rows in selections.groupby("clip", sort=True):
        video_path = video_for_clip(video_dir, str(clip))
        _, _, fps, frame_count = read_video_meta(video_path)
        clip_events = []
        clip_events.extend(gap_events(str(clip), rows, frame_count, args.min_gap_frames))
        clip_events.extend(jump_events(str(clip), rows, args.jump_px))
        clip_events.extend(low_margin_events(str(clip), rows))
        clip_events.sort(key=lambda e: e.score, reverse=True)
        clip_events = clip_events[: args.max_events_per_clip]
        all_events.extend(clip_events)

        rows_by_frame = {int(r.frame): pd.Series(r._asdict()) for r in rows.itertuples(index=False)}
        selected_frames = sorted(int(r.frame) for r in rows.itertuples(index=False) if int(getattr(r, "selected", 0)) == 1)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {video_path}")

        for event in clip_events:
            for frame_no in choose_review_frames(event):
                if len(review_rows) >= args.max_review_frames:
                    break
                frame_no = max(0, min(frame_count - 1, frame_no))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if not ok:
                    continue

                prev_selected = max((f for f in selected_frames if f <= frame_no), default=None)
                next_selected = min((f for f in selected_frames if f >= frame_no), default=None)
                row = rows_by_frame.get(frame_no)
                prev_row = rows_by_frame.get(prev_selected) if prev_selected is not None else None
                next_row = rows_by_frame.get(next_selected) if next_selected is not None else None

                clean = frame.copy()
                diag = frame.copy()
                short = str(clip).split("-")[0]
                title = f"{short} f{frame_no} t={frame_no / fps:.2f}s {event.issue_type}"
                put_text(clean, title, (8, 22))
                put_text(diag, title, (8, 22), (0, 255, 255))
                centers: list[tuple[int, int]] = []
                cur_center = draw_detector_box(diag, row, "current", (0, 255, 0), args.selection_scale)
                if cur_center is not None:
                    centers.append(cur_center)
                if prev_row is not None and (row is None or int(prev_row.get("frame", -1)) != frame_no):
                    prev_center = draw_detector_box(diag, prev_row, f"prev {prev_selected}", (255, 180, 0), args.selection_scale)
                    if prev_center is not None:
                        centers.append(prev_center)
                if next_row is not None and (row is None or int(next_row.get("frame", -1)) != frame_no):
                    next_center = draw_detector_box(diag, next_row, f"next {next_selected}", (0, 180, 255), args.selection_scale)
                    if next_center is not None:
                        centers.append(next_center)
                if len(centers) >= 2:
                    for a, b in zip(centers, centers[1:]):
                        cv2.line(diag, a, b, (0, 0, 255), 2, cv2.LINE_AA)
                crop = crop_context(diag, centers, args.crop_half)

                stem = f"{len(review_rows) + 1:03d}_{short}_f{frame_no:05d}_{event.issue_type}"
                clean_path = clean_dir / f"{stem}.jpg"
                diag_path = diag_dir / f"{stem}_diag.jpg"
                crop_path = crop_dir / f"{stem}_crop.jpg"
                cv2.imwrite(str(clean_path), clean)
                cv2.imwrite(str(diag_path), diag)
                cv2.imwrite(str(crop_path), crop)
                clean_paths.append(clean_path)
                diag_paths.append(diag_path)
                crop_paths.append(crop_path)
                selected_now = int(fnum(row.get("selected"), 0)) if row is not None else 0
                review_rows.append(
                    {
                        "review_id": len(review_rows) + 1,
                        "clip": clip,
                        "frame": frame_no,
                        "time_s": round(frame_no / fps, 3) if fps else "",
                        "issue_type": event.issue_type,
                        "event_start_frame": event.start_frame,
                        "event_end_frame": event.end_frame,
                        "event_score": round(event.score, 3),
                        "event_detail": event.detail,
                        "router_bucket": event.router_bucket or (str(row.get("router_bucket", "")) if row is not None else ""),
                        "selected_now": selected_now,
                        "rank": row.get("rank", "") if row is not None else "",
                        "target_margin": row.get("target_margin", "") if row is not None else "",
                        "x": row.get("x", "") if row is not None else "",
                        "y": row.get("y", "") if row is not None else "",
                        "w": row.get("w", "") if row is not None else "",
                        "h": row.get("h", "") if row is not None else "",
                        "prev_selected_frame": prev_selected if prev_selected is not None else "",
                        "next_selected_frame": next_selected if next_selected is not None else "",
                        "clean_image": str(clean_path),
                        "diagnostic_image": str(diag_path),
                        "crop_image": str(crop_path),
                        "review_visible_target": "",
                        "review_target_x": "",
                        "review_target_y": "",
                        "review_target_w": "",
                        "review_target_h": "",
                        "review_label": "",
                        "review_notes": "",
                    }
                )
            if len(review_rows) >= args.max_review_frames:
                break
        cap.release()

    event_rows = [
        {
            "clip": e.clip,
            "issue_type": e.issue_type,
            "start_frame": e.start_frame,
            "end_frame": e.end_frame,
            "length": e.length,
            "score": round(e.score, 3),
            "detail": e.detail,
            "router_bucket": e.router_bucket,
        }
        for e in sorted(all_events, key=lambda e: (e.clip, -e.score))
    ]
    write_csv(out_dir / "failure_events.csv", event_rows)
    write_csv(out_dir / "review_frames.csv", review_rows)
    contact_sheet(clean_paths, out_dir / "contact_clean.jpg", args.cols, 360)
    contact_sheet(diag_paths, out_dir / "contact_diagnostic.jpg", args.cols, 360)
    contact_sheet(crop_paths, out_dir / "contact_crops.jpg", args.cols, 240)

    by_clip: dict[str, dict[str, Any]] = {}
    for event in all_events:
        rec = by_clip.setdefault(event.clip, {"clip": event.clip, "events": 0, "gap_events": 0, "jump_events": 0, "risky_lock_events": 0})
        rec["events"] += 1
        if event.issue_type == "no_selection_gap":
            rec["gap_events"] += 1
        elif event.issue_type == "selected_jump":
            rec["jump_events"] += 1
        elif event.issue_type == "risky_selected_lock":
            rec["risky_lock_events"] += 1
    write_csv(out_dir / "by_clip_failure_summary.csv", list(by_clip.values()))

    (out_dir / "README.md").write_text(
        "# Full Sample Tracking Failure Packet\n\n"
        "This packet mines the current full-sample render selections for long no-box gaps, "
        "large selected-box jumps, and risky surface/low-margin selected locks.\n\n"
        "Use `frames_clean/` for unbiased visual checks. Use `frames_diagnostic/` and "
        "`crops_context/` to inspect current tracker behavior after deciding whether the "
        "target is visible. Review coordinates are detector-space half-resolution, matching "
        "the training labels.\n\n"
        "Files:\n"
        "- `failure_events.csv`: ranked continuity/selection failure events.\n"
        "- `review_frames.csv`: frame-level review template for labels or false-lock notes.\n"
        "- `contact_clean.jpg`, `contact_diagnostic.jpg`, `contact_crops.jpg`: quick sheets.\n"
    )

    print(out_dir / "failure_events.csv")
    print(out_dir / "review_frames.csv")
    print(out_dir / "contact_diagnostic.jpg")


if __name__ == "__main__":
    main()
