#!/usr/bin/env python3
"""High-recall proposal recovery experiments for tiny drone labels.

This is intentionally separate from tbd_motion_detector.py. It answers one
question first: can a higher-resolution / temporal-stack proposal layer put the
manually labeled target in the candidate set?
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import motion_detector_v2 as base  # noqa: E402


@dataclass
class Proposal:
    frame: int
    x: float
    y: float
    w: float
    h: float
    score: float
    source: str
    radius: int

    @property
    def cx(self) -> float:
        return self.x + 0.5 * self.w

    @property
    def cy(self) -> float:
        return self.y + 0.5 * self.h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--temporal_offsets", default="-5,-3,-2,-1,1,2,3,5")
    p.add_argument("--radii", default="2,3,4,5,7")
    p.add_argument("--top_per_map", type=int, default=250)
    p.add_argument("--max_per_source", type=int, default=500)
    p.add_argument("--max_combined", type=int, default=800)
    p.add_argument("--nms_px", type=float, default=5.5)
    p.add_argument("--hit_dist_orig_px", type=float, default=6.0)
    p.add_argument("--recenter_px", type=int, default=0)
    p.add_argument("--temporal_dark_weight", type=float, default=1.0)
    p.add_argument("--temporal_native_weight", type=float, default=0.55)
    p.add_argument("--temporal_clahe_weight", type=float, default=0.25)
    p.add_argument("--halo_top_bases", type=int, default=48)
    p.add_argument("--halo_offsets", default="0:0,-6:0,6:0,0:-6,0:6,-9:0,9:0,0:-9,0:9,-6:-6,-6:6,6:-6,6:6")
    p.add_argument("--halo_penalty", type=float, default=0.025)
    p.add_argument("--render_source", default="temporal_halo")
    p.add_argument("--render_failures", type=int, default=24)
    return p.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def parse_offsets(text: str) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for part in text.split(","):
        if not part.strip():
            continue
        x_text, y_text = part.split(":")
        offsets.append((float(x_text), float(y_text)))
    return offsets


def read_labels(path: Path, clip_filter: str) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if clip_filter:
        rows = [r for r in rows if r.get("clip", "") == clip_filter]
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def load_gray_frames(video: Path, frame_numbers: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    frames: dict[int, np.ndarray] = {}
    for fno in sorted(n for n in frame_numbers if n >= 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        frames[fno] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return frames


def clahe_image(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def compact_dark_map(gray: np.ndarray, radius: int, texture_weight: float = 0.025) -> np.ndarray:
    img = gray.astype(np.float32)
    r = max(1, int(radius))
    inner_k = 2 * r + 1
    outer_k = 6 * r + 3
    inner_n = float(inner_k * inner_k)
    outer_n = float(outer_k * outer_k)
    ring_n = max(1.0, outer_n - inner_n)

    inner_sum = cv2.boxFilter(img, cv2.CV_32F, (inner_k, inner_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    outer_sum = cv2.boxFilter(img, cv2.CV_32F, (outer_k, outer_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    inner_mean = inner_sum / inner_n
    ring_mean = (outer_sum - inner_sum) / ring_n

    outer_sq = cv2.boxFilter(img * img, cv2.CV_32F, (outer_k, outer_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    outer_mean = outer_sum / outer_n
    outer_var = np.maximum(0.0, outer_sq / outer_n - outer_mean * outer_mean)
    outer_std = np.sqrt(outer_var + 9.0)

    dark_z = (ring_mean - inner_mean) / outer_std

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    texture = cv2.boxFilter(grad, cv2.CV_32F, (outer_k, outer_k), normalize=True, borderType=cv2.BORDER_REFLECT101)

    score = dark_z - texture_weight * texture
    border = max(outer_k, 10)
    score[:border, :] = -999.0
    score[-border:, :] = -999.0
    score[:, :border] = -999.0
    score[:, -border:] = -999.0
    return score.astype(np.float32)


def recenter_on_dark_pixel(gray: np.ndarray, x: float, y: float, search_px: int) -> tuple[float, float]:
    if search_px <= 0:
        return x, y
    h, w = gray.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - search_px)
    y0 = max(0, cy - search_px)
    x1 = min(w, cx + search_px + 1)
    y1 = min(h, cy + search_px + 1)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return x, y
    smooth = cv2.GaussianBlur(crop, (3, 3), 0)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(-smooth)
    # maxLoc on -smooth is the darkest pixel in the original crop.
    return float(x0 + max_loc[0]), float(y0 + max_loc[1])


def nms_peaks(
    score: np.ndarray,
    radius: int,
    top_k: int,
    source: str,
    frame: int,
    recenter_img: np.ndarray | None = None,
    recenter_px: int = 0,
) -> list[Proposal]:
    nms = max(3, int(round(2 * radius + 1)))
    dil = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_RECT, (nms, nms)))
    mask = (score >= dil - 1e-6) & np.isfinite(score) & (score > -100.0)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    vals = score[ys, xs]
    order = np.argsort(vals)[::-1][:top_k]
    side = max(3, int(round(2 * radius + 1)))
    props: list[Proposal] = []
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        if recenter_img is not None:
            x, y = recenter_on_dark_pixel(recenter_img, x, y, recenter_px)
        score_val = float(vals[idx])
        props.append(Proposal(frame, x - side / 2, y - side / 2, side, side, score_val, source, radius))
    return props


def transform_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_corners=1200,
        quality=0.006,
        min_distance=7,
        ransac_px=2.2,
        model="auto",
    )


def estimate_h_to_current(src: np.ndarray, dst: np.ndarray, args: SimpleNamespace) -> tuple[np.ndarray | None, str, float]:
    g0, g1 = base.lk_tracks(src, dst, None, args)
    if g0 is None or g1 is None:
        return None, "none", 0.0
    chosen = base.choose_model(src, dst, g0, g1, args)
    if chosen is None:
        return None, "none", 0.0
    return chosen["h"].astype(np.float32), str(chosen["name"]), float(chosen["inlier_ratio"])


def stabilized_temporal_maps(
    frames: dict[int, np.ndarray],
    frame: int,
    offsets: list[int],
) -> tuple[np.ndarray | None, dict[str, float]]:
    cur = frames[frame]
    h_img, w_img = cur.shape[:2]
    targs = transform_args()
    warped: list[np.ndarray] = []
    inliers: list[float] = []
    for off in offsets:
        other = frames.get(frame + off)
        if other is None:
            continue
        h_mat, _name, inlier = estimate_h_to_current(other, cur, targs)
        if h_mat is None:
            continue
        w = cv2.warpPerspective(other, h_mat, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        warped.append(w)
        inliers.append(inlier)
    if len(warped) < 2:
        return None, {"temporal_neighbors": len(warped), "mean_inlier": 0.0}
    med = np.median(np.stack(warped, axis=0), axis=0).astype(np.float32)
    cur_f = cur.astype(np.float32)
    residual_dark = med - cur_f
    # Local normalization keeps cloud/terrain gradients from dominating quite as hard.
    local_mean = cv2.GaussianBlur(residual_dark, (0, 0), 5.0)
    local_sq = cv2.GaussianBlur(residual_dark * residual_dark, (0, 0), 5.0)
    local_std = np.sqrt(np.maximum(4.0, local_sq - local_mean * local_mean))
    z = (residual_dark - local_mean) / local_std
    return z.astype(np.float32), {"temporal_neighbors": float(len(warped)), "mean_inlier": float(np.mean(inliers))}


def dedupe(props: list[Proposal], nms_px: float, max_n: int) -> list[Proposal]:
    out: list[Proposal] = []
    for p in sorted(props, key=lambda q: q.score, reverse=True):
        if len(out) >= max_n:
            break
        keep = True
        for q in out:
            if math.hypot(p.cx - q.cx, p.cy - q.cy) <= nms_px:
                keep = False
                break
        if keep:
            out.append(p)
    return out


def halo_proposals(
    base_props: list[Proposal],
    offsets: list[tuple[float, float]],
    max_bases: int,
    penalty: float,
    w_img: int,
    h_img: int,
) -> list[Proposal]:
    out: list[Proposal] = []
    for p in sorted(base_props, key=lambda q: q.score, reverse=True)[:max_bases]:
        for dx, dy in offsets:
            dist = math.hypot(dx, dy)
            x = min(max(0.0, p.x + dx), max(0.0, w_img - p.w))
            y = min(max(0.0, p.y + dy), max(0.0, h_img - p.h))
            out.append(
                Proposal(
                    frame=p.frame,
                    x=x,
                    y=y,
                    w=p.w,
                    h=p.h,
                    score=p.score - penalty * dist,
                    source="temporal_halo",
                    radius=p.radius,
                )
            )
    return sorted(out, key=lambda q: q.score, reverse=True)


def generate_proposals_for_frame(
    frames: dict[int, np.ndarray],
    frame: int,
    radii: list[int],
    offsets: list[int],
    args: argparse.Namespace,
) -> tuple[dict[str, list[Proposal]], dict[str, float]]:
    gray = frames[frame]
    clahe = clahe_image(gray)
    by_source: dict[str, list[Proposal]] = {}
    stats: dict[str, float] = {}

    raw_props: list[Proposal] = []
    clahe_props: list[Proposal] = []
    for r in radii:
        raw_map = compact_dark_map(gray, r)
        clahe_map = compact_dark_map(clahe, r, texture_weight=0.015)
        raw_props.extend(nms_peaks(raw_map, r, args.top_per_map, "native_dark", frame, gray, args.recenter_px))
        clahe_props.extend(nms_peaks(clahe_map, r, args.top_per_map, "clahe_dark", frame, gray, args.recenter_px))
    by_source["native_dark"] = dedupe(raw_props, args.nms_px, args.max_per_source)
    by_source["clahe_dark"] = dedupe(clahe_props, args.nms_px, args.max_per_source)

    temp_map, temp_stats = stabilized_temporal_maps(frames, frame, offsets)
    stats.update(temp_stats)
    temp_props: list[Proposal] = []
    temp_combo_props: list[Proposal] = []
    if temp_map is not None:
        for r in radii:
            raw_map = compact_dark_map(gray, r)
            clahe_map = compact_dark_map(clahe, r, texture_weight=0.015)
            temp_combo = (
                args.temporal_dark_weight * temp_map
                + args.temporal_native_weight * raw_map
                + args.temporal_clahe_weight * clahe_map
            )
            temp_props.extend(nms_peaks(temp_map, r, args.top_per_map, "temporal_dark", frame, gray, args.recenter_px))
            temp_combo_props.extend(
                nms_peaks(temp_combo, r, args.top_per_map, "temporal_combo", frame, gray, args.recenter_px)
            )
    by_source["temporal_dark"] = dedupe(temp_props, args.nms_px, args.max_per_source)
    by_source["temporal_combo"] = dedupe(temp_combo_props, args.nms_px, args.max_per_source)
    by_source["temporal_halo"] = dedupe(
        halo_proposals(
            by_source["temporal_combo"],
            parse_offsets(args.halo_offsets),
            args.halo_top_bases,
            args.halo_penalty,
            gray.shape[1],
            gray.shape[0],
        ),
        max(2.5, 0.45 * args.nms_px),
        args.max_per_source,
    )

    combined = []
    for props in by_source.values():
        combined.extend(props)
    by_source["combined"] = dedupe(combined, args.nms_px, args.max_combined)
    return by_source, stats


def bbox_center_xy(row: dict[str, str]) -> tuple[float, float]:
    return float(row["orig_cx"]), float(row["orig_cy"])


def audit_source(
    labels: list[dict[str, str]],
    proposals: dict[int, list[Proposal]],
    k_values: list[int],
    hit_dist: float,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for lab in labels:
        frame = int(lab["frame"])
        tx, ty = bbox_center_xy(lab)
        props = proposals.get(frame, [])
        best_i = None
        best_d = None
        for i, p in enumerate(props, start=1):
            d = math.hypot(p.cx - tx, p.cy - ty)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        rec: dict[str, str | int | float] = {
            "clip": lab.get("clip", ""),
            "frame": frame,
            "confidence": lab.get("confidence", ""),
            "target_orig_cx": round(tx, 2),
            "target_orig_cy": round(ty, 2),
            "best_rank": best_i if best_i is not None else "",
            "best_dist_orig_px": round(best_d, 3) if best_d is not None else "",
        }
        for k in k_values:
            rec[f"hit_top{k}"] = int(best_i is not None and best_i <= k and best_d is not None and best_d <= hit_dist)
        rows.append(rec)
    return rows


def summarize_audit(rows: list[dict[str, str | int | float]], k_values: list[int]) -> dict[str, str | int | float]:
    total = len(rows)
    high = [r for r in rows if r["confidence"] == "high"]
    out: dict[str, str | int | float] = {"labels": total, "high_labels": len(high)}
    for k in k_values:
        hits = sum(int(r[f"hit_top{k}"]) for r in rows)
        high_hits = sum(int(r[f"hit_top{k}"]) for r in high)
        out[f"recall_top{k}"] = round(hits / total, 3) if total else 0.0
        out[f"high_recall_top{k}"] = round(high_hits / len(high), 3) if high else 0.0
    ranks = [int(r["best_rank"]) for r in rows if r.get("best_rank") != "" and float(r["best_dist_orig_px"]) <= 6.0]
    out["median_hit_rank"] = sorted(ranks)[len(ranks) // 2] if ranks else ""
    out["max_hit_rank"] = max(ranks) if ranks else ""
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_failures(
    frames: dict[int, np.ndarray],
    labels: list[dict[str, str]],
    props: dict[int, list[Proposal]],
    audit_rows: list[dict[str, str | int | float]],
    out_dir: Path,
    top_k: int,
    max_pages: int,
) -> None:
    failures = [r for r in audit_rows if int(r.get(f"hit_top{top_k}", 0)) == 0]
    failures = failures[: max_pages * 4]
    label_by_frame = {int(r["frame"]): r for r in labels}
    out = out_dir / f"failure_sheets_top{top_k}"
    out.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 640, 480
    for page_idx in range(0, len(failures), 4):
        sheet = np.zeros((2 * cell_h, 2 * cell_w, 3), dtype=np.uint8) + 18
        for i, rec in enumerate(failures[page_idx : page_idx + 4]):
            fno = int(rec["frame"])
            gray = frames[fno]
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            lab = label_by_frame[fno]
            tx, ty = bbox_center_xy(lab)
            cv2.rectangle(bgr, (int(tx - 4), int(ty - 4)), (int(tx + 4), int(ty + 4)), (0, 255, 255), 2)
            cv2.line(bgr, (int(tx - 12), int(ty)), (int(tx + 12), int(ty)), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(bgr, (int(tx), int(ty - 12)), (int(tx), int(ty + 12)), (0, 255, 255), 2, cv2.LINE_AA)
            for rank, p in enumerate(props.get(fno, [])[: min(top_k, 30)], start=1):
                color = (0, 0, 255) if rank == 1 else (80, 180, 80)
                cv2.rectangle(bgr, (int(p.x), int(p.y)), (int(p.x + p.w), int(p.y + p.h)), color, 1)
                if rank <= 12:
                    cv2.putText(bgr, str(rank), (int(p.x), max(12, int(p.y - 3))), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"f{fno} target cyan, proposals green/red, best_dist={rec.get('best_dist_orig_px')}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            scale = min(cell_w / bgr.shape[1], cell_h / bgr.shape[0])
            resized_w = max(1, int(round(bgr.shape[1] * scale)))
            resized_h = max(1, int(round(bgr.shape[0] * scale)))
            panel = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
            y = (i // 2) * cell_h
            x = (i % 2) * cell_w
            y_pad = y + (cell_h - resized_h) // 2
            x_pad = x + (cell_w - resized_w) // 2
            sheet[y_pad : y_pad + resized_h, x_pad : x_pad + resized_w] = panel
        cv2.imwrite(str(out / f"page_{page_idx // 4 + 1:02d}.png"), sheet)


def main() -> None:
    args = parse_args()
    video = Path(args.video)
    labels = read_labels(Path(args.labels), args.clip)
    if not labels:
        raise SystemExit("no labels loaded")
    offsets = parse_ints(args.temporal_offsets)
    radii = parse_ints(args.radii)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_frames = {int(r["frame"]) for r in labels}
    needed = set(label_frames)
    for f in label_frames:
        for off in offsets:
            needed.add(f + off)
    frames = load_gray_frames(video, needed)
    missing = sorted(label_frames - set(frames))
    if missing:
        raise RuntimeError(f"missing labeled frames from video: {missing[:8]}")

    source_props: dict[str, dict[int, list[Proposal]]] = {
        "native_dark": {},
        "clahe_dark": {},
        "temporal_dark": {},
        "temporal_combo": {},
        "temporal_halo": {},
        "combined": {},
    }
    stat_rows: list[dict[str, str | int | float]] = []
    for n, fno in enumerate(sorted(label_frames), start=1):
        by_source, stats = generate_proposals_for_frame(frames, fno, radii, offsets, args)
        for source, props in by_source.items():
            source_props[source][fno] = props
        stat_rows.append(
            {
                "frame": fno,
                "temporal_neighbors": stats.get("temporal_neighbors", 0.0),
                "mean_temporal_inlier": round(stats.get("mean_inlier", 0.0), 3),
                "native_count": len(by_source["native_dark"]),
                "clahe_count": len(by_source["clahe_dark"]),
                "temporal_count": len(by_source["temporal_dark"]),
                "temporal_combo_count": len(by_source["temporal_combo"]),
                "temporal_halo_count": len(by_source["temporal_halo"]),
                "combined_count": len(by_source["combined"]),
            }
        )
        if n % 10 == 0:
            print(f"processed {n}/{len(label_frames)} frames", flush=True)

    k_values = [20, 50, 80, 120, 200, 500]
    summary_rows: list[dict[str, str | int | float]] = []
    for source, per_frame in source_props.items():
        prop_rows: list[dict[str, str | int | float]] = []
        for fno, props in sorted(per_frame.items()):
            for rank, p in enumerate(props, start=1):
                prop_rows.append(
                    {
                        "frame": fno,
                        "rank": rank,
                        "x": round(p.x, 2),
                        "y": round(p.y, 2),
                        "w": round(p.w, 2),
                        "h": round(p.h, 2),
                        "cx": round(p.cx, 2),
                        "cy": round(p.cy, 2),
                        "score": round(p.score, 4),
                        "source": p.source,
                        "radius": p.radius,
                    }
                )
        write_csv(out_dir / f"proposals_{source}.csv", prop_rows)
        audit_rows = audit_source(labels, per_frame, k_values, args.hit_dist_orig_px)
        write_csv(out_dir / f"audit_{source}.csv", audit_rows)
        summary = summarize_audit(audit_rows, k_values)
        summary["source"] = source
        summary["avg_candidates"] = round(float(np.mean([len(v) for v in per_frame.values()])), 1)
        summary_rows.append(summary)
        if source == args.render_source:
            render_failures(frames, labels, per_frame, audit_rows, out_dir, top_k=80, max_pages=args.render_failures)

    write_csv(out_dir / "frame_stats.csv", stat_rows)
    write_csv(out_dir / "summary.csv", summary_rows)

    lines = ["# Proposal Recovery Experiment\n", ""]
    lines.append(f"Video: `{video}`")
    lines.append(f"Labels: `{args.labels}`")
    lines.append(f"Hit threshold: center distance <= {args.hit_dist_orig_px:.1f} original px")
    lines.append("")
    lines.append("| Source | Avg cand | R@20 | R@50 | R@80 | R@120 | R@200 | R@500 | High R@80 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| `{row['source']}` | {row['avg_candidates']} | {row['recall_top20']} | {row['recall_top50']} | "
            f"{row['recall_top80']} | {row['recall_top120']} | {row['recall_top200']} | {row['recall_top500']} | "
            f"{row['high_recall_top80']} |"
        )
    lines.append("")
    lines.append(f"Failure sheets for `{args.render_source}` top-80 are in `failure_sheets_top80/`.")
    (out_dir / "README.md").write_text("\n".join(lines))
    print(out_dir / "summary.csv")


if __name__ == "__main__":
    main()
