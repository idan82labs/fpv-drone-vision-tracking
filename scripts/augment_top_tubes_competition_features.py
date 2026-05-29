#!/usr/bin/env python3
"""Add frame-local competitor normalization features to top-tube rows.

This is a cheap offline probe for the professor's "local distractor-normalized
margin" recommendation. The CLBA augmenter already scores each candidate against
synthetic annulus controls. This script adds a second comparison: does this
candidate beat the other candidates in the same frame, especially candidates in
similar local context or spatially nearby clutter?

The output is another top_tubes.csv-compatible file, so existing ranker and
selector harnesses can consume the new numeric ``comp_*`` columns without code
changes.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--top_tubes", help="Single top_tubes.csv input.")
    group.add_argument("--results_dir", help="Directory containing clip/top_tubes.csv files.")
    p.add_argument("--out_csv", help="Single-file output path for --top_tubes.")
    p.add_argument("--out_dir", help="Output directory for --results_dir.")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--near_radius_px", type=float, default=36.0)
    p.add_argument("--min_context_controls", type=int, default=4)
    p.add_argument("--sigma_floor", type=float, default=0.25)
    p.add_argument(
        "--metrics",
        default="target_margin,gain_norm",
        help=(
            "Comma-separated competition metrics. Defaults to the CLBA-only "
            "signals; add proposal_score only for diagnostics."
        ),
    )
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


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def logsumexp(values: list[float]) -> float:
    if not values:
        return -1.0e9
    m = max(values)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def center(row: dict[str, Any]) -> tuple[float, float]:
    x = safe_float(row.get("x"))
    y = safe_float(row.get("y"))
    w = max(1.0, safe_float(row.get("w"), 1.0))
    h = max(1.0, safe_float(row.get("h"), 1.0))
    return x + 0.5 * w, y + 0.5 * h


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def row_metric(row: dict[str, Any], key: str) -> float:
    if key == "target_margin":
        target = safe_float(row.get("clba_target_likelihood"))
        static = safe_float(row.get("clba_bg_static_likelihood"))
        attached = safe_float(row.get("clba_attached_likelihood"))
        return target - logsumexp([static, attached, 0.0])
    if key == "gain_norm":
        return safe_float(row.get("clba_gain_norm"), safe_float(row.get("clba_gain")))
    if key == "proposal_score":
        return safe_float(row.get("learned_score"), safe_float(row.get("verified_score"), safe_float(row.get("score"))))
    raise KeyError(key)


def max_context_value(row: dict[str, Any], names: tuple[str, ...]) -> float:
    return max(safe_float(row.get(name)) for name in names)


def context_bucket(row: dict[str, Any]) -> str:
    router = str(row.get("cand_router_state", "")).strip()
    if router and router not in {"unrouted", "unknown"}:
        return router
    line = max_context_value(row, ("cand_line_context", "tube_mean_line_context", "tube_max_line_context"))
    support = max_context_value(row, ("cand_attached_support", "tube_mean_attached_support", "tube_max_attached_support"))
    texture = max_context_value(row, ("cand_texture", "tube_mean_texture"))
    sky = max_context_value(row, ("cand_sky_like", "tube_mean_sky_like", "tube_max_sky_like"))
    surface_rate = safe_float(row.get("tube_router_surface_backed_rate"))
    boundary_rate = safe_float(row.get("tube_router_boundary_rate"))
    line_rate = safe_float(row.get("tube_router_line_attached_rate"))
    clean_sky_rate = safe_float(row.get("tube_router_clean_sky_rate"))

    if line_rate >= 0.35 or support >= 6.0 or line >= 0.35:
        return "attached_linear"
    if boundary_rate >= 0.35:
        return "skyline_boundary"
    if clean_sky_rate >= 0.5 or (sky >= 0.25 and texture < 45.0):
        return "clean_sky"
    if sky >= 0.08 and texture < 70.0:
        return "cloud_sky_texture"
    if surface_rate >= 0.35 or texture >= 65.0:
        return "surface_texture"
    return "unknown"


def robust_z(value: float, controls: list[float], sigma_floor: float) -> tuple[float, float, float, float]:
    if not controls:
        return 0.0, 0.0, sigma_floor, 0.0
    arr = np.asarray(controls, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = max(float(sigma_floor), 1.4826 * mad)
    best = float(np.max(arr)) if arr.size else med
    return float((value - med) / sigma), med, sigma, float(value - best)


def percentile_rank(value: float, controls: list[float]) -> float:
    if not controls:
        return 1.0
    arr = np.asarray(controls, dtype=np.float32)
    return float(np.mean(arr <= value))


def control_sets(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    buckets: dict[int, str],
    index: int,
    near_radius_px: float,
    min_context_controls: int,
) -> dict[str, list[dict[str, Any]]]:
    others = [r for i, r in enumerate(rows) if i != index]
    same_context = [r for i, r in enumerate(rows) if i != index and buckets[i] == buckets[index]]
    if len(same_context) < min_context_controls:
        same_context = others
    near = [r for i, r in enumerate(rows) if i != index and distance(row, r) <= near_radius_px]
    if len(near) < min_context_controls:
        near = same_context
    return {"frame": others, "context": same_context, "near": near}


def add_competition_features(
    rows: list[dict[str, str]],
    max_rank: int,
    near_radius_px: float,
    min_context_controls: int,
    sigma_floor: float,
    metrics: tuple[str, ...] = ("target_margin", "gain_norm"),
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        rank = safe_int(row.get("rank"), 999999)
        if rank <= max_rank:
            by_frame[safe_int(row.get("frame"), -1)].append(row)
        else:
            out_rows.append(dict(row))

    for frame in sorted(by_frame):
        frame_rows = sorted(by_frame[frame], key=lambda r: safe_int(r.get("rank"), 999999))
        buckets = {i: context_bucket(row) for i, row in enumerate(frame_rows)}
        for i, row in enumerate(frame_rows):
            out = dict(row)
            bucket = buckets[i]
            controls = control_sets(row, frame_rows, buckets, i, near_radius_px, min_context_controls)
            out["comp_context_bucket"] = bucket
            out["comp_bucket_clean_sky"] = int(bucket == "clean_sky")
            out["comp_bucket_cloud_sky_texture"] = int(bucket == "cloud_sky_texture")
            out["comp_bucket_surface_texture"] = int(bucket == "surface_texture")
            out["comp_bucket_attached_linear"] = int(bucket == "attached_linear")
            out["comp_bucket_skyline_boundary"] = int(bucket == "skyline_boundary")
            out["comp_frame_control_count"] = len(controls["frame"])
            out["comp_context_control_count"] = len(controls["context"])
            out["comp_near_control_count"] = len(controls["near"])
            for metric in metrics:
                value = row_metric(row, metric)
                out[f"comp_{metric}_raw"] = round(value, 6)
                for scope, scope_rows in controls.items():
                    vals = [row_metric(r, metric) for r in scope_rows]
                    z, med, sigma, best_margin = robust_z(value, vals, sigma_floor)
                    out[f"comp_{metric}_{scope}_z"] = round(z, 6)
                    out[f"comp_{metric}_{scope}_median"] = round(med, 6)
                    out[f"comp_{metric}_{scope}_sigma"] = round(sigma, 6)
                    out[f"comp_{metric}_{scope}_best_margin"] = round(best_margin, 6)
                    out[f"comp_{metric}_{scope}_pct_rank"] = round(percentile_rank(value, vals), 6)
            out_rows.append(out)
    out_rows.sort(key=lambda r: (safe_int(r.get("frame"), 0), safe_int(r.get("rank"), 999999)))
    return out_rows


def process_file(
    in_csv: Path,
    out_csv: Path,
    max_rank: int,
    near_radius_px: float,
    min_context_controls: int,
    sigma_floor: float,
    metrics: tuple[str, ...],
) -> None:
    rows = read_csv(in_csv)
    out_rows = add_competition_features(rows, max_rank, near_radius_px, min_context_controls, sigma_floor, metrics)
    write_csv(out_csv, out_rows)
    meta_src = in_csv.parent / "clba_augment_metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, out_csv.parent / "clba_augment_metadata.json")


def main() -> None:
    args = parse_args()
    metrics = tuple(x.strip() for x in args.metrics.split(",") if x.strip())
    unknown = [x for x in metrics if x not in {"target_margin", "gain_norm", "proposal_score"}]
    if unknown:
        raise SystemExit(f"unknown --metrics values: {', '.join(unknown)}")
    if args.top_tubes:
        if not args.out_csv:
            raise SystemExit("--out_csv is required with --top_tubes")
        process_file(
            Path(args.top_tubes),
            Path(args.out_csv),
            args.max_rank,
            args.near_radius_px,
            args.min_context_controls,
            args.sigma_floor,
            metrics,
        )
        print(args.out_csv)
        return

    if not args.out_dir:
        raise SystemExit("--out_dir is required with --results_dir")
    in_root = Path(args.results_dir)
    out_root = Path(args.out_dir)
    count = 0
    for in_csv in sorted(in_root.glob("*/top_tubes.csv")):
        rel = in_csv.relative_to(in_root)
        process_file(
            in_csv,
            out_root / rel,
            args.max_rank,
            args.near_radius_px,
            args.min_context_controls,
            args.sigma_floor,
            metrics,
        )
        count += 1
    if count == 0:
        raise SystemExit(f"no */top_tubes.csv files found under {in_root}")
    print(f"{count} files -> {out_root}")


if __name__ == "__main__":
    main()
