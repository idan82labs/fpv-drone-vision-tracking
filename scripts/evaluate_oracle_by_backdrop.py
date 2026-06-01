#!/usr/bin/env python3
"""Evaluate candidate oracle recall split by target backdrop.

This is the Phase 2 gate for the against-ground plan. It answers whether a
candidate stream contains the labeled target before selector/ranker behavior is
considered. Use it before tuning a selector: if oracle is low, the issue is
proposal recovery; if oracle is high and selected recall is low, the issue is
observation/routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_ground_profile_continuity import (
    DEFAULT_BACKDROP_MANIFEST,
    center_dist,
    existing_file_hashes,
    fnum,
    label_bbox,
    load_ground_labels,
    resolve_manifest_paths,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    template: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", action="append", default=[], help="Label CSV. May be repeated.")
    p.add_argument(
        "--view",
        choices=("terrain_only", "core_ground", "ground_plus_boundary", "true_ground", "skyline_above_terrain"),
        default="true_ground",
        help="Label view to evaluate. true_ground and skyline_above_terrain require a backdrop manifest.",
    )
    p.add_argument(
        "--backdrop_manifest",
        "--label_manifest",
        dest="backdrop_manifest",
        action="append",
        default=[],
        help=f"Backdrop manifest. Defaults to {DEFAULT_BACKDROP_MANIFEST} when present.",
    )
    p.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate stream as NAME=TEMPLATE, where TEMPLATE may contain {clip}. May be repeated.",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", action="append", type=int, default=[], help="Oracle K threshold. May be repeated.")
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--max_rank", type=int, default=1000000, help="Do not load rows above this rank.")
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


def parse_candidate_specs(items: list[str]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--candidate must be NAME=TEMPLATE, got {raw!r}")
        name, template = raw.split("=", 1)
        name = name.strip()
        template = template.strip()
        if not name or not template:
            raise SystemExit(f"--candidate must be NAME=TEMPLATE, got {raw!r}")
        specs.append(CandidateSpec(name=name, template=template))
    if not specs:
        raise SystemExit("at least one --candidate NAME=TEMPLATE is required")
    return specs


def candidate_bbox(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Read common candidate bbox shapes.

    Most top-tube exports use x/y/w/h. Some selected-frame exports use
    pred_x/pred_y/pred_w/pred_h or selected_x/...; support those so the script
    can also make it obvious when a selected-output stream is not oracle-capable.
    """

    for prefix in ("", "pred_", "selected_", "bbox_"):
        x = fnum(row.get(f"{prefix}x"))
        y = fnum(row.get(f"{prefix}y"))
        w = fnum(row.get(f"{prefix}w"), 1.0)
        h = fnum(row.get(f"{prefix}h"), 1.0)
        if x is not None and y is not None and w is not None and h is not None:
            return x, y, max(1.0, w), max(1.0, h)
    return None


def row_rank(row: dict[str, Any], fallback: int) -> int:
    rank = fnum(row.get("rank"))
    if rank is None:
        rank = fnum(row.get("candidate_rank"))
    if rank is None:
        rank = fnum(row.get("source_rank"))
    return fallback if rank is None else max(1, int(round(rank)))


def row_score(row: dict[str, Any]) -> float:
    for key in ("score", "verified_score", "surface_halo_score", "crop_score", "raw_score"):
        value = fnum(row.get(key))
        if value is not None:
            return value
    return 0.0


def load_candidates(
    spec: CandidateSpec,
    clips: list[str],
    max_rank: int,
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], dict[str, str], dict[str, int]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paths_by_clip: dict[str, str] = {}
    rows_by_clip: dict[str, int] = {}
    for clip in clips:
        path = Path(spec.template.format(clip=clip))
        if not path.exists():
            continue
        paths_by_clip[clip] = str(path)
        for fallback_rank, row in enumerate(read_csv(path), start=1):
            frame = fnum(row.get("frame"))
            if frame is None:
                continue
            rank = row_rank(row, fallback_rank)
            if rank > max_rank:
                continue
            box = candidate_bbox(row)
            if box is None:
                continue
            out = dict(row)
            out["clip"] = clip
            out["frame"] = int(frame)
            out["_rank"] = rank
            out["_score"] = row_score(row)
            out["_bbox"] = box
            by_key[(clip, int(frame))].append(out)
            rows_by_clip[clip] = rows_by_clip.get(clip, 0) + 1
    for rows in by_key.values():
        rows.sort(key=lambda r: (int(r["_rank"]), -float(r["_score"])))
    return by_key, paths_by_clip, rows_by_clip


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def evaluate_stream(
    spec: CandidateSpec,
    labels: list[dict[str, Any]],
    ks: list[int],
    strict_tol: float,
    loose_tol: float,
    max_rank: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clips = sorted({str(row["clip"]) for row in labels})
    candidates, paths_by_clip, rows_by_clip = load_candidates(spec, clips, max_rank)
    frame_rows: list[dict[str, Any]] = []
    for lab in labels:
        clip = str(lab["clip"])
        frame = int(lab["frame"])
        target = label_bbox(lab)
        if target is None:
            continue
        rows = candidates.get((clip, frame), [])
        best_dist = None
        best_rank = None
        best_loose_rank = None
        best_strict_rank = None
        candidate_count = 0
        for row in rows:
            candidate_count += 1
            dist = center_dist(target, row["_bbox"])
            rank = int(row["_rank"])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_rank = rank
            if dist <= loose_tol and (best_loose_rank is None or rank < best_loose_rank):
                best_loose_rank = rank
            if dist <= strict_tol and (best_strict_rank is None or rank < best_strict_rank):
                best_strict_rank = rank
        out: dict[str, Any] = {
            "stream": spec.name,
            "clip": clip,
            "frame": frame,
            "target_backdrop": lab.get("target_backdrop", ""),
            "frame_context": lab.get("frame_context", ""),
            "audit_status": lab.get("audit_status", ""),
            "evidence_class": lab.get("evidence_class", ""),
            "candidate_count": candidate_count,
            "best_dist": "" if best_dist is None else round(best_dist, 3),
            "best_rank": "" if best_rank is None else best_rank,
            "strict_rank": "" if best_strict_rank is None else best_strict_rank,
            "loose_rank": "" if best_loose_rank is None else best_loose_rank,
        }
        for k in ks:
            out[f"oracle_strict_at_{k}"] = int(best_strict_rank is not None and best_strict_rank <= k)
            out[f"oracle_loose_at_{k}"] = int(best_loose_rank is not None and best_loose_rank <= k)
        frame_rows.append(out)

    summary_rows = summarize(frame_rows, ks, group_fields=[])
    by_backdrop = summarize(frame_rows, ks, group_fields=["target_backdrop"])
    by_clip = summarize(frame_rows, ks, group_fields=["clip", "target_backdrop"])
    metadata = {
        "stream": spec.name,
        "template": spec.template,
        "paths_by_clip": paths_by_clip,
        "candidate_rows_by_clip": rows_by_clip,
        "missing_clips": [clip for clip in clips if clip not in paths_by_clip],
    }
    return frame_rows, summary_rows, by_backdrop + by_clip, metadata


def summarize(rows: list[dict[str, Any]], ks: list[int], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        row: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        stream_values = sorted({str(r.get("stream", "")) for r in group})
        row["stream"] = "+".join(stream_values)
        row["label_frames"] = len(group)
        row["frames_with_candidates"] = sum(int(r.get("candidate_count", 0)) > 0 for r in group)
        row["frames_with_candidates_rate"] = safe_rate(row["frames_with_candidates"], len(group))
        for k in ks:
            strict = sum(int(r.get(f"oracle_strict_at_{k}", 0)) for r in group)
            loose = sum(int(r.get(f"oracle_loose_at_{k}", 0)) for r in group)
            row[f"oracle_strict_at_{k}"] = strict
            row[f"oracle_strict_rate_at_{k}"] = safe_rate(strict, len(group))
            row[f"oracle_loose_at_{k}"] = loose
            row[f"oracle_loose_rate_at_{k}"] = safe_rate(loose, len(group))
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_paths = [Path(p) for p in args.label]
    manifest_paths = resolve_manifest_paths(args.backdrop_manifest, args.view)
    labels = load_ground_labels(labels_paths, args.view, manifest_paths)
    ks = sorted(set(args.k or [1, 5, 10, 20, 40, 80]))
    specs = parse_candidate_specs(args.candidate)

    all_frames: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []
    streams_meta: list[dict[str, Any]] = []
    for spec in specs:
        frame_rows, summary_rows, group_rows, metadata = evaluate_stream(
            spec,
            labels,
            ks,
            args.strict_tol_px,
            args.loose_tol_px,
            args.max_rank,
        )
        all_frames.extend(frame_rows)
        all_summary.extend(summary_rows)
        all_groups.extend(group_rows)
        streams_meta.append(metadata)

    write_csv(out_dir / "oracle_frame_eval.csv", all_frames)
    write_csv(out_dir / "oracle_summary.csv", all_summary)
    write_csv(out_dir / "oracle_by_backdrop.csv", [r for r in all_groups if "clip" not in r])
    write_csv(out_dir / "oracle_by_clip_backdrop.csv", [r for r in all_groups if "clip" in r])
    metadata = {
        "view": args.view,
        "strict_tol_px": args.strict_tol_px,
        "loose_tol_px": args.loose_tol_px,
        "ks": ks,
        "label_files": [str(p) for p in labels_paths],
        "backdrop_manifests": [str(p) for p in manifest_paths],
        "input_hashes": existing_file_hashes(labels_paths + manifest_paths),
        "label_frames": len(labels),
        "streams": streams_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
