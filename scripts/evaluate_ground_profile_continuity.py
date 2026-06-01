#!/usr/bin/env python3
"""Evaluate profile continuity on visually reviewed ground/surface labels.

This is intentionally stricter than the general tracking evaluator: it filters
to visible non-sky labels, reports consecutive labeled-frame runs, and can
compare many historical profile outputs that use slightly different CSV shapes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import fnmatch
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORE_GROUND_SPLITS = {
    "textured_non_sky",
    "surface_terrain_tree_line",
}
TERRAIN_ONLY_SPLITS = {
    "surface_terrain_tree_line",
}
BOUNDARY_GROUND_SPLITS = CORE_GROUND_SPLITS | {
    "surface_horizon_manual",
    "skyline_surface",
}
TARGET_BACKDROPS = {
    "clean_sky",
    "cloud_sky",
    "skyline_above_terrain",
    "vegetation",
    "tree_canopy",
    "terrain",
    "road",
    "mixed_ground",
    "unknown",
}
TRUE_GROUND_BACKDROPS = {
    "vegetation",
    "tree_canopy",
    "terrain",
    "road",
    "mixed_ground",
}
SKYLINE_BACKDROPS = {
    "skyline_above_terrain",
}
AUDIT_STATUSES = {
    "visual_confirmed",
    "contact_sheet_reviewed",
    "interpolated",
    "weak_vision_assisted",
    "rejected",
}
TRUE_GROUND_AUDIT_STATUSES = {
    "visual_confirmed",
    "contact_sheet_reviewed",
}
LABEL_SCHEMA_FIELDS = (
    "target_backdrop",
    "frame_context",
    "audit_status",
    "label_provenance",
    "evidence_class",
    "exclude_from_true_ground",
)
DEFAULT_BACKDROP_MANIFEST = Path("configs/target_backdrop_corrections_2026_05_31.csv")


DEFAULT_LABELS = [
    "artifacts/full_sample_failure_taxonomy_v1/labels_plus_failure_7bd_gap_v1.csv",
    "artifacts/cs_js2_multiclass_g_v2_aaf1_manual_v1/focused_aaf1_gap/labels_aaf1_354_478.csv",
]


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    template: str


@dataclass(frozen=True)
class BackdropManifestEntry:
    path: str
    row_num: int
    clip_pattern: str
    frame_start: int | None
    frame_end: int | None
    updates: dict[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", action="append", default=[], help="Label CSV. Defaults to current surface label sets.")
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--view",
        choices=("terrain_only", "core_ground", "ground_plus_boundary", "true_ground", "skyline_above_terrain"),
        default="core_ground",
        help=(
            "terrain_only keeps only hand-reviewed surface/terrain/tree-line labels; "
            "core_ground also includes textured non-sky; ground_plus_boundary includes skyline/boundary labels; "
            "true_ground requires a backdrop manifest and keeps only audited target-backed-by-ground rows; "
            "skyline_above_terrain requires a backdrop manifest and keeps audited target-backed-by-skyline rows."
        ),
    )
    p.add_argument(
        "--backdrop_manifest",
        "--label_manifest",
        dest="backdrop_manifest",
        action="append",
        default=[],
        help=(
            "CSV manifest assigning target_backdrop/frame_context/audit_status by clip/frame range. "
            f"Defaults to {DEFAULT_BACKDROP_MANIFEST} when present, and is required for --view true_ground."
        ),
    )
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Optional NAME=TEMPLATE where TEMPLATE may contain {clip}. If omitted, default historical profiles are discovered.",
    )
    p.add_argument("--min_profile_frames", type=int, default=20)
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_file_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths if path.exists()}


def joined_values(rows: list[dict[str, Any]], field: str, default: str = "unknown") -> str:
    values = sorted({str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()})
    return "+".join(values) if values else default


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def parse_frame_bound(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    number = fnum(value)
    if number is None:
        raise ValueError(f"invalid frame bound {value!r}")
    return int(number)


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def truthy(value: Any) -> bool:
    raw = str(value).strip().lower()
    return raw not in {"", "0", "false", "no", "none", "nan", "not_visible", "not visible"}


def resolve_manifest_paths(items: list[str], view: str) -> list[Path]:
    if items:
        return [Path(p) for p in items]
    if DEFAULT_BACKDROP_MANIFEST.exists():
        return [DEFAULT_BACKDROP_MANIFEST]
    if view in {"true_ground", "skyline_above_terrain"}:
        raise SystemExit(f"--view {view} requires --backdrop_manifest or {DEFAULT_BACKDROP_MANIFEST}")
    return []


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return ""


def normalize_bool_flag(value: Any) -> str:
    raw = str(value or "").strip()
    if raw == "":
        return ""
    return "1" if truthy(raw) else "0"


def load_backdrop_manifest(paths: list[Path]) -> list[BackdropManifestEntry]:
    entries: list[BackdropManifestEntry] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"backdrop manifest not found: {path}")
        for row_num, row in enumerate(read_csv(path), start=2):
            clip_pattern = first_present(row, ("clip_pattern", "clip", "clip_glob"))
            if not clip_pattern:
                raise ValueError(f"{path}:{row_num} missing clip_pattern")
            target_backdrop = normalize_label(row.get("target_backdrop", ""))
            if not target_backdrop:
                raise ValueError(f"{path}:{row_num} missing target_backdrop")
            if target_backdrop not in TARGET_BACKDROPS:
                raise ValueError(f"{path}:{row_num} invalid target_backdrop {target_backdrop!r}")
            audit_status = normalize_label(row.get("audit_status", ""))
            if audit_status and audit_status not in AUDIT_STATUSES:
                raise ValueError(f"{path}:{row_num} invalid audit_status {audit_status!r}")
            updates = {
                "target_backdrop": target_backdrop,
                "label_provenance": first_present(row, ("label_provenance", "provenance")) or f"manifest:{path.name}:{row_num}",
            }
            for field in ("frame_context", "audit_status", "evidence_class"):
                value = normalize_label(row.get(field, ""))
                if value:
                    updates[field] = value
            exclude_flag = normalize_bool_flag(row.get("exclude_from_true_ground", ""))
            if exclude_flag:
                updates["exclude_from_true_ground"] = exclude_flag
            entries.append(
                BackdropManifestEntry(
                    path=str(path),
                    row_num=row_num,
                    clip_pattern=clip_pattern,
                    frame_start=parse_frame_bound(first_present(row, ("frame_start", "start_frame", "start"))),
                    frame_end=parse_frame_bound(first_present(row, ("frame_end", "end_frame", "end"))),
                    updates=updates,
                )
            )
    return entries


def apply_backdrop_manifest(row: dict[str, Any], entries: list[BackdropManifestEntry]) -> dict[str, Any]:
    out = dict(row)
    for field in LABEL_SCHEMA_FIELDS:
        out.setdefault(field, "")
    clip = str(out.get("clip", "")).strip()
    frame = fnum(out.get("frame"))
    if not clip or frame is None:
        return out
    frame_int = int(frame)
    for entry in entries:
        if not fnmatch.fnmatchcase(clip, entry.clip_pattern):
            continue
        if entry.frame_start is not None and frame_int < entry.frame_start:
            continue
        if entry.frame_end is not None and frame_int > entry.frame_end:
            continue
        out.update(entry.updates)
    return out


def is_true_ground_label(row: dict[str, Any]) -> bool:
    if truthy(row.get("exclude_from_true_ground", "")):
        return False
    return (
        normalize_label(row.get("target_backdrop", "")) in TRUE_GROUND_BACKDROPS
        and normalize_label(row.get("audit_status", "")) in TRUE_GROUND_AUDIT_STATUSES
    )


def is_skyline_above_terrain_label(row: dict[str, Any]) -> bool:
    return (
        normalize_label(row.get("target_backdrop", "")) in SKYLINE_BACKDROPS
        and normalize_label(row.get("audit_status", "")) in TRUE_GROUND_AUDIT_STATUSES
    )


def row_is_selected(row: dict[str, Any]) -> bool:
    if "selected" in row and not truthy(row.get("selected")):
        return False
    x = fnum(row.get("x"))
    y = fnum(row.get("y"))
    return x is not None and y is not None


def bbox_from_row(row: dict[str, Any], prefix: str = "") -> tuple[float, float, float, float] | None:
    x = fnum(row.get(f"{prefix}x"))
    y = fnum(row.get(f"{prefix}y"))
    w = fnum(row.get(f"{prefix}w"), 1.0)
    h = fnum(row.get(f"{prefix}h"), 1.0)
    if x is None or y is None or w is None or h is None:
        return None
    return x, y, max(1.0, w), max(1.0, h)


def label_bbox(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for prefix in ("det_", "label_", ""):
        box = bbox_from_row(row, prefix)
        if box is not None:
            return box
    return None


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax = a[0] + 0.5 * a[2]
    ay = a[1] + 0.5 * a[3]
    bx = b[0] + 0.5 * b[2]
    by = b[1] + 0.5 * b[3]
    return float(math.hypot(ax - bx, ay - by))


def source_priority(row: dict[str, Any]) -> tuple[int, int, int]:
    source = str(row.get("source", ""))
    confidence = str(row.get("confidence", "")).lower()
    vision_conf = str(row.get("vision_confidence", "")).lower()
    manual = 2 if "vision_manual" in source or "frame_by_frame" in source else 0
    conf = 2 if confidence == "high" or "vision_high" in vision_conf else 1 if "medium" in confidence else 0
    dense = 1 if "surface_audit" in source or "terrain" in source else 0
    return manual, conf, dense


def load_ground_labels(
    paths: list[Path],
    view: str,
    backdrop_manifest_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    manifest_entries = load_backdrop_manifest(backdrop_manifest_paths or [])
    if view in {"true_ground", "skyline_above_terrain"}:
        if not manifest_entries:
            raise ValueError(f"{view} view requires at least one backdrop manifest")
        splits: set[str] | None = None
    elif view == "terrain_only":
        splits = TERRAIN_ONLY_SPLITS
    elif view == "core_ground":
        splits = CORE_GROUND_SPLITS
    else:
        splits = BOUNDARY_GROUND_SPLITS
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            clip = str(row.get("clip", "")).strip()
            frame = fnum(row.get("frame"))
            if not clip or frame is None:
                continue
            if not truthy(row.get("visible", "")):
                continue
            clean = apply_backdrop_manifest(row, manifest_entries)
            if view == "true_ground":
                if not is_true_ground_label(clean):
                    continue
            elif view == "skyline_above_terrain":
                if not is_skyline_above_terrain_label(clean):
                    continue
            else:
                split = str(clean.get("bg_split", "")).strip()
                if split not in splits:
                    continue
            box = label_bbox(clean)
            if box is None:
                continue
            key = (clip, int(frame))
            clean["clip"] = clip
            clean["frame"] = int(frame)
            clean["det_x"], clean["det_y"], clean["det_w"], clean["det_h"] = box
            clean["label_file"] = str(path)
            for field in LABEL_SCHEMA_FIELDS:
                clean.setdefault(field, "")
            prev = by_key.get(key)
            if prev is None or source_priority(clean) >= source_priority(prev):
                by_key[key] = clean
    return sorted(by_key.values(), key=lambda r: (str(r["clip"]), int(r["frame"])))


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_profile_specs(items: list[str]) -> list[ProfileSpec]:
    specs: list[ProfileSpec] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--profile must be NAME=TEMPLATE, got {item!r}")
        name, template = item.split("=", 1)
        specs.append(ProfileSpec(sanitize_name(name), template))
    return specs


def discover_profiles() -> list[ProfileSpec]:
    specs = [
        ProfileSpec("current_profile_top_tubes", "artifacts/current_profile_top_tubes_2026_05_29/{clip}/selected_tracks.csv"),
        ProfileSpec("surface_stack_all", "artifacts/surface_stack_profile_2026_05_29/all_stack_results/{clip}/selected_tracks.csv"),
        ProfileSpec("crop_ranker_score", "artifacts/crop_stack_ranker_top20_selector_probe_v1/crop_ranker_score/{clip}/sequence_selected_tracks.csv"),
        ProfileSpec("crop_ranker_viterbi_w9", "artifacts/crop_stack_ranker_top20_selector_probe_v1/crop_ranker_viterbi_w9/{clip}/sequence_selected_tracks.csv"),
        ProfileSpec("crop_ranker_hmm", "artifacts/crop_stack_ranker_top20_selector_probe_v1/crop_ranker_hmm/{clip}/sequence_selected_tracks.csv"),
        ProfileSpec("aaf1_recovery_local_seed24_halo3", "artifacts/proposal_recovery_aaf1_v1/local_seed24_halo3/{clip}/selected_tracks.csv"),
        ProfileSpec("aaf1_recovery_local_seed36_halo5", "artifacts/proposal_recovery_aaf1_v1/local_seed36_halo5/{clip}/selected_tracks.csv"),
        ProfileSpec("aaf1_recovery_full_stack_upper", "artifacts/proposal_recovery_aaf1_v1/full_stack_upper/{clip}/selected_tracks.csv"),
    ]
    for root in [
        Path("artifacts/surface_selector_mode_eval_e6_aaf1_hard_w15_v1"),
        Path("artifacts/hmm_scoremode_probe_v1"),
    ]:
        if not root.exists():
            continue
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            filename = "sequence_selected_tracks.csv" if "surface_selector" in str(root) else "selected_tracks.csv"
            specs.append(ProfileSpec(f"{root.name}_{child.name}", str(child / "{clip}" / filename)))
    specs.extend(discover_gated_js1_profiles(Path("artifacts/cs_js2_multiclass_g_v3_terrain_precision")))
    fast = Path("artifacts/cs_js2_multiclass_g_v3_terrain_precision/js1_loco_label_frames_cont_v1_fast")
    if fast.exists():
        specs.append(ProfileSpec("cs_js2_loco_label_frames_cont_fast", str(fast / "{clip}" / "best_frame_predictions.csv")))
    fast_k80 = Path("artifacts/cs_js2_multiclass_g_v3_terrain_precision/js1_loco_label_frames_cont_v1_fast_k80")
    if fast_k80.exists():
        specs.append(ProfileSpec("cs_js2_loco_label_frames_cont_fast_k80", str(fast_k80 / "{clip}" / "best_frame_predictions.csv")))
    return specs


def discover_gated_js1_profiles(root: Path) -> list[ProfileSpec]:
    """Discover routed surface-branch JS1 replay outputs.

    Gated branch experiments have been generated in a few directory layouts:
    flat one-config runs such as `gated_surface_branch_v1/top5_keep3/...`, and
    risk sweeps such as `gated_surface_branch_v2_repeated_risk_sweep/risk07/...`.
    The stable signal is the per-clip `js1_eval/best_frame_predictions.csv`.
    """

    if not root.exists():
        return []
    specs: list[ProfileSpec] = []
    seen: set[str] = set()
    for path in sorted(root.glob("gated_surface_branch*/**/js1_eval/best_frame_predictions.csv")):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 4:
            continue
        clip_idx = parts.index("js1_eval") - 1
        if clip_idx <= 0:
            continue
        run_parts = parts[:clip_idx]
        clip = parts[clip_idx]
        if "{" in clip or "}" in clip:
            continue
        template = str(root.joinpath(*run_parts) / "{clip}" / "js1_eval" / "best_frame_predictions.csv")
        if template in seen:
            continue
        seen.add(template)
        name = sanitize_name("cs_js2_" + "_".join(run_parts))
        specs.append(ProfileSpec(name, template))
    return specs


def load_profile_rows(spec: ProfileSpec, clips: list[str]) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, str]]:
    rows_by_clip: dict[str, dict[int, dict[str, Any]]] = {}
    paths_by_clip: dict[str, str] = {}
    for clip in clips:
        path = Path(spec.template.format(clip=clip))
        if not path.exists():
            continue
        frame_rows: dict[int, dict[str, Any]] = {}
        for row in read_csv(path):
            frame = fnum(row.get("frame"))
            if frame is None:
                continue
            frame_rows[int(frame)] = dict(row)
        rows_by_clip[clip] = frame_rows
        paths_by_clip[clip] = str(path)
    return rows_by_clip, paths_by_clip


def evaluate_profile(
    spec: ProfileSpec,
    labels: list[dict[str, Any]],
    strict_tol_px: float,
    loose_tol_px: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clips = sorted({str(r["clip"]) for r in labels})
    profile_rows, paths_by_clip = load_profile_rows(spec, clips)
    frame_eval: list[dict[str, Any]] = []
    for lab in labels:
        clip = str(lab["clip"])
        frame = int(lab["frame"])
        rows = profile_rows.get(clip)
        if rows is None:
            continue
        pred = rows.get(frame)
        label_box = label_bbox(lab)
        selected = False
        strict = False
        loose = False
        dist: float | None = None
        sx = sy = sw = sh = ""
        if pred is not None and {"x", "y"}.issubset(pred):
            selected = row_is_selected(pred)
            pred_box = bbox_from_row(pred) if selected else None
            if pred_box is not None and label_box is not None:
                sx, sy, sw, sh = pred_box
                dist = center_dist(pred_box, label_box)
                strict = dist <= strict_tol_px
                loose = dist <= loose_tol_px
        elif pred is not None and "strict_hit" in pred and "loose_hit" in pred:
            # Some JS1 replay files only retain hit flags/dist, not selected coordinates.
            selected = truthy(pred.get("selected", ""))
            strict = truthy(pred.get("strict_hit", ""))
            loose = truthy(pred.get("loose_hit", ""))
            dist = fnum(pred.get("dist_px"))
        frame_eval.append(
            {
                "profile": spec.name,
                "clip": clip,
                "frame": frame,
                "bg_split": lab.get("bg_split", ""),
                "target_backdrop": lab.get("target_backdrop", ""),
                "frame_context": lab.get("frame_context", ""),
                "audit_status": lab.get("audit_status", ""),
                "label_provenance": lab.get("label_provenance", ""),
                "evidence_class": lab.get("evidence_class", ""),
                "exclude_from_true_ground": lab.get("exclude_from_true_ground", ""),
                "label_source": lab.get("source", lab.get("label_file", "")),
                "confidence": lab.get("confidence", ""),
                "selected": int(selected),
                "strict_hit": int(strict),
                "loose_hit": int(loose),
                "dist_px": "" if dist is None else round(dist, 3),
                "det_x": round(float(lab["det_x"]), 3),
                "det_y": round(float(lab["det_y"]), 3),
                "det_w": round(float(lab["det_w"]), 3),
                "det_h": round(float(lab["det_h"]), 3),
                "selected_x": sx,
                "selected_y": sy,
                "selected_w": sw,
                "selected_h": sh,
                "profile_path": paths_by_clip.get(clip, ""),
            }
        )
    by_clip = summarize_by_clip(spec.name, frame_eval)
    summary = summarize_profile(spec.name, frame_eval, labels, paths_by_clip)
    return frame_eval, by_clip, summary


def run_lengths(rows: list[dict[str, Any]], key: str) -> tuple[int, int, int]:
    best = 0
    current = 0
    best_start = -1
    current_start = -1
    prev_frame: int | None = None
    for row in sorted(rows, key=lambda r: int(r["frame"])):
        frame = int(row["frame"])
        hit = bool(int(row[key]))
        consecutive = prev_frame is not None and frame == prev_frame + 1
        if hit:
            if current == 0 or not consecutive:
                current = 1
                current_start = frame
            else:
                current += 1
            if current > best:
                best = current
                best_start = current_start
        else:
            current = 0
            current_start = -1
        prev_frame = frame
    best_end = best_start + best - 1 if best_start >= 0 else -1
    return best, best_start, best_end


def miss_gap_lengths(rows: list[dict[str, Any]], key: str) -> tuple[int, int, int]:
    best = 0
    current = 0
    best_start = -1
    current_start = -1
    prev_frame: int | None = None
    for row in sorted(rows, key=lambda r: int(r["frame"])):
        frame = int(row["frame"])
        miss = not bool(int(row[key]))
        consecutive = prev_frame is not None and frame == prev_frame + 1
        if miss:
            if current == 0 or not consecutive:
                current = 1
                current_start = frame
            else:
                current += 1
            if current > best:
                best = current
                best_start = current_start
        else:
            current = 0
            current_start = -1
        prev_frame = frame
    best_end = best_start + best - 1 if best_start >= 0 else -1
    return best, best_start, best_end


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    strict = sum(int(r["strict_hit"]) for r in rows)
    loose = sum(int(r["loose_hit"]) for r in rows)
    selected = sum(int(r["selected"]) for r in rows)
    wrong = sum(1 for r in rows if int(r["selected"]) and not int(r["loose_hit"]))
    strict_run, strict_start, strict_end = run_lengths(rows, "strict_hit")
    loose_run, loose_start, loose_end = run_lengths(rows, "loose_hit")
    strict_gap, strict_gap_start, strict_gap_end = miss_gap_lengths(rows, "strict_hit")
    return {
        "ground_label_frames": n,
        "selected_frames": selected,
        "selected_rate": round(selected / max(1, n), 4),
        "strict_hits": strict,
        "strict_recall": round(strict / max(1, n), 4),
        "loose_hits": loose,
        "loose_recall": round(loose / max(1, n), 4),
        "selected_wrong_loose_frames": wrong,
        "selected_wrong_loose_rate": round(wrong / max(1, n), 4),
        "longest_strict_run": strict_run,
        "longest_strict_run_start": strict_start if strict_start >= 0 else "",
        "longest_strict_run_end": strict_end if strict_end >= 0 else "",
        "longest_loose_run": loose_run,
        "longest_loose_run_start": loose_start if loose_start >= 0 else "",
        "longest_loose_run_end": loose_end if loose_end >= 0 else "",
        "longest_strict_miss_gap": strict_gap,
        "longest_strict_miss_gap_start": strict_gap_start if strict_gap_start >= 0 else "",
        "longest_strict_miss_gap_end": strict_gap_end if strict_gap_end >= 0 else "",
    }


def summarize_by_clip(profile: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    clips = sorted({str(r["clip"]) for r in rows})
    for clip in clips:
        clip_rows = [r for r in rows if r["clip"] == clip]
        row = {
            "profile": profile,
            "clip": clip,
            "evidence_class": joined_values(clip_rows, "evidence_class", "unlabeled"),
            "target_backdrops": joined_values(clip_rows, "target_backdrop", "legacy_bg_split"),
            "audit_statuses": joined_values(clip_rows, "audit_status", "unknown"),
        }
        row.update(summarize_rows(clip_rows))
        out.append(row)
    return out


def summarize_profile(
    profile: str,
    rows: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    paths_by_clip: dict[str, str],
) -> dict[str, Any]:
    row = {
        "profile": profile,
        "evidence_class": joined_values(rows, "evidence_class", "unlabeled"),
        "target_backdrops": joined_values(rows, "target_backdrop", "legacy_bg_split"),
        "audit_statuses": joined_values(rows, "audit_status", "unknown"),
    }
    row.update(summarize_rows(rows))
    total_labels = len(labels)
    row["total_ground_label_frames"] = total_labels
    row["covered_ground_label_frames"] = len(rows)
    row["coverage_rate"] = round(len(rows) / max(1, total_labels), 4)
    row["covered_clips"] = len(paths_by_clip)
    row["profile_paths"] = json.dumps(paths_by_clip, sort_keys=True)
    return row


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_paths = [Path(p) for p in (args.label or DEFAULT_LABELS)]
    manifest_paths = resolve_manifest_paths(args.backdrop_manifest, args.view)
    label_hashes = existing_file_hashes(label_paths)
    manifest_hashes = existing_file_hashes(manifest_paths)
    labels = load_ground_labels(label_paths, args.view, manifest_paths)
    write_csv(out_dir / "ground_visible_labels.csv", labels)
    label_manifest = {
        "view": args.view,
        "labels": len(labels),
        "label_files": [str(p) for p in label_paths],
        "label_sha256": label_hashes,
        "backdrop_manifests": [str(p) for p in manifest_paths],
        "backdrop_manifest_sha256": manifest_hashes,
        "rows_by_clip": {},
        "rows_by_backdrop": {},
        "rows_by_evidence_class": {},
        "rows_by_audit_status": {},
    }
    for row in labels:
        for key, field in (
            ("rows_by_clip", "clip"),
            ("rows_by_backdrop", "target_backdrop"),
            ("rows_by_evidence_class", "evidence_class"),
            ("rows_by_audit_status", "audit_status"),
        ):
            value = str(row.get(field, "") or "unknown")
            label_manifest[key][value] = int(label_manifest[key].get(value, 0)) + 1
    (out_dir / "label_manifest.json").write_text(json.dumps(label_manifest, indent=2, sort_keys=True) + "\n")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "command": sys.argv,
                "view": args.view,
                "strict_tol_px": args.strict_tol_px,
                "loose_tol_px": args.loose_tol_px,
                "min_profile_frames": args.min_profile_frames,
                "label_manifest": label_manifest,
                "profile_args": args.profile,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    specs = parse_profile_specs(args.profile) if args.profile else discover_profiles()
    frame_dir = out_dir / "frame_eval"
    clip_dir = out_dir / "by_clip"
    all_summary: list[dict[str, Any]] = []
    all_clip: list[dict[str, Any]] = []
    for spec in specs:
        frame_rows, clip_rows, summary = evaluate_profile(spec, labels, args.strict_tol_px, args.loose_tol_px)
        if summary["covered_ground_label_frames"] < args.min_profile_frames:
            continue
        summary["evaluation_view"] = args.view
        summary["label_files"] = json.dumps([str(p) for p in label_paths])
        summary["backdrop_manifests"] = json.dumps([str(p) for p in manifest_paths])
        safe = sanitize_name(spec.name)
        write_csv(frame_dir / f"{safe}.csv", frame_rows)
        for row in clip_rows:
            row["evaluation_view"] = args.view
        write_csv(clip_dir / f"{safe}.csv", clip_rows)
        all_summary.append(summary)
        all_clip.extend(clip_rows)

    all_summary.sort(
        key=lambda r: (
            float(r["coverage_rate"]),
            int(r["longest_strict_run"]),
            float(r["strict_recall"]),
            -float(r["selected_wrong_loose_rate"]),
        ),
        reverse=True,
    )
    write_csv(out_dir / "profile_summary.csv", all_summary)
    write_csv(out_dir / "by_clip_summary.csv", all_clip)
    if args.view == "true_ground":
        view_note = (
            "true_ground keeps only visible rows whose target_backdrop is "
            "vegetation/tree_canopy/terrain/road/mixed_ground and whose audit_status "
            "is visual_confirmed/contact_sheet_reviewed."
        )
    elif args.view == "skyline_above_terrain":
        view_note = (
            "skyline_above_terrain keeps only visible rows whose target_backdrop is "
            "skyline_above_terrain and whose audit_status is visual_confirmed/contact_sheet_reviewed."
        )
    else:
        view_note = "Legacy views keep visible rows selected by bg_split."
    (out_dir / "README.md").write_text(
        "# Ground Frame Continuity Benchmark\n\n"
        f"View: `{args.view}`. {view_note} "
        f"Backdrop manifests: `{', '.join(str(p) for p in manifest_paths) or 'none'}`. "
        "The main metric is consecutive labeled-frame continuity: longest strict/loose hit runs. "
        "Strict is center distance <= 8 px; loose is <= 16 px.\n\n"
        "Files:\n"
        "- `ground_visible_labels.csv`: deduplicated labels used for this run.\n"
        "- `profile_summary.csv`: profile-level continuity and hit metrics.\n"
        "- `by_clip_summary.csv`: same metrics split by clip.\n"
        "- `frame_eval/`: per-profile frame-level hit/miss rows.\n"
        "- `metadata.json`: command, view, thresholds, and input checksums.\n"
        "- `label_manifest.json`: label counts by clip/backdrop/evidence class/audit status.\n"
    )
    print(json.dumps({"labels": len(labels), "profiles": len(all_summary), "out_dir": str(out_dir)}, indent=2))
    if all_summary:
        print(json.dumps(all_summary[0], indent=2))


if __name__ == "__main__":
    main()
