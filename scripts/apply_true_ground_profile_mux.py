#!/usr/bin/env python3
"""Mux complementary surface-tracking profile outputs with continuity.

This is an offline experiment for verified true-ground frames. It does not
change runtime defaults. The intent is to test whether complementary evidence
from the current CS-JS2 replay, large-dark score084 path, and crop-ranker path
can recover terrain-backed spans that each profile misses alone.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from selector_core import SequenceItem, append_viterbi_layer, first_layer_selection, fnum

BBox = tuple[float, float, float, float]

DEFAULT_CS_TRACE = (
    "artifacts/cs_js2_multiclass_g_v3_terrain_precision/"
    "js1_loco_label_frames_cont_v1_fast/{clip}/best_frame_predictions.csv"
)
DEFAULT_CS_TUBES = (
    "artifacts/cs_js2_multiclass_g_v3_terrain_precision/"
    "scored_top_tubes_loco_v1/{clip}/top_tubes.csv"
)
DEFAULT_FALLBACKS = [
    (
        "score084",
        "artifacts/surface_selector_mode_eval_e6_aaf1_hard_w15_v1/"
        "score084/{clip}/sequence_selected_tracks.csv",
    ),
    (
        "cropw9",
        "artifacts/crop_stack_ranker_top20_selector_probe_v1/"
        "crop_ranker_viterbi_w9/{clip}/sequence_selected_tracks.csv",
    ),
]
DEFAULT_WEIGHTS = {
    "cs_js1": 3.25,
    "cs_proposal": 1.55,
    "score084": 2.85,
    "cropw9": 3.00,
}


@dataclass(frozen=True)
class CandidatePayload:
    clip: str
    frame: int
    bbox: BBox
    mux_source: str
    mux_score: float
    rank: str = ""
    learned_score: str = ""
    verified_score: str = ""
    source: str = ""
    track_id: str = ""
    reason: str = ""
    source_path: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", action="append", default=[], help="Clip id. May be repeated.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cs_trace_template", default=DEFAULT_CS_TRACE)
    p.add_argument("--cs_tubes_template", default=DEFAULT_CS_TUBES)
    p.add_argument(
        "--fallback",
        action="append",
        default=[],
        help="NAME=TEMPLATE for selected-track fallback CSVs. Defaults to score084 and cropw9.",
    )
    p.add_argument("--include_cs_proposals", type=int, default=3)
    p.add_argument("--proposal_weak_margin", type=float, default=1.0)
    p.add_argument("--max_jump_px", type=float, default=34.0)
    p.add_argument("--transition_weight", type=float, default=0.85)
    p.add_argument("--size_jump_weight", type=float, default=1.0)
    p.add_argument("--restart_penalty", type=float, default=0.6)
    p.add_argument("--min_emit_score", type=float, default=3.05)
    p.add_argument("--commit_lag", type=int, default=9)
    p.add_argument(
        "--bridge_gap_frames",
        type=int,
        default=0,
        help=(
            "Offline continuity bridge for short gaps between two emitted boxes. "
            "Disabled by default because it uses the future endpoint and is not a live policy."
        ),
    )
    p.add_argument("--bridge_score_floor", type=float, default=3.25)
    p.add_argument(
        "--target_local_recovery_frames",
        type=int,
        default=0,
        help="Causal recovery horizon after a recent emitted target track. Disabled by default.",
    )
    p.add_argument("--target_local_recovery_top_k", type=int, default=8)
    p.add_argument("--target_local_recovery_max_error", type=float, default=14.0)
    p.add_argument(
        "--target_local_recovery_last_anchor_px",
        type=float,
        default=0.0,
        help=(
            "Optional causal fallback radius around the last emitted target box. "
            "Useful for close terrain passes where one-frame velocity overshoots."
        ),
    )
    p.add_argument(
        "--target_local_recovery_anchor_mux_sources",
        default="",
        help=(
            "Optional comma-separated low-score mux_source allow-list required for last-anchor recovery. "
            "This leaves normal velocity recovery unchanged."
        ),
    )
    p.add_argument("--target_local_recovery_raw_min", type=float, default=10.0)
    p.add_argument(
        "--target_local_recovery_replace_existing_error",
        type=float,
        default=0.0,
        help=(
            "Opt-in selector override: when an already-emitted box is farther than this many "
            "pixels from the target-local motion prediction, allow a nearby top-tube candidate "
            "to replace it. Disabled at 0."
        ),
    )
    p.add_argument(
        "--target_local_recovery_replace_improvement_px",
        type=float,
        default=4.0,
        help="Minimum prediction-error improvement required before replacing an emitted box.",
    )
    p.add_argument(
        "--target_local_recovery_replace_min_side",
        type=float,
        default=0.0,
        help="Optional minimum candidate width/height for replacing an already-emitted box.",
    )
    p.add_argument(
        "--target_local_recovery_replace_top_k",
        type=int,
        default=0,
        help="Optional deeper rank window for replacing existing boxes. Defaults to recovery top_k.",
    )
    p.add_argument(
        "--target_local_recovery_replace_raw_min",
        type=float,
        default=None,
        help="Optional lower raw-score floor for replacing existing boxes. Defaults to recovery raw_min.",
    )
    p.add_argument("--recenter_emitted_boxes", action="store_true")
    p.add_argument("--recenter_top_k", type=int, default=80)
    p.add_argument("--recenter_max_delta_px", type=float, default=18.0)
    p.add_argument("--recenter_raw_min", type=float, default=-5.0)
    p.add_argument("--recenter_min_area_ratio", type=float, default=0.05)
    p.add_argument("--recenter_max_area_ratio", type=float, default=0.90)
    p.add_argument(
        "--recenter_t_margin_weight",
        type=float,
        default=0.0,
        help="Optional weight for crop T probability minus max S/E/H/G probability during recentering.",
    )
    p.add_argument(
        "--recenter_clutter_weight",
        type=float,
        default=0.0,
        help="Optional penalty for max S/E/H/G crop clutter probability during recentering.",
    )
    p.add_argument(
        "--recenter_tiny_area_px",
        type=float,
        default=0.0,
        help="Optional area threshold for tiny-candidate recenter penalty. Disabled at 0.",
    )
    p.add_argument(
        "--recenter_tiny_penalty",
        type=float,
        default=0.0,
        help="Optional score penalty for tiny recenter candidates.",
    )
    p.add_argument(
        "--recenter_min_score_gain",
        type=float,
        default=-1.0e9,
        help=(
            "Minimum recenter score gain over the current emitted item before replacing the box. "
            "The default preserves historical exploratory behavior."
        ),
    )
    p.add_argument(
        "--recenter_mux_sources",
        default="",
        help="Optional comma-separated mux_source allow-list for recentering, e.g. score084,target_local_recovery.",
    )
    p.add_argument(
        "--recenter_preserve_sources",
        default="",
        help=(
            "Optional comma-separated candidate source names to leave untouched during recentering "
            "when learned_score is at least --recenter_preserve_min_learned."
        ),
    )
    p.add_argument("--recenter_preserve_min_learned", type=float, default=1.1)
    p.add_argument(
        "--source_weight",
        action="append",
        default=[],
        help="Override source weight as NAME=FLOAT.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    return raw not in {"", "0", "false", "no", "none", "nan"}


def fint(value: Any, default: int = 0) -> int:
    out = fnum(value)
    return default if out is None else int(round(out))


def bbox_from_row(row: dict[str, Any]) -> BBox | None:
    x = fnum(row.get("x"))
    y = fnum(row.get("y"))
    w = fnum(row.get("w"), 1.0)
    h = fnum(row.get("h"), 1.0)
    if x is None or y is None or w is None or h is None:
        return None
    return float(x), float(y), max(1.0, float(w)), max(1.0, float(h))


def clip01(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def clipped(value: float | None, lo: float, hi: float, default: float = 0.0) -> float:
    if value is None:
        return default
    return max(lo, min(hi, float(value)))


def parse_name_templates(items: list[str]) -> list[tuple[str, str]]:
    if not items:
        return DEFAULT_FALLBACKS
    out: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--fallback must be NAME=TEMPLATE, got {item!r}")
        name, template = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--fallback has empty name: {item!r}")
        out.append((name, template))
    return out


def parse_weights(items: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--source_weight must be NAME=FLOAT, got {item!r}")
        name, raw = item.split("=", 1)
        weights[name.strip()] = float(raw)
    return weights


def candidate_from_payload(payload: CandidatePayload) -> SequenceItem:
    return SequenceItem(frame=payload.frame, bbox=payload.bbox, score=payload.mux_score, payload=payload)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return float(bbox[0]) + 0.5 * float(bbox[2]), float(bbox[1]) + 0.5 * float(bbox[3])


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(math.hypot(ax - bx, ay - by))


def bbox_area(bbox: BBox) -> float:
    return max(1.0, float(bbox[2])) * max(1.0, float(bbox[3]))


def selected_track_score(name: str, row: dict[str, str], weights: dict[str, float]) -> float:
    learned = fnum(row.get("learned_score"))
    verified = fnum(row.get("verified_score"))
    rank = fnum(row.get("rank"))
    source = str(row.get("source", "")).strip().lower()
    score = weights.get(name, 2.0)
    if learned is not None and 0.0 <= learned <= 1.0:
        score += 0.75 * learned
    if verified is not None:
        score += 0.55 * clipped(verified / 80.0, -0.5, 1.5)
    if source == "large_dark":
        score += 0.35
    elif source in {"appearance", "map"}:
        score += 0.12
    elif source:
        score += 0.04
    if rank is not None:
        score -= 0.08 * math.log1p(max(0.0, rank))
    return score


def js1_trace_score(row: dict[str, str], weights: dict[str, float]) -> float:
    margin = fnum(row.get("target_margin"))
    raw_score = fnum(row.get("raw_score"))
    rank = fnum(row.get("rank"))
    score = weights.get("cs_js1", 3.0)
    score += 0.38 * clipped(margin, -1.0, 3.5)
    score += 0.35 * clipped((raw_score or 0.0) / 30.0, -0.5, 1.5)
    if str(row.get("state", "")) == "T":
        score += 0.35
    if rank is not None:
        score -= 0.06 * math.log1p(max(0.0, rank))
    return score


def top_tube_score(row: dict[str, str], weights: dict[str, float]) -> float:
    raw_score = fnum(row.get("score"))
    verified = fnum(row.get("verified_score"))
    rank = fnum(row.get("rank"))
    t_prob = fnum(row.get("crop_t_prob"))
    clutter_probs = [
        fnum(row.get("crop_s_prob")),
        fnum(row.get("crop_e_prob")),
        fnum(row.get("crop_h_prob")),
        fnum(row.get("crop_g_prob")),
    ]
    max_clutter = max((p for p in clutter_probs if p is not None), default=0.0)
    source = str(row.get("cand_source", "")).strip().lower()
    score = weights.get("cs_proposal", 1.4)
    score += 0.35 * clipped((raw_score or 0.0) / 30.0, -0.5, 1.6)
    score += 0.35 * clipped((verified or 0.0) / 80.0, -0.5, 1.5)
    score += 0.85 * (clip01(t_prob) - 0.65 * clip01(max_clutter))
    if source == "large_dark":
        score += 0.35
    elif source in {"map", "appearance"}:
        score += 0.10
    elif source == "temporal_stack":
        score -= 0.10
    if rank is not None:
        score -= 0.14 * math.log1p(max(0.0, rank))
    return score


def load_tubes_by_frame(path: Path) -> dict[int, list[dict[str, str]]]:
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        frame = fnum(row.get("frame"))
        if frame is None:
            continue
        by_frame[int(frame)].append(row)
    for rows in by_frame.values():
        rows.sort(key=lambda r: fnum(r.get("rank"), 1.0e9) or 1.0e9)
    return by_frame


def add_cs_js1_candidates(
    clip: str,
    trace_path: Path,
    tubes_path: Path,
    weights: dict[str, float],
    frame_items: dict[int, list[SequenceItem]],
) -> None:
    tubes = load_tubes_by_frame(tubes_path)
    for trace in read_csv(trace_path):
        frame = fnum(trace.get("frame"))
        if frame is None or not truthy(trace.get("selected")):
            continue
        rank = fnum(trace.get("rank"))
        if rank is None:
            continue
        tube = next((r for r in tubes.get(int(frame), []) if fnum(r.get("rank")) == rank), None)
        if tube is None:
            continue
        bbox = bbox_from_row(tube)
        if bbox is None:
            continue
        payload = CandidatePayload(
            clip=clip,
            frame=int(frame),
            bbox=bbox,
            mux_source="cs_js1",
            mux_score=js1_trace_score(trace, weights),
            rank=str(trace.get("rank", "")),
            learned_score=str(tube.get("crop_t_prob", "")),
            verified_score=str(tube.get("verified_score", "")),
            source=str(tube.get("cand_source", "")),
            track_id=str(tube.get("track_id", "")),
            reason=str(trace.get("reason", "")),
            source_path=str(trace_path),
        )
        frame_items[int(frame)].append(candidate_from_payload(payload))


def add_cs_proposal_candidates(
    clip: str,
    trace_path: Path,
    tubes_path: Path,
    weights: dict[str, float],
    frame_items: dict[int, list[SequenceItem]],
    *,
    top_k: int,
    weak_margin: float,
) -> None:
    if top_k <= 0:
        return
    tubes = load_tubes_by_frame(tubes_path)
    trace_by_frame: dict[int, dict[str, str]] = {}
    for row in read_csv(trace_path):
        frame = fnum(row.get("frame"))
        if frame is not None:
            trace_by_frame[int(frame)] = row
    for frame, rows in tubes.items():
        trace = trace_by_frame.get(frame, {})
        selected = truthy(trace.get("selected", ""))
        state = str(trace.get("state", "") or "A")
        margin = fnum(trace.get("target_margin"))
        weak = (not selected) or state in {"A", "P", "C", "S", "E"} or (margin is not None and margin <= weak_margin)
        if not weak:
            continue
        for row in rows[:top_k]:
            bbox = bbox_from_row(row)
            if bbox is None:
                continue
            payload = CandidatePayload(
                clip=clip,
                frame=frame,
                bbox=bbox,
                mux_source="cs_proposal",
                mux_score=top_tube_score(row, weights),
                rank=str(row.get("rank", "")),
                learned_score=str(row.get("crop_t_prob", "")),
                verified_score=str(row.get("verified_score", "")),
                source=str(row.get("cand_source", "")),
                track_id=str(row.get("track_id", "")),
                reason="weak_trace_proposal",
                source_path=str(tubes_path),
            )
            frame_items[frame].append(candidate_from_payload(payload))


def add_selected_track_candidates(
    clip: str,
    name: str,
    path: Path,
    weights: dict[str, float],
    frame_items: dict[int, list[SequenceItem]],
) -> None:
    for row in read_csv(path):
        frame = fnum(row.get("frame"))
        if frame is None or not truthy(row.get("selected")):
            continue
        bbox = bbox_from_row(row)
        if bbox is None:
            continue
        payload = CandidatePayload(
            clip=clip,
            frame=int(frame),
            bbox=bbox,
            mux_source=name,
            mux_score=selected_track_score(name, row, weights),
            rank=str(row.get("rank", "")),
            learned_score=str(row.get("learned_score", "")),
            verified_score=str(row.get("verified_score", "")),
            source=str(row.get("source", "")),
            track_id=str(row.get("track_id", "")),
            reason="selected_track",
            source_path=str(path),
        )
        frame_items[int(frame)].append(candidate_from_payload(payload))


def dedupe_items(items: list[SequenceItem]) -> list[SequenceItem]:
    best: dict[tuple[int, int, int, int, str], SequenceItem] = {}
    for item in items:
        x, y, w, h = item.bbox
        source = getattr(item.payload, "mux_source", "")
        key = (round(x), round(y), round(w), round(h), str(source))
        prev = best.get(key)
        if prev is None or item.score > prev.score:
            best[key] = item
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def output_rows_for_clip(
    clip: str,
    selected: dict[int, SequenceItem],
    *,
    min_emit_score: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in sorted(selected):
        item = selected[frame]
        payload: CandidatePayload = item.payload
        emit = item.score >= min_emit_score
        x, y, w, h = payload.bbox
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "selected": int(emit),
                "rank": payload.rank,
                "learned_score": payload.learned_score,
                "threshold": min_emit_score,
                "x": round(x, 3) if emit else "",
                "y": round(y, 3) if emit else "",
                "w": round(w, 3) if emit else "",
                "h": round(h, 3) if emit else "",
                "verified_score": payload.verified_score,
                "source": payload.source,
                "track_id": payload.track_id,
                "mux_source": payload.mux_source,
                "mux_score": round(item.score, 6),
                "mux_reason": payload.reason,
                "source_path": payload.source_path,
            }
        )
    return rows


def interpolate_payload(
    prev: CandidatePayload,
    nxt: CandidatePayload,
    frame: int,
    *,
    score_floor: float,
) -> CandidatePayload:
    gap = max(1, int(nxt.frame) - int(prev.frame))
    alpha = (int(frame) - int(prev.frame)) / float(gap)
    bbox = tuple(
        (1.0 - alpha) * float(a) + alpha * float(b)
        for a, b in zip(prev.bbox, nxt.bbox)
    )
    score = max(float(score_floor), min(float(prev.mux_score), float(nxt.mux_score)) - 0.10 * min(alpha, 1.0 - alpha))
    return CandidatePayload(
        clip=prev.clip,
        frame=int(frame),
        bbox=(float(bbox[0]), float(bbox[1]), max(1.0, float(bbox[2])), max(1.0, float(bbox[3]))),
        mux_source="bridge",
        mux_score=score,
        rank="",
        learned_score=f"{score:.6f}",
        verified_score="",
        source="bridge",
        track_id=f"{prev.track_id}|{nxt.track_id}",
        reason=f"bridge_{prev.frame}_{nxt.frame}",
        source_path=f"{prev.source_path};{nxt.source_path}",
    )


def bridge_selected_gaps(
    selected: dict[int, SequenceItem],
    *,
    max_gap_frames: int,
    score_floor: float,
) -> dict[int, SequenceItem]:
    """Fill short gaps between emitted endpoint candidates.

    This is an offline smoothing experiment, not a live default. It only fills
    frames for which the mux produced no selected item and both neighboring
    endpoints were already selected by the upstream mux path.
    """

    if max_gap_frames <= 0 or len(selected) < 2:
        return selected
    out = dict(selected)
    frames = sorted(selected)
    for left, right in zip(frames, frames[1:]):
        gap_missing = int(right) - int(left) - 1
        if gap_missing <= 0 or gap_missing > int(max_gap_frames):
            continue
        prev_payload: CandidatePayload = selected[left].payload
        next_payload: CandidatePayload = selected[right].payload
        if prev_payload.mux_score < score_floor or next_payload.mux_score < score_floor:
            continue
        for frame in range(int(left) + 1, int(right)):
            payload = interpolate_payload(prev_payload, next_payload, frame, score_floor=score_floor)
            out[frame] = candidate_from_payload(payload)
    return out


def predict_next_bbox(prev2: CandidatePayload, prev1: CandidatePayload, frame: int) -> BBox:
    gap = max(1, int(prev1.frame) - int(prev2.frame))
    dt = max(1, int(frame) - int(prev1.frame))
    c2x, c2y = bbox_center(prev2.bbox)
    c1x, c1y = bbox_center(prev1.bbox)
    vx = (c1x - c2x) / float(gap)
    vy = (c1y - c2y) / float(gap)
    px = c1x + vx * dt
    py = c1y + vy * dt
    w = max(1.0, float(prev1.bbox[2]))
    h = max(1.0, float(prev1.bbox[3]))
    return px - 0.5 * w, py - 0.5 * h, w, h


def recover_target_local_gaps(
    selected: dict[int, SequenceItem],
    tubes_by_frame: dict[int, list[dict[str, str]]],
    *,
    min_emit_score: float,
    max_recovery_frames: int,
    top_k: int,
    max_pred_error: float,
    last_anchor_px: float,
    anchor_mux_sources: set[str] | None,
    raw_min: float,
    replace_existing_error: float = 0.0,
    replace_improvement_px: float = 4.0,
    replace_min_side: float = 0.0,
    replace_top_k: int = 0,
    replace_raw_min: float | None = None,
) -> dict[int, SequenceItem]:
    """Causal target-local recovery using recent emitted track motion.

    This only runs after two emitted boxes exist and only for a bounded number
    of frames after the last emitted target. It is for terrain-backed candidates
    whose visual class logits are overly clutter-like, not for new acquisition.
    """

    if max_recovery_frames <= 0:
        return selected
    out = dict(selected)
    emitted_history: list[CandidatePayload] = []
    missed_since_emit = 0
    all_frames = sorted(set(tubes_by_frame) | set(out))
    for frame in all_frames:
        current = out.get(frame)
        if current is not None and current.score >= min_emit_score:
            frame_gap = (
                int(frame) - int(emitted_history[-1].frame)
                if emitted_history
                else max_recovery_frames + 1
            )
            if (
                replace_existing_error > 0.0
                and len(emitted_history) >= 2
                and frame_gap <= max_recovery_frames
            ):
                pred = predict_next_bbox(emitted_history[-2], emitted_history[-1], frame)
                current_err = center_distance(pred, current.payload.bbox)
                if current_err >= replace_existing_error:
                    best_payload = target_local_candidate_near_prediction(
                        frame,
                        tubes_by_frame.get(frame, [])[: max(1, replace_top_k or top_k)],
                        pred,
                        emitted_history[-1].bbox,
                        min_emit_score=min_emit_score,
                        max_pred_error=max_pred_error,
                        last_anchor_px=0.0,
                        current=current,
                        anchor_mux_sources=anchor_mux_sources,
                        raw_min=raw_min if replace_raw_min is None else replace_raw_min,
                        min_side=replace_min_side,
                    )
                    if best_payload is not None:
                        best_err = center_distance(pred, best_payload.bbox)
                        if best_err + replace_improvement_px < current_err:
                            best_payload = replace(
                                best_payload,
                                reason=(
                                    f"replace_current_error_{current_err:.3f}_"
                                    f"predicted_error_{best_err:.3f}"
                                ),
                            )
                            item = candidate_from_payload(best_payload)
                            out[frame] = item
                            emitted_history.append(item.payload)
                            emitted_history = emitted_history[-4:]
                            missed_since_emit = 0
                            continue
            emitted_history.append(current.payload)
            emitted_history = emitted_history[-4:]
            missed_since_emit = 0
            continue
        if len(emitted_history) < 2:
            missed_since_emit += 1
            continue
        missed_since_emit += 1
        if missed_since_emit > max_recovery_frames:
            continue
        if int(frame) - int(emitted_history[-1].frame) > max_recovery_frames:
            continue
        pred = predict_next_bbox(emitted_history[-2], emitted_history[-1], frame)
        last_box = emitted_history[-1].bbox
        best_payload = target_local_candidate_near_prediction(
            frame,
            tubes_by_frame.get(frame, [])[: max(1, top_k)],
            pred,
            last_box,
            min_emit_score=min_emit_score,
            max_pred_error=max_pred_error,
            last_anchor_px=last_anchor_px,
            current=current,
            anchor_mux_sources=anchor_mux_sources,
            raw_min=raw_min,
        )
        if best_payload is not None:
            item = candidate_from_payload(best_payload)
            out[frame] = item
            emitted_history.append(best_payload)
            emitted_history = emitted_history[-4:]
            missed_since_emit = 0
    return out


def target_local_candidate_near_prediction(
    frame: int,
    rows: list[dict[str, str]],
    pred: BBox,
    last_box: BBox,
    *,
    min_emit_score: float,
    max_pred_error: float,
    last_anchor_px: float,
    current: SequenceItem | None,
    anchor_mux_sources: set[str] | None,
    raw_min: float,
    min_side: float = 0.0,
) -> CandidatePayload | None:
    best_payload: CandidatePayload | None = None
    best_score = -1.0e18
    for row in rows:
        raw = fnum(row.get("score"), 0.0) or 0.0
        if raw < raw_min:
            continue
        bbox = bbox_from_row(row)
        if bbox is None:
            continue
        if min_side > 0.0 and (bbox[2] < min_side or bbox[3] < min_side):
            continue
        pred_err = center_distance(pred, bbox)
        anchor_err = center_distance(last_box, bbox)
        if last_anchor_px > 0.0:
            anchor_source_ok = (
                current is not None
                and (
                    anchor_mux_sources is None
                    or current.payload.mux_source in anchor_mux_sources
                )
            )
            err = min(pred_err, anchor_err)
            accepted_by_anchor = anchor_source_ok and anchor_err <= last_anchor_px
        else:
            err = pred_err
            accepted_by_anchor = False
        if pred_err > max_pred_error and not accepted_by_anchor:
            continue
        rank = fnum(row.get("rank"), 99.0) or 99.0
        score = (
            min_emit_score
            + 0.55
            + 0.25 * clipped(raw / 30.0, 0.0, 1.5)
            - 0.55 * (err / max(1.0, max_pred_error)) ** 2
            - (0.18 if pred_err > max_pred_error else 0.0)
            - 0.06 * math.log1p(max(0.0, rank))
        )
        if score <= best_score:
            continue
        best_score = score
        best_payload = CandidatePayload(
            clip=str(row.get("clip", "")),
            frame=int(frame),
            bbox=bbox,
            mux_source="target_local_recovery",
            mux_score=score,
            rank=str(row.get("rank", "")),
            learned_score=str(row.get("crop_t_prob", "")),
            verified_score=str(row.get("verified_score", "")),
            source=str(row.get("cand_source", "")),
            track_id=str(row.get("track_id", "")),
            reason=(
                f"anchor_error_{anchor_err:.3f}_predicted_error_{pred_err:.3f}"
                if pred_err > max_pred_error
                else f"predicted_error_{pred_err:.3f}"
            ),
            source_path="cs_top_tubes",
        )
    return best_payload


def recenter_emitted_boxes(
    selected: dict[int, SequenceItem],
    tubes_by_frame: dict[int, list[dict[str, str]]],
    *,
    min_emit_score: float,
    top_k: int,
    max_delta_px: float,
    raw_min: float,
    min_area_ratio: float,
    max_area_ratio: float,
    t_margin_weight: float = 0.0,
    clutter_weight: float = 0.0,
    tiny_area_px: float = 0.0,
    tiny_penalty: float = 0.0,
    min_score_gain: float = -1.0e9,
    allowed_sources: set[str] | None = None,
    preserve_sources: set[str] | None = None,
    preserve_min_learned: float = 1.1,
) -> dict[int, SequenceItem]:
    """Tighten already-emitted boxes to nearby top-tube candidates.

    This does not create new detections. It is designed for terrain frames
    where a large/offset box covers the target while a stricter candidate is
    present nearby but scored as clutter.
    """

    out = dict(selected)
    for frame, item in list(selected.items()):
        if item.score < min_emit_score:
            continue
        current_payload: CandidatePayload = item.payload
        if allowed_sources is not None and current_payload.mux_source not in allowed_sources:
            continue
        if preserve_sources is not None:
            source = str(current_payload.source).strip().lower()
            learned = fnum(current_payload.learned_score)
            if source in preserve_sources and learned is not None and learned >= float(preserve_min_learned):
                continue
        current_area = bbox_area(current_payload.bbox)
        best_payload: CandidatePayload | None = None
        best_score = -1.0e18
        for row in tubes_by_frame.get(frame, [])[: max(1, top_k)]:
            raw = fnum(row.get("score"), 0.0) or 0.0
            if raw < raw_min:
                continue
            bbox = bbox_from_row(row)
            if bbox is None:
                continue
            delta = center_distance(current_payload.bbox, bbox)
            if delta > max_delta_px:
                continue
            area_ratio = bbox_area(bbox) / max(1.0, current_area)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            rank = fnum(row.get("rank"), 99.0) or 99.0
            t_prob = clip01(fnum(row.get("crop_t_prob")), 0.0)
            clutter_prob = max(
                clip01(fnum(row.get("crop_s_prob")), 0.0),
                clip01(fnum(row.get("crop_e_prob")), 0.0),
                clip01(fnum(row.get("crop_h_prob")), 0.0),
                clip01(fnum(row.get("crop_g_prob")), 0.0),
            )
            pred_class = str(row.get("crop_pred_class", "")).strip().upper()
            class_bonus = 0.20 if pred_class == "T" else 0.0
            area = bbox_area(bbox)
            tiny_cost = float(tiny_penalty) if tiny_area_px > 0.0 and area <= float(tiny_area_px) else 0.0
            score = (
                item.score
                + 0.35 * clipped(raw / 30.0, -0.5, 1.2)
                + 0.30 * t_prob
                + float(t_margin_weight) * (t_prob - clutter_prob)
                + class_bonus
                - float(clutter_weight) * clutter_prob
                - tiny_cost
                - 0.25 * (delta / max(1.0, max_delta_px)) ** 2
                - 0.04 * math.log1p(max(0.0, rank))
                - 0.10 * abs(math.log(max(0.05, area_ratio)))
            )
            if score <= best_score:
                continue
            best_score = score
            best_payload = CandidatePayload(
                clip=str(row.get("clip", current_payload.clip)),
                frame=int(frame),
                bbox=bbox,
                mux_source=f"{current_payload.mux_source}+recenter",
                mux_score=max(item.score, score),
                rank=str(row.get("rank", "")),
                learned_score=str(row.get("crop_t_prob", current_payload.learned_score)),
                verified_score=str(row.get("verified_score", current_payload.verified_score)),
                source=str(row.get("cand_source", current_payload.source)),
                track_id=str(row.get("track_id", current_payload.track_id)),
                reason=f"recenter_delta_{delta:.3f}_from_{current_payload.mux_source}",
                source_path=current_payload.source_path,
            )
        if best_payload is not None:
            if best_score <= item.score + float(min_score_gain):
                continue
            out[frame] = candidate_from_payload(best_payload)
    return out


def select_streaming_sequence(
    layers: list[tuple[int, list[SequenceItem]]],
    *,
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float,
    restart_penalty: float,
    commit_lag: int,
) -> dict[int, SequenceItem]:
    """Commit a short-lag continuity path instead of one full-clip path."""

    live_layers = []
    selected: dict[int, SequenceItem] = {}
    for frame, items in layers:
        append_viterbi_layer(
            live_layers,
            frame,
            items,
            max_jump_px=max_jump_px,
            transition_weight=transition_weight,
            size_jump_weight=size_jump_weight,
            restart_penalty=restart_penalty,
        )
        while len(live_layers) > max(1, commit_lag):
            emit_frame, item, _score, _path_indices = first_layer_selection(live_layers)
            if emit_frame is not None and item is not None:
                selected[int(emit_frame)] = item
            if live_layers:
                live_layers.pop(0)
    while live_layers:
        emit_frame, item, _score, _path_indices = first_layer_selection(live_layers)
        if emit_frame is not None and item is not None:
            selected[int(emit_frame)] = item
        live_layers.pop(0)
    return selected


def run_clip(args: argparse.Namespace, clip: str, weights: dict[str, float], fallbacks: list[tuple[str, str]]) -> list[dict[str, Any]]:
    frame_items: dict[int, list[SequenceItem]] = defaultdict(list)
    cs_trace = ROOT / args.cs_trace_template.format(clip=clip)
    cs_tubes = ROOT / args.cs_tubes_template.format(clip=clip)
    tubes_by_frame = load_tubes_by_frame(cs_tubes)
    add_cs_js1_candidates(clip, cs_trace, cs_tubes, weights, frame_items)
    add_cs_proposal_candidates(
        clip,
        cs_trace,
        cs_tubes,
        weights,
        frame_items,
        top_k=args.include_cs_proposals,
        weak_margin=args.proposal_weak_margin,
    )
    for name, template in fallbacks:
        add_selected_track_candidates(clip, name, ROOT / template.format(clip=clip), weights, frame_items)

    layers = [(frame, dedupe_items(items)) for frame, items in sorted(frame_items.items()) if items]
    selected = select_streaming_sequence(
        layers,
        max_jump_px=args.max_jump_px,
        transition_weight=args.transition_weight,
        size_jump_weight=args.size_jump_weight,
        restart_penalty=args.restart_penalty,
        commit_lag=args.commit_lag,
    )
    if args.bridge_gap_frames > 0:
        emitted = {frame: item for frame, item in selected.items() if item.score >= args.min_emit_score}
        bridged = bridge_selected_gaps(
            emitted,
            max_gap_frames=args.bridge_gap_frames,
            score_floor=args.bridge_score_floor,
        )
        selected = {**selected, **bridged}
    if args.target_local_recovery_frames > 0:
        anchor_mux_sources = {
            value.strip()
            for value in str(args.target_local_recovery_anchor_mux_sources or "").split(",")
            if value.strip()
        } or None
        selected = recover_target_local_gaps(
            selected,
            tubes_by_frame,
            min_emit_score=args.min_emit_score,
            max_recovery_frames=args.target_local_recovery_frames,
            top_k=args.target_local_recovery_top_k,
            max_pred_error=args.target_local_recovery_max_error,
            last_anchor_px=args.target_local_recovery_last_anchor_px,
            anchor_mux_sources=anchor_mux_sources,
            raw_min=args.target_local_recovery_raw_min,
            replace_existing_error=args.target_local_recovery_replace_existing_error,
            replace_improvement_px=args.target_local_recovery_replace_improvement_px,
            replace_min_side=args.target_local_recovery_replace_min_side,
            replace_top_k=args.target_local_recovery_replace_top_k,
            replace_raw_min=args.target_local_recovery_replace_raw_min,
        )
    if args.recenter_emitted_boxes:
        allowed_sources = {
            value.strip()
            for value in str(args.recenter_mux_sources or "").split(",")
            if value.strip()
        } or None
        preserve_sources = {
            value.strip().lower()
            for value in str(args.recenter_preserve_sources or "").split(",")
            if value.strip()
        } or None
        selected = recenter_emitted_boxes(
            selected,
            tubes_by_frame,
            min_emit_score=args.min_emit_score,
            top_k=args.recenter_top_k,
            max_delta_px=args.recenter_max_delta_px,
            raw_min=args.recenter_raw_min,
            min_area_ratio=args.recenter_min_area_ratio,
            max_area_ratio=args.recenter_max_area_ratio,
            t_margin_weight=args.recenter_t_margin_weight,
            clutter_weight=args.recenter_clutter_weight,
            tiny_area_px=args.recenter_tiny_area_px,
            tiny_penalty=args.recenter_tiny_penalty,
            min_score_gain=args.recenter_min_score_gain,
            allowed_sources=allowed_sources,
            preserve_sources=preserve_sources,
            preserve_min_learned=args.recenter_preserve_min_learned,
        )
    return output_rows_for_clip(clip, selected, min_emit_score=args.min_emit_score)


def main() -> None:
    args = parse_args()
    clips = sorted(set(args.clip))
    if not clips:
        raise SystemExit("--clip is required at least once")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.source_weight)
    fallbacks = parse_name_templates(args.fallback)
    all_rows: list[dict[str, Any]] = []
    for clip in clips:
        rows = run_clip(args, clip, weights, fallbacks)
        write_csv(out_dir / clip / "sequence_selected_tracks.csv", rows)
        all_rows.extend(rows)
    write_csv(out_dir / "all_sequence_selected_tracks.csv", all_rows)
    metadata = [
        {"key": "clips", "value": ",".join(clips)},
        {"key": "fallbacks", "value": ",".join(name for name, _template in fallbacks)},
        {"key": "weights", "value": repr(weights)},
        {"key": "include_cs_proposals", "value": args.include_cs_proposals},
        {"key": "proposal_weak_margin", "value": args.proposal_weak_margin},
        {"key": "max_jump_px", "value": args.max_jump_px},
        {"key": "transition_weight", "value": args.transition_weight},
        {"key": "size_jump_weight", "value": args.size_jump_weight},
        {"key": "restart_penalty", "value": args.restart_penalty},
        {"key": "min_emit_score", "value": args.min_emit_score},
        {"key": "commit_lag", "value": args.commit_lag},
        {"key": "bridge_gap_frames", "value": args.bridge_gap_frames},
        {"key": "bridge_score_floor", "value": args.bridge_score_floor},
        {"key": "target_local_recovery_frames", "value": args.target_local_recovery_frames},
        {"key": "target_local_recovery_top_k", "value": args.target_local_recovery_top_k},
        {"key": "target_local_recovery_max_error", "value": args.target_local_recovery_max_error},
        {"key": "target_local_recovery_last_anchor_px", "value": args.target_local_recovery_last_anchor_px},
        {
            "key": "target_local_recovery_anchor_mux_sources",
            "value": args.target_local_recovery_anchor_mux_sources,
        },
        {"key": "target_local_recovery_raw_min", "value": args.target_local_recovery_raw_min},
        {
            "key": "target_local_recovery_replace_existing_error",
            "value": args.target_local_recovery_replace_existing_error,
        },
        {
            "key": "target_local_recovery_replace_improvement_px",
            "value": args.target_local_recovery_replace_improvement_px,
        },
        {
            "key": "target_local_recovery_replace_min_side",
            "value": args.target_local_recovery_replace_min_side,
        },
        {
            "key": "target_local_recovery_replace_top_k",
            "value": args.target_local_recovery_replace_top_k,
        },
        {
            "key": "target_local_recovery_replace_raw_min",
            "value": args.target_local_recovery_replace_raw_min,
        },
        {"key": "recenter_emitted_boxes", "value": int(args.recenter_emitted_boxes)},
        {"key": "recenter_top_k", "value": args.recenter_top_k},
        {"key": "recenter_max_delta_px", "value": args.recenter_max_delta_px},
        {"key": "recenter_raw_min", "value": args.recenter_raw_min},
        {"key": "recenter_min_area_ratio", "value": args.recenter_min_area_ratio},
        {"key": "recenter_max_area_ratio", "value": args.recenter_max_area_ratio},
        {"key": "recenter_t_margin_weight", "value": args.recenter_t_margin_weight},
        {"key": "recenter_clutter_weight", "value": args.recenter_clutter_weight},
        {"key": "recenter_tiny_area_px", "value": args.recenter_tiny_area_px},
        {"key": "recenter_tiny_penalty", "value": args.recenter_tiny_penalty},
        {"key": "recenter_min_score_gain", "value": args.recenter_min_score_gain},
        {"key": "recenter_mux_sources", "value": args.recenter_mux_sources},
        {"key": "recenter_preserve_sources", "value": args.recenter_preserve_sources},
        {"key": "recenter_preserve_min_learned", "value": args.recenter_preserve_min_learned},
    ]
    write_csv(out_dir / "metadata.csv", metadata)
    print({"out_dir": str(out_dir), "rows": len(all_rows), "clips": len(clips)})


if __name__ == "__main__":
    main()
