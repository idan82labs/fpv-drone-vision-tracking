#!/usr/bin/env python3
"""Shared selector primitives for offline replay and live/deferred tracking.

This module intentionally has no OpenCV, numpy, sklearn, or detector imports.
It owns the small continuity Viterbi primitive that used to be duplicated in
the Mac detector and Raspberry Pi replay path. Higher-level selectors still
own their domain-specific scoring, hysteresis, telemetry, and output formats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SequenceItem:
    """One candidate in a frame layer for continuity selection."""

    frame: int
    bbox: BBox
    score: float
    payload: Any = None


@dataclass
class ViterbiLayer:
    """One scored frame layer in a continuity Viterbi graph."""

    frame: int
    items: list[SequenceItem]
    path_scores: list[float]
    backptrs: list[int | None]


def fnum(value: Any, default: float | None = None) -> float | None:
    """Parse a finite float from CSV/runtime telemetry values."""

    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def fint(value: Any, default: int = 0) -> int:
    out = fnum(value)
    return default if out is None else int(round(out))


def trace_router_bucket(row: dict[str, Any]) -> str:
    """Normalize router labels into the buckets used by rescue policies."""

    raw = str(row.get("router_bucket", "") or "unknown").strip().lower()
    if raw.startswith("surface"):
        return "surface"
    if raw.startswith("line") or raw.startswith("attached"):
        return "line"
    if raw.startswith("boundary") or raw.startswith("skyline"):
        return "boundary"
    if raw.startswith("clean"):
        return "clean"
    return raw or "unknown"


def surface_gate_low_confidence(
    trace: dict[str, Any] | None,
    *,
    gate_states: set[str],
    gate_routers: set[str],
    gate_rank_min: int,
    gate_margin_max: float,
    gate_raw_score_max: float,
) -> tuple[bool, str]:
    """Return whether the base selector trace is weak enough for rescue.

    This is intentionally pure so offline replay and runtime telemetry can use
    one policy instead of drifting through copy-pasted gate logic.
    """

    if trace is None:
        return False, "no_trace"
    router = trace_router_bucket(trace)
    if router not in gate_routers:
        return False, f"router_{router}"

    state = str(trace.get("state", "") or "A")
    selected = fint(trace.get("selected"), 0)
    rank = fint(trace.get("rank"), 999999)
    margin = fnum(trace.get("target_margin"), None)
    raw_score = fnum(trace.get("raw_score"), None)

    reasons: list[str] = []
    if state in gate_states:
        reasons.append(f"state_{state}")
    if not selected:
        reasons.append("no_selected")
    if rank >= gate_rank_min:
        reasons.append(f"rank_ge_{gate_rank_min}")
    if margin is not None and margin <= gate_margin_max:
        reasons.append(f"margin_le_{gate_margin_max:g}")
    if raw_score is not None and raw_score <= gate_raw_score_max:
        reasons.append(f"raw_le_{gate_raw_score_max:g}")
    if not reasons:
        return False, "confident_base"
    return True, "+".join(reasons)


def max_row_feature(rows: list[dict[str, Any]], names: tuple[str, ...], default: float = 0.0) -> float:
    best = default
    for row in rows:
        for name in names:
            value = fnum(row.get(name), None)
            if value is not None:
                best = max(best, value)
    return best


def surface_rescue_risk(
    trace: dict[str, Any] | None,
    base_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
) -> tuple[float, str]:
    """Cheap local risk score for when a surface rescue branch is justified."""

    router = trace_router_bucket(trace or {})
    state = str((trace or {}).get("state", "") or "A")
    selected = fint((trace or {}).get("selected"), 0)
    rank = fint((trace or {}).get("rank"), 999999)
    margin = fnum((trace or {}).get("target_margin"), None)

    score = {
        "surface": 0.45,
        "line": 0.75,
        "boundary": 0.65,
        "unknown": 0.35,
        "clean": -0.85,
    }.get(router, 0.20)
    reasons = [f"router_{router}"]

    if state in {"S", "E"}:
        score += 0.65
        reasons.append(f"lock_{state}")
    elif state in {"A", "P", "C"}:
        score += 0.45
        reasons.append(f"state_{state}")

    if not selected:
        score += 0.50
        reasons.append("no_selected")
    elif rank >= 20:
        score += 0.35
        reasons.append("rank_ge_20")
    elif rank >= 10:
        score += 0.20
        reasons.append("rank_ge_10")

    if margin is not None:
        if margin <= 0.30:
            score += 0.40
            reasons.append("low_margin")
        elif margin <= 0.80:
            score += 0.25
            reasons.append("weak_margin")
        elif selected and rank <= 3 and margin >= 1.50:
            score -= 0.75
            reasons.append("easy_selected_track")

    rows = base_rows[:8] + surface_rows[:8]
    line_context = max_row_feature(rows, ("cand_line_context", "tube_mean_line_context"))
    attached = max_row_feature(rows, ("cand_attached_support", "tube_mean_attached_support"))
    texture = max_row_feature(rows, ("cand_texture", "tube_mean_texture"))
    surface_rate = max_row_feature(rows, ("tube_router_surface_backed_rate",))
    boundary_rate = max_row_feature(rows, ("tube_router_boundary_rate",))
    line_rate = max_row_feature(rows, ("tube_router_line_attached_rate",))

    if line_context >= 0.50 or line_rate >= 0.25:
        score += 0.25
        reasons.append("line_context")
    if attached >= 6.0:
        score += 0.30
        reasons.append("attached_support")
    if texture >= 55.0:
        score += 0.25
        reasons.append("texture")
    if surface_rate >= 0.45:
        score += 0.20
        reasons.append("surface_rate")
    if boundary_rate >= 0.35:
        score += 0.20
        reasons.append("boundary_rate")

    return score, "+".join(reasons)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x, y, w, h = bbox
    return float(x) + 0.5 * float(w), float(y) + 0.5 * float(h)


def center_distance_bbox(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(math.hypot(ax - bx, ay - by))


def max_bbox_span(a: BBox, b: BBox) -> float:
    return max(float(a[2]), float(a[3]), float(b[2]), float(b[3]))


def append_viterbi_layer(
    layers: list[ViterbiLayer],
    frame: int,
    items: Iterable[SequenceItem],
    *,
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float = 0.0,
    restart_penalty: float = 1.0,
) -> ViterbiLayer:
    """Append one frame layer using the shared continuity transition model.

    A candidate links to the best previous candidate within
    ``max_jump_px * frame_gap``. If no previous candidate is reachable, the
    candidate starts a new path with ``restart_penalty`` subtracted. This
    preserves the historical behavior where a stronger late birth does not
    backfill earlier frames.
    """

    current_items = list(items)
    prev = layers[-1] if layers else None
    path_scores: list[float] = []
    backptrs: list[int | None] = []
    if prev is not None and prev.items and prev.path_scores:
        gap = max(1, int(frame) - int(prev.frame))
        for item in current_items:
            best_score = -1.0e18
            best_idx: int | None = None
            for pi, prev_item in enumerate(prev.items):
                size_allowance = float(size_jump_weight) * max_bbox_span(item.bbox, prev_item.bbox)
                allowed = max(1e-6, float(max_jump_px) * float(gap) + size_allowance)
                jump = center_distance_bbox(item.bbox, prev_item.bbox)
                if jump > allowed:
                    continue
                cost = float(transition_weight) * (jump / allowed) ** 2
                score = float(prev.path_scores[pi]) + float(item.score) - cost
                if score > best_score:
                    best_score = score
                    best_idx = pi
            if best_idx is None:
                best_score = float(item.score) - float(restart_penalty)
            path_scores.append(best_score)
            backptrs.append(best_idx)
    else:
        path_scores = [float(item.score) for item in current_items]
        backptrs = [None for _item in current_items]

    layer = ViterbiLayer(frame=int(frame), items=current_items, path_scores=path_scores, backptrs=backptrs)
    layers.append(layer)
    return layer


def best_path_indices(
    layers: list[ViterbiLayer],
    *,
    backfill_unreachable: bool = False,
) -> tuple[list[tuple[int, int]], float | None]:
    """Return the best backtracked path as ``(layer_index, item_index)`` pairs."""

    if not layers or not layers[-1].path_scores:
        return [], None
    last_scores = layers[-1].path_scores
    idx: int | None = max(range(len(last_scores)), key=lambda i: float(last_scores[i]))
    best_score = float(last_scores[idx])
    selected_pairs: list[tuple[int, int]] = []
    for li in range(len(layers) - 1, -1, -1):
        if idx is None:
            if not backfill_unreachable or not layers[li].items:
                break
            idx = max(range(len(layers[li].items)), key=lambda i: float(layers[li].items[i].score))
        selected_pairs.append((li, idx))
        idx = layers[li].backptrs[idx]
    selected_pairs.reverse()
    return selected_pairs, best_score


def first_layer_selection(
    layers: list[ViterbiLayer],
) -> tuple[int | None, SequenceItem | None, float | None, list[int]]:
    """Return the committed oldest-layer item if the best path reaches it.

    This is the delayed live-selector behavior: if the best path is a late
    restart, the oldest frame emits no box instead of inheriting the late birth.
    The returned score is the oldest item's own score, not the cumulative path
    score, matching the existing threshold/hysteresis semantics.
    """

    if not layers:
        return None, None, None, []
    first_frame = layers[0].frame
    pairs, _best_path_score = best_path_indices(layers)
    if len(pairs) != len(layers) or not pairs or pairs[0][0] != 0:
        return first_frame, None, None, []
    path_indices = [idx for _li, idx in pairs]
    first_idx = path_indices[0]
    if first_idx < 0 or first_idx >= len(layers[0].items):
        return first_frame, None, None, []
    item = layers[0].items[first_idx]
    return first_frame, item, float(item.score), path_indices


def commit_path_prefix(layers: list[ViterbiLayer], path_indices: list[int]) -> None:
    """Prune retained layers to the already selected path prefix in-place."""

    if not path_indices:
        return
    for li, keep_idx in enumerate(path_indices[: len(layers)]):
        layer = layers[li]
        if keep_idx < 0 or keep_idx >= len(layer.items):
            return
        layer.items = [layer.items[keep_idx]]
        layer.path_scores = [layer.path_scores[keep_idx]]
        layer.backptrs = [None if li == 0 else 0]


class StreamingViterbiSelector:
    """Small reusable streaming continuity selector."""

    def __init__(self, *, max_jump_px: float, transition_weight: float, restart_penalty: float = 1.0):
        self.max_jump_px = float(max_jump_px)
        self.transition_weight = float(transition_weight)
        self.restart_penalty = float(restart_penalty)
        self.layers: list[ViterbiLayer] = []

    def add_layer(self, frame: int, items: Iterable[SequenceItem]) -> None:
        append_viterbi_layer(
            self.layers,
            frame,
            items,
            max_jump_px=self.max_jump_px,
            transition_weight=self.transition_weight,
            restart_penalty=self.restart_penalty,
        )

    def ready(self, window: int) -> bool:
        return len(self.layers) > max(1, int(window))

    def first_selection(self) -> tuple[int | None, SequenceItem | None, float | None, list[int]]:
        return first_layer_selection(self.layers)

    def pop_first(self) -> None:
        if self.layers:
            self.layers.pop(0)

    def flush(self) -> list[tuple[int, SequenceItem | None, float | None, list[int]]]:
        out: list[tuple[int, SequenceItem | None, float | None, list[int]]] = []
        while self.layers:
            frame, item, score, path_indices = self.first_selection()
            if frame is None:
                break
            out.append((frame, item, score, path_indices))
            self.pop_first()
        return out

    def commit_prefix(self, path_indices: list[int]) -> None:
        commit_path_prefix(self.layers, path_indices)


def select_viterbi_sequence(
    frame_items: Iterable[tuple[int, Iterable[SequenceItem]]],
    *,
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float = 0.0,
    restart_penalty: float = 1.0,
    backfill_unreachable: bool = False,
) -> dict[int, SequenceItem]:
    """Select the best continuity path across a bounded set of frame layers."""

    layers: list[ViterbiLayer] = []
    for frame, items in frame_items:
        append_viterbi_layer(
            layers,
            frame,
            items,
            max_jump_px=max_jump_px,
            transition_weight=transition_weight,
            size_jump_weight=size_jump_weight,
            restart_penalty=restart_penalty,
        )
    pairs, _best_score = best_path_indices(layers, backfill_unreachable=backfill_unreachable)
    return {layers[li].frame: layers[li].items[idx] for li, idx in pairs}
