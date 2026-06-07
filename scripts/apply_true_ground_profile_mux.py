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
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
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


@dataclass
class IdentitySeedPolicy:
    model: Any
    threshold: float
    feature_columns: list[str] = field(default_factory=list)
    model_path: str = ""
    policy_path: str = ""

    def score_payload(
        self,
        payload: CandidatePayload,
        *,
        row_source: str,
        source_reappearance_count_5: int,
    ) -> float:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("pandas is required for identity seed arbiter scoring") from exc
        row = identity_seed_feature_row(
            payload,
            row_source=row_source,
            source_reappearance_count_5=source_reappearance_count_5,
        )
        for col in self.feature_columns:
            row.setdefault(col, math.nan)
        df = pd.DataFrame([row])
        proba = self.model.predict_proba(df)
        classes = [str(c) for c in self.model.classes_]
        if "1" not in classes:
            return 0.0
        return float(proba[0, classes.index("1")])


@dataclass
class ReacquisitionPolicy:
    model: Any
    threshold: float
    feature_columns: list[str] = field(default_factory=list)
    model_path: str = ""
    policy_path: str = ""

    def score_payload(
        self,
        payload: CandidatePayload,
        *,
        current: CandidatePayload | None,
        candidate_slot: int,
        seed_score: float | None = None,
    ) -> float:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("pandas is required for reacquisition arbiter scoring") from exc
        row = reacquisition_feature_row(
            payload,
            current=current,
            candidate_slot=candidate_slot,
            seed_score=seed_score,
        )
        for col in self.feature_columns:
            row.setdefault(col, math.nan)
        df = pd.DataFrame([row])
        proba = self.model.predict_proba(df)
        classes = [str(c) for c in self.model.classes_]
        if "1" not in classes:
            return 0.0
        return float(proba[0, classes.index("1")])


@dataclass
class CurrentTrustPolicy:
    model: Any
    threshold: float
    feature_columns: list[str] = field(default_factory=list)
    model_path: str = ""
    policy_path: str = ""

    def score_payload(self, payload: CandidatePayload) -> float:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("pandas is required for current trust scoring") from exc
        row = current_trust_feature_row(payload)
        for col in self.feature_columns:
            row.setdefault(col, math.nan)
        df = pd.DataFrame([row])
        proba = self.model.predict_proba(df)
        classes = [str(c) for c in self.model.classes_]
        if "1" not in classes:
            return 0.0
        return float(proba[0, classes.index("1")])


@dataclass
class SourcePromoterPolicy:
    model: Any
    threshold: float
    feature_columns: list[str] = field(default_factory=list)
    numeric_feature_columns: list[str] = field(default_factory=list)
    model_path: str = ""
    policy_path: str = ""

    def score_rows(self, rows: list[dict[str, str]]) -> list[float]:
        if not rows:
            return []
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("pandas is required for source promoter scoring") from exc
        normalized: list[dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            for col in self.feature_columns:
                out.setdefault(col, math.nan)
            normalized.append(out)
        df = pd.DataFrame(normalized)
        for col in self.numeric_feature_columns:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        proba = self.model.predict_proba(df)
        classes = [str(c) for c in self.model.classes_]
        if "1" not in classes:
            return [0.0 for _ in rows]
        idx = classes.index("1")
        return [float(v) for v in proba[:, idx]]


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
        "--target_local_recovery_materialize_prediction",
        action="store_true",
        help=(
            "When no existing top-tube candidate is close to the recent-track prediction, "
            "materialize the predicted box itself as a bounded target-local candidate. "
            "Disabled by default."
        ),
    )
    p.add_argument(
        "--target_local_recovery_seed_arbiter",
        default="",
        help=(
            "Optional ID54 seed-arbiter policy JSON or joblib model. When set, raw "
            "path-prediction materialization requires a trusted identity streak."
        ),
    )
    p.add_argument(
        "--target_local_recovery_seed_threshold",
        type=float,
        default=None,
        help="Optional override threshold for --target_local_recovery_seed_arbiter.",
    )
    p.add_argument(
        "--target_local_recovery_seed_streak",
        type=int,
        default=2,
        help="Consecutive trusted non-prediction identities required before materialization.",
    )
    p.add_argument(
        "--target_local_recovery_seed_trace",
        action="store_true",
        help="Write identity_seed_trace.csv when seed arbiter scoring is active.",
    )
    p.add_argument(
        "--target_reacquisition_arbiter",
        default="",
        help=(
            "Optional ID56 switch/reacquisition policy JSON or joblib model. "
            "Scores bounded same-frame alternatives when current output is absent, low-score, or untrusted."
        ),
    )
    p.add_argument(
        "--target_reacquisition_threshold",
        type=float,
        default=None,
        help="Optional override threshold for --target_reacquisition_arbiter.",
    )
    p.add_argument("--target_reacquisition_top_k", type=int, default=5)
    p.add_argument("--target_reacquisition_margin", type=float, default=0.0)
    p.add_argument(
        "--target_reacquisition_trace",
        action="store_true",
        help="Write target_reacquisition_trace.csv when reacquisition scoring is active.",
    )
    p.add_argument(
        "--target_current_trust_model",
        default="",
        help="Optional ID57 current trust policy JSON or joblib model.",
    )
    p.add_argument(
        "--target_current_release_all",
        action="store_true",
        help=(
            "Diagnostic-only: do not protect any current track during target reacquisition. "
            "Useful when current-trust labels contain only wrong current tracks."
        ),
    )
    p.add_argument(
        "--target_current_trust_threshold",
        type=float,
        default=None,
        help="Optional override threshold for --target_current_trust_model.",
    )
    p.add_argument(
        "--target_reacquisition_source_bridge",
        action="store_true",
        help="Expose a bounded top-tube source bridge to the reacquisition scorer.",
    )
    p.add_argument("--target_reacquisition_source_bridge_top_k", type=int, default=5)
    p.add_argument("--target_reacquisition_source_bridge_min_seed", type=float, default=0.0)
    p.add_argument(
        "--target_reacquisition_external_bridge_csv",
        default="",
        help=(
            "Optional diagnostic-only candidate CSV used as an external source bridge. "
            "This is for source-parity replay, not a production default."
        ),
    )
    p.add_argument("--target_reacquisition_external_bridge_top_k", type=int, default=5)
    p.add_argument(
        "--target_source_promoter_model",
        default="",
        help=(
            "Optional ID58 source-promoter policy JSON/joblib. Materializes bounded candidates "
            "from --target_source_promoter_source_table behind explicit flags only."
        ),
    )
    p.add_argument(
        "--target_source_promoter_threshold",
        type=float,
        default=None,
        help="Optional override threshold for --target_source_promoter_model.",
    )
    p.add_argument(
        "--target_source_promoter_source_table",
        default="",
        help="Pi-computable candidate table to score and materialize with the source promoter.",
    )
    p.add_argument("--target_source_promoter_top_k", type=int, default=5)
    p.add_argument(
        "--target_source_promoter_trace",
        action="store_true",
        help="Write target_source_promoter_trace.csv when the ID58 materializer is active.",
    )
    p.add_argument(
        "--target_reacquisition_source_bridge_trace",
        action="store_true",
        help="Write target_reacquisition_source_bridge_trace.csv when the bridge is active.",
    )
    p.add_argument(
        "--target_local_recovery_prediction_score_delta",
        type=float,
        default=0.65,
        help="Score added above min_emit_score for materialized path-prediction candidates.",
    )
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
            frame_items.setdefault(frame, []).append(candidate_from_payload(payload))


def add_source_bridge_candidates(
    clip: str,
    tubes_by_frame: dict[int, list[dict[str, str]]],
    frame_items: dict[int, list[SequenceItem]],
    *,
    top_k: int,
    min_seed: float,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    """Expose a bounded top-tube source bank for reacquisition only.

    This does not run a new detector and does not use future frames. It simply
    surfaces a few already-computed top-tube candidates under a distinct source
    so the reacquisition scorer can evaluate them without widening the global
    mux path.
    """

    if top_k <= 0:
        return
    for frame, rows in tubes_by_frame.items():
        added = 0
        for row in rows[: max(1, int(top_k))]:
            bbox = bbox_from_row(row)
            if bbox is None:
                continue
            raw_score = fnum(row.get("score"), 0.0) or 0.0
            rank = fnum(row.get("rank"), 99.0) or 99.0
            t_prob = clip01(fnum(row.get("crop_t_prob")), 0.0)
            seed_like = max(t_prob, clipped(raw_score / 60.0, 0.0, 1.0))
            if seed_like < float(min_seed):
                continue
            payload = CandidatePayload(
                clip=clip,
                frame=int(frame),
                bbox=bbox,
                mux_source="source_bridge",
                mux_score=1.35 + 0.25 * clipped(raw_score / 30.0, -0.5, 1.5) + 0.15 * t_prob - 0.06 * math.log1p(max(0.0, rank)),
                rank=str(row.get("rank", "")),
                learned_score=str(row.get("crop_t_prob", "")),
                verified_score=str(row.get("verified_score", "")),
                source=str(row.get("cand_source", "")),
                track_id=str(row.get("track_id", "")),
                reason=f"bounded_source_bridge_slot_{added + 1}",
                source_path="cs_top_tubes_source_bridge",
            )
            frame_items[int(frame)].append(candidate_from_payload(payload))
            added += 1
            if trace is not None:
                trace.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "rank": payload.rank,
                        "source": payload.source,
                        "track_id": payload.track_id,
                        "seed_like": round(float(seed_like), 6),
                        "mux_score": round(float(payload.mux_score), 6),
                        "action": "materialized_source_bridge",
                    }
                )


def add_external_bridge_candidates(
    clip: str,
    csv_path: Path,
    frame_items: dict[int, list[SequenceItem]],
    *,
    top_k: int,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    if not csv_path.exists() or top_k <= 0:
        return
    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(csv_path):
        row_clip = str(row.get("clip", ""))
        if row_clip and row_clip != clip:
            continue
        frame = fnum(row.get("frame"))
        if frame is None:
            continue
        rows_by_frame[int(frame)].append(row)
    for frame, rows in rows_by_frame.items():
        rows.sort(
            key=lambda r: (
                -clipped(fnum(r.get("hgb_allfit_score"), 0.0), 0.0, 1.0),
                -(fnum(r.get("score"), 0.0) or 0.0),
                fnum(r.get("rank"), 9999.0) or 9999.0,
            )
        )
        for slot, row in enumerate(rows[: max(1, int(top_k))], start=1):
            bbox = bbox_from_row(row)
            if bbox is None:
                continue
            raw_score = fnum(row.get("score"), 0.0) or 0.0
            hgb = clip01(fnum(row.get("hgb_allfit_score")), 0.0)
            rank = fnum(row.get("rank"), 99.0) or 99.0
            payload = CandidatePayload(
                clip=clip,
                frame=frame,
                bbox=bbox,
                mux_source="external_source_bridge",
                mux_score=1.55 + 0.4 * hgb + 0.25 * clipped(raw_score / 60.0, 0.0, 1.5) - 0.05 * math.log1p(max(0.0, rank)),
                rank=str(row.get("rank", "")),
                learned_score=str(hgb),
                verified_score=str(row.get("verified_score", "")),
                source=str(row.get("source") or row.get("source_family") or ""),
                track_id=str(row.get("track_id", "")),
                reason=f"external_source_bridge_slot_{slot}",
                source_path=str(csv_path),
            )
            frame_items.setdefault(frame, []).append(candidate_from_payload(payload))
            if trace is not None:
                trace.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "rank": payload.rank,
                        "source": payload.source,
                        "track_id": payload.track_id,
                        "seed_like": round(float(hgb), 6),
                        "mux_score": round(float(payload.mux_score), 6),
                        "action": "materialized_external_source_bridge",
                    }
                )


def add_target_source_promoter_candidates(
    clip: str,
    csv_path: Path,
    frame_items: dict[int, list[SequenceItem]],
    *,
    policy: SourcePromoterPolicy | None,
    top_k: int,
    threshold: float,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    """Materialize a bounded source-promoter candidate bank from Pi rows."""

    if policy is None or top_k <= 0 or not csv_path.exists():
        return
    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    all_rows = read_csv(csv_path)
    for row in all_rows:
        row_clip = str(row.get("clip", ""))
        if row_clip and row_clip != clip:
            continue
        frame = fnum(row.get("frame"))
        if frame is None:
            continue
        rows_by_frame[int(frame)].append(row)
    for frame, rows in rows_by_frame.items():
        scores = policy.score_rows(rows)
        scored = []
        for row, promoter_score in zip(rows, scores):
            bbox = bbox_from_row(row)
            if bbox is None:
                continue
            scored.append((float(promoter_score), row, bbox))
        scored.sort(
            key=lambda item: (
                -item[0],
                fnum(item[1].get("rank"), 9999.0) or 9999.0,
                -(fnum(item[1].get("score"), 0.0) or 0.0),
            )
        )
        for slot, (promoter_score, row, bbox) in enumerate(scored[: max(1, int(top_k))], start=1):
            if promoter_score < threshold:
                continue
            raw_score = fnum(row.get("score"), 0.0) or 0.0
            verified = fnum(row.get("verified_score"), 0.0) or 0.0
            rank = fnum(row.get("rank"), 99.0) or 99.0
            mux_score = (
                1.5
                + 0.9 * clipped(promoter_score, 0.0, 1.0)
                + 0.18 * clipped(raw_score / 60.0, 0.0, 1.5)
                + 0.06 * clipped(verified / 70.0, 0.0, 1.5)
                - 0.04 * math.log1p(max(0.0, rank))
            )
            payload = CandidatePayload(
                clip=clip,
                frame=frame,
                bbox=bbox,
                mux_source="target_source_promoter",
                mux_score=mux_score,
                rank=str(row.get("rank", "")),
                learned_score=str(round(float(promoter_score), 8)),
                verified_score=str(row.get("verified_score", "")),
                source=str(row.get("source") or row.get("source_family") or ""),
                track_id=str(row.get("track_id", "")),
                reason=f"target_source_promoter_slot_{slot}",
                source_path=str(csv_path),
            )
            frame_items.setdefault(frame, []).append(candidate_from_payload(payload))
            if trace is not None:
                trace.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "candidate_id": row.get("candidate_id", ""),
                        "rank": payload.rank,
                        "source": payload.source,
                        "track_id": payload.track_id,
                        "source_promoter_score": round(float(promoter_score), 8),
                        "threshold": round(float(threshold), 8),
                        "mux_score": round(float(mux_score), 6),
                        "action": "materialized_target_source_promoter",
                    }
                )


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


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_identity_seed_policy(path_text: str, threshold_override: float | None = None) -> IdentitySeedPolicy | None:
    if not path_text:
        return None
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("joblib is required to load --target_local_recovery_seed_arbiter") from exc

    path = resolve_root_path(path_text)
    feature_columns: list[str] = []
    model_path = path
    threshold = 0.5 if threshold_override is None else float(threshold_override)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        model_path = resolve_root_path(str(payload.get("model_path", "")))
        if threshold_override is None:
            threshold = float(payload.get("threshold", threshold))
        feature_path_text = str(payload.get("feature_columns_path", ""))
        if feature_path_text:
            feature_path = resolve_root_path(feature_path_text)
            if feature_path.exists():
                feature_payload = json.loads(feature_path.read_text())
                feature_columns = [str(c) for c in feature_payload.get("columns", [])]
        if payload.get("feature_columns"):
            feature_columns = [str(c) for c in payload.get("feature_columns", [])]
    model = joblib.load(model_path)
    return IdentitySeedPolicy(
        model=model,
        threshold=threshold,
        feature_columns=feature_columns,
        model_path=str(model_path),
        policy_path=str(path),
    )


def load_reacquisition_policy(path_text: str, threshold_override: float | None = None) -> ReacquisitionPolicy | None:
    if not path_text:
        return None
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("joblib is required to load --target_reacquisition_arbiter") from exc

    path = resolve_root_path(path_text)
    feature_columns: list[str] = []
    model_path = path
    threshold = 0.5 if threshold_override is None else float(threshold_override)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        model_text = str(payload.get("model_path") or payload.get("model") or "")
        model_candidate = Path(model_text)
        if model_candidate.is_absolute():
            model_path = model_candidate
        elif (path.parent / model_candidate).exists():
            model_path = path.parent / model_candidate
        else:
            model_path = resolve_root_path(model_text)
        if threshold_override is None:
            threshold = float(payload.get("threshold", threshold))
        feature_path_text = str(payload.get("feature_columns_path", ""))
        if feature_path_text:
            feature_path = resolve_root_path(feature_path_text)
            if feature_path.exists():
                feature_payload = json.loads(feature_path.read_text())
                feature_columns = [str(c) for c in feature_payload.get("columns", [])]
        if payload.get("feature_columns"):
            feature_columns = [str(c) for c in payload.get("feature_columns", [])]
    model = joblib.load(model_path)
    return ReacquisitionPolicy(
        model=model,
        threshold=threshold,
        feature_columns=feature_columns,
        model_path=str(model_path),
        policy_path=str(path),
    )


def load_current_trust_policy(path_text: str, threshold_override: float | None = None) -> CurrentTrustPolicy | None:
    if not path_text:
        return None
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("joblib is required to load --target_current_trust_model") from exc

    path = resolve_root_path(path_text)
    feature_columns: list[str] = []
    model_path = path
    threshold = 0.5 if threshold_override is None else float(threshold_override)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        model_text = str(payload.get("model_path") or payload.get("model") or "")
        model_candidate = Path(model_text)
        if model_candidate.is_absolute():
            model_path = model_candidate
        elif (path.parent / model_candidate).exists():
            model_path = path.parent / model_candidate
        else:
            model_path = resolve_root_path(model_text)
        if threshold_override is None:
            threshold = float(payload.get("threshold", threshold))
        feature_path_text = str(payload.get("feature_columns_path", ""))
        if feature_path_text:
            feature_path = resolve_root_path(feature_path_text)
            if feature_path.exists():
                feature_payload = json.loads(feature_path.read_text())
                feature_columns = [str(c) for c in feature_payload.get("columns", [])]
        if payload.get("feature_columns"):
            feature_columns = [str(c) for c in payload.get("feature_columns", [])]
    model = joblib.load(model_path)
    return CurrentTrustPolicy(
        model=model,
        threshold=threshold,
        feature_columns=feature_columns,
        model_path=str(model_path),
        policy_path=str(path),
    )


def load_source_promoter_policy(path_text: str, threshold_override: float | None = None) -> SourcePromoterPolicy | None:
    if not path_text:
        return None
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("joblib is required to load --target_source_promoter_model") from exc

    path = resolve_root_path(path_text)
    feature_columns: list[str] = []
    model_path = path
    threshold = 0.0 if threshold_override is None else float(threshold_override)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        model_text = str(payload.get("model_path") or payload.get("model") or "")
        model_candidate = Path(model_text)
        if model_candidate.is_absolute():
            model_path = model_candidate
        elif (path.parent / model_candidate).exists():
            model_path = path.parent / model_candidate
        else:
            model_path = resolve_root_path(model_text)
        if threshold_override is None:
            threshold = float(payload.get("threshold", threshold))
        if payload.get("feature_columns"):
            feature_columns = [str(c) for c in payload.get("feature_columns", [])]
        numeric_feature_columns = [str(c) for c in payload.get("numeric_feature_columns", [])]
        feature_path_text = str(payload.get("feature_columns_path", ""))
        if feature_path_text:
            feature_path = resolve_root_path(feature_path_text)
            if feature_path.exists():
                feature_payload = json.loads(feature_path.read_text())
                feature_columns = [str(c) for c in feature_payload.get("columns", [])]
                numeric_feature_columns = [str(c) for c in feature_payload.get("numeric", [])]
    else:
        numeric_feature_columns = []
    model = joblib.load(model_path)
    return SourcePromoterPolicy(
        model=model,
        threshold=threshold,
        feature_columns=feature_columns,
        numeric_feature_columns=numeric_feature_columns,
        model_path=str(model_path),
        policy_path=str(path),
    )


def payload_source_key(payload: CandidatePayload) -> str:
    return f"{payload.mux_source}:{payload.source}:{payload.track_id}"


def identity_seed_feature_row(
    payload: CandidatePayload,
    *,
    row_source: str,
    source_reappearance_count_5: int,
) -> dict[str, Any]:
    rank = fnum(payload.rank, 9999.0) or 9999.0
    learned = fnum(payload.learned_score)
    verified = fnum(payload.verified_score, 0.0) or 0.0
    score = learned if learned is not None and math.isfinite(learned) else float(payload.mux_score)
    x, y, w, h = payload.bbox
    p_t = learned if learned is not None and 0.0 <= learned <= 1.0 else 0.0
    crop_prob_is_replay_safe = payload.mux_source in {
        "cs_js1",
        "cs_proposal",
        "target_local_recovery",
    }
    model_p_t = p_t if crop_prob_is_replay_safe else 0.0
    source_family = payload.source or payload.mux_source or "unknown"
    frame = int(payload.frame)
    background = "unknown"
    if payload.clip.startswith("e271") or payload.clip == "":
        if frame <= 668:
            background = "road"
        elif frame <= 683:
            background = "mixed_ground"
        else:
            background = "terrain"
    row = {
        "row_source": row_source,
        "source_family": source_family,
        "source": payload.source or payload.mux_source or "unknown",
        "mux_source": payload.mux_source or "",
        "rank": rank,
        "score": score,
        "verified_score": verified,
        "pi_selected": 0,
        "box_area": max(0.0, float(w)) * max(0.0, float(h)),
        "box_aspect": float(w) / max(1e-6, float(h)),
        "is_materialized_prediction": 1 if payload.mux_source == "target_local_path_prediction" else 0,
        "best_p_T": model_p_t,
        "best_T_margin": model_p_t - (1.0 - model_p_t),
        "score_rank_ratio": float(score) / math.log1p(max(1.0, float(rank))),
        "verified_rank_ratio": float(verified) / math.log1p(max(1.0, float(rank))),
        "source_reappearance_count_5": int(source_reappearance_count_5),
        "mux_score": float(payload.mux_score),
        "background_bucket": background,
        "split_group": "mux_replay",
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
    }
    for prefix in ["blocked_hgb_mc", "blocked_logistic_mc", "blocked_extratrees_mc", "allfit_hgb_mc", "allfit_logistic_mc", "allfit_extratrees_mc"]:
        row[f"{prefix}_p_T"] = model_p_t
        row[f"{prefix}_p_S"] = 0.0
        row[f"{prefix}_p_E"] = 0.0
        row[f"{prefix}_p_H"] = 0.0
        row[f"{prefix}_p_G"] = 1.0 - model_p_t
        row[f"{prefix}_T_margin"] = model_p_t - (1.0 - model_p_t)
        row[f"{prefix}_score"] = model_p_t
    return row


def reacquisition_feature_row(
    payload: CandidatePayload,
    *,
    current: CandidatePayload | None,
    candidate_slot: int,
    seed_score: float | None = None,
) -> dict[str, Any]:
    rank = fnum(payload.rank, 9999.0) or 9999.0
    learned = fnum(payload.learned_score)
    verified = fnum(payload.verified_score, 0.0) or 0.0
    score = learned if learned is not None and math.isfinite(learned) else float(payload.mux_score)
    current_score = float(current.mux_score) if current is not None else 0.0
    current_dist = center_distance(payload.bbox, current.bbox) if current is not None else 0.0
    frame = int(payload.frame)
    background = "unknown"
    if payload.clip.startswith("e271") or payload.clip == "":
        if frame <= 668:
            background = "road"
        elif frame <= 683:
            background = "mixed_ground"
        else:
            background = "terrain"
    model_p_t = learned if learned is not None and 0.0 <= learned <= 1.0 else 0.0
    seed_value = 0.0 if seed_score is None else float(seed_score)
    return {
        "row_type": "RUNTIME_ALTERNATIVE",
        "row_runtime_source": "mux_runtime",
        "candidate_slot": int(candidate_slot),
        "current_selected_present": int(current is not None),
        "current_mux_source": "" if current is None else current.mux_source,
        "current_mux_score": 0.0 if current is None else float(current.mux_score),
        "rank": rank,
        "score": score,
        "verified_score": verified,
        "mux_score": float(payload.mux_score),
        "seed_score": seed_value,
        "seed_trusted": int(seed_score is not None and seed_value > 0.0),
        "candidate_is_current": int(current is not None and center_distance(payload.bbox, current.bbox) < 1e-6),
        "distance_to_current_selected": current_dist,
        "candidate_score_minus_current": float(payload.mux_score) - current_score,
        "pred_hgb_mc_score": model_p_t,
        "pred_logistic_mc_score": model_p_t,
        "pred_extratrees_mc_score": model_p_t,
        "hgb_allfit_score": model_p_t,
        "logistic_allfit_score": model_p_t,
        "extratrees_allfit_score": model_p_t,
        "seed_logistic_score": seed_value,
        "seed_logistic_threshold": 0.0,
        "seed_hgb_score": seed_value,
        "seed_hgb_threshold": 0.0,
        "seed_extratrees_score": seed_value,
        "seed_extratrees_threshold": 0.0,
        "cross_seed_logistic_score": seed_value,
        "cross_seed_hgb_score": seed_value,
        "cross_seed_extratrees_score": seed_value,
        "source_family": payload.source or payload.mux_source or "unknown",
        "source": payload.source or payload.mux_source or "unknown",
        "mux_source": payload.mux_source or "",
        "candidate_role": payload.reason or "",
        "row_source": "mux_candidate",
        "background_bucket": background,
        "background": background,
    }


def current_trust_feature_row(payload: CandidatePayload) -> dict[str, Any]:
    rank = fnum(payload.rank, 9999.0) or 9999.0
    learned = fnum(payload.learned_score)
    verified = fnum(payload.verified_score, 0.0) or 0.0
    score = learned if learned is not None and math.isfinite(learned) else float(payload.mux_score)
    frame = int(payload.frame)
    background = "unknown"
    if payload.clip.startswith("e271") or payload.clip == "":
        if frame <= 668:
            background = "road"
        elif frame <= 683:
            background = "mixed_ground"
        else:
            background = "terrain"
    model_p_t = learned if learned is not None and 0.0 <= learned <= 1.0 else 0.0
    return {
        "row_type": "CURRENT_TRACK",
        "row_runtime_source": "selected_output",
        "candidate_slot": 0,
        "current_selected_present": 1,
        "current_mux_source": payload.mux_source or "",
        "current_mux_score": float(payload.mux_score),
        "rank": rank,
        "score": score,
        "verified_score": verified,
        "mux_score": float(payload.mux_score),
        "seed_score": 0.0,
        "seed_trusted": 0,
        "candidate_is_current": 1,
        "distance_to_current_selected": 0.0,
        "candidate_score_minus_current": 0.0,
        "pred_hgb_mc_score": model_p_t,
        "pred_logistic_mc_score": model_p_t,
        "pred_extratrees_mc_score": model_p_t,
        "hgb_allfit_score": model_p_t,
        "logistic_allfit_score": model_p_t,
        "extratrees_allfit_score": model_p_t,
        "seed_logistic_score": 0.0,
        "seed_logistic_threshold": 0.0,
        "seed_hgb_score": 0.0,
        "seed_hgb_threshold": 0.0,
        "seed_extratrees_score": 0.0,
        "seed_extratrees_threshold": 0.0,
        "cross_seed_logistic_score": 0.0,
        "cross_seed_hgb_score": 0.0,
        "cross_seed_extratrees_score": 0.0,
        "source_family": payload.source or payload.mux_source or "unknown",
        "source": payload.source or payload.mux_source or "unknown",
        "mux_source": payload.mux_source or "",
        "background_bucket": background,
        "background": background,
    }


def score_identity_seed(
    policy: IdentitySeedPolicy | None,
    payload: CandidatePayload,
    *,
    source_reappearance_count_5: int,
) -> tuple[float | None, bool]:
    if policy is None:
        return None, True
    score = policy.score_payload(
        payload,
        row_source="mux_selected",
        source_reappearance_count_5=source_reappearance_count_5,
    )
    trusted = score >= policy.threshold and payload.mux_source != "target_local_path_prediction"
    return score, trusted


def append_identity_trace(
    trace: list[dict[str, Any]] | None,
    payload: CandidatePayload,
    *,
    score: float | None,
    threshold: float | None,
    trusted: bool,
    streak: int,
    action: str,
    reason: str = "",
) -> None:
    if trace is None:
        return
    trace.append(
        {
            "clip": payload.clip,
            "frame": payload.frame,
            "mux_source": payload.mux_source,
            "source": payload.source,
            "track_id": payload.track_id,
            "rank": payload.rank,
            "mux_score": round(float(payload.mux_score), 6),
            "seed_score": "" if score is None else round(float(score), 6),
            "seed_threshold": "" if threshold is None else round(float(threshold), 6),
            "trusted": int(bool(trusted)),
            "trusted_streak": int(streak),
            "action": action,
            "reason": reason,
        }
    )


def append_reacquisition_trace(
    trace: list[dict[str, Any]] | None,
    *,
    clip: str,
    frame: int,
    current: CandidatePayload | None,
    candidate: CandidatePayload | None,
    switch_score: float | None,
    threshold: float,
    action: str,
    reason: str,
    current_trust_score: float | None = None,
    current_trust_threshold: float | None = None,
) -> None:
    if trace is None:
        return
    trace.append(
        {
            "clip": clip,
            "frame": frame,
            "current_mux_source": "" if current is None else current.mux_source,
            "current_mux_score": "" if current is None else round(float(current.mux_score), 6),
            "current_trust_score": "" if current_trust_score is None else round(float(current_trust_score), 6),
            "current_trust_threshold": "" if current_trust_threshold is None else round(float(current_trust_threshold), 6),
            "candidate_id": "" if candidate is None else payload_source_key(candidate),
            "candidate_source": "" if candidate is None else candidate.source,
            "candidate_mux_source": "" if candidate is None else candidate.mux_source,
            "candidate_mux_score": "" if candidate is None else round(float(candidate.mux_score), 6),
            "switch_score": "" if switch_score is None else round(float(switch_score), 6),
            "switch_threshold": round(float(threshold), 6),
            "action": action,
            "reason": reason,
        }
    )


def score_reacquisition_candidate(
    reacquisition_policy: ReacquisitionPolicy,
    payload: CandidatePayload,
    *,
    current: CandidatePayload | None,
    candidate_slot: int,
    seed_policy: IdentitySeedPolicy | None,
) -> tuple[float, float | None, bool]:
    seed_score, seed_trusted = score_identity_seed(
        seed_policy,
        payload,
        source_reappearance_count_5=1,
    )
    score = reacquisition_policy.score_payload(
        payload,
        current=current,
        candidate_slot=candidate_slot,
        seed_score=seed_score,
    )
    return score, seed_score, seed_trusted


def score_current_trust(
    current_trust_policy: CurrentTrustPolicy | None,
    payload: CandidatePayload,
    *,
    fallback_trusted: bool,
) -> tuple[float | None, bool]:
    if current_trust_policy is None:
        return None, fallback_trusted
    score = current_trust_policy.score_payload(payload)
    return score, score >= current_trust_policy.threshold


def apply_target_reacquisition(
    selected: dict[int, SequenceItem],
    frame_items: dict[int, list[SequenceItem]],
    *,
    min_emit_score: float,
    reacquisition_policy: ReacquisitionPolicy | None,
    seed_policy: IdentitySeedPolicy | None,
    top_k: int,
    margin: float,
    current_trust_policy: CurrentTrustPolicy | None = None,
    force_release_current: bool = False,
    trace: list[dict[str, Any]] | None = None,
) -> dict[int, SequenceItem]:
    if reacquisition_policy is None:
        return selected
    out = dict(selected)
    candidate_frames = sorted(set(frame_items) | set(selected))
    for frame in candidate_frames:
        current_item = out.get(frame)
        current_payload: CandidatePayload | None = None if current_item is None else current_item.payload
        current_seed_score: float | None = None
        current_trust_score: float | None = None
        current_trusted = False
        if current_payload is not None and current_item is not None:
            current_seed_score, current_seed_trusted = score_identity_seed(
                seed_policy,
                current_payload,
                source_reappearance_count_5=1,
            )
            current_trusted = (
                current_item.score >= min_emit_score
                and current_payload.mux_source != "target_local_path_prediction"
                and (seed_policy is None or current_seed_trusted)
            )
            current_trust_score, current_trusted_by_model = score_current_trust(
                current_trust_policy,
                current_payload,
                fallback_trusted=current_trusted,
            )
            current_trusted = current_item.score >= min_emit_score and current_trusted_by_model
            if force_release_current:
                current_trusted = False
        if current_trusted:
            append_reacquisition_trace(
                trace,
                clip=current_payload.clip if current_payload else "",
                frame=frame,
                current=current_payload,
                candidate=None,
                switch_score=None,
                threshold=reacquisition_policy.threshold,
                action="keep_current",
                reason="trusted_current",
                current_trust_score=current_trust_score,
                current_trust_threshold=None if current_trust_policy is None else current_trust_policy.threshold,
            )
            continue

        candidates = []
        for item in dedupe_items(frame_items.get(frame, []))[: max(1, int(top_k))]:
            payload: CandidatePayload = item.payload
            if payload.mux_source == "target_local_path_prediction":
                continue
            if current_payload is not None and center_distance(payload.bbox, current_payload.bbox) < 1e-6:
                continue
            candidates.append(item)
        if not candidates:
            append_reacquisition_trace(
                trace,
                clip=current_payload.clip if current_payload else "",
                frame=frame,
                current=current_payload,
                candidate=None,
                switch_score=None,
                threshold=reacquisition_policy.threshold,
                action="no_safe_candidate",
                reason="no_bounded_alternative",
                current_trust_score=current_trust_score,
                current_trust_threshold=None if current_trust_policy is None else current_trust_policy.threshold,
            )
            continue

        scored: list[tuple[float, SequenceItem, float | None, bool]] = []
        for slot, item in enumerate(candidates, start=1):
            score, seed_score, seed_trusted = score_reacquisition_candidate(
                reacquisition_policy,
                item.payload,
                current=current_payload,
                candidate_slot=slot,
                seed_policy=seed_policy,
            )
            scored.append((score, item, seed_score, seed_trusted))
        best_score, best_item, _seed_score, _seed_trusted = max(scored, key=lambda row: row[0])
        current_score = 0.0
        if not force_release_current and current_item is not None and current_item.score >= min_emit_score:
            current_score = min(1.0, max(0.0, float(current_item.score) / max(1.0, float(min_emit_score) * 2.0)))
        if best_score >= reacquisition_policy.threshold and best_score >= current_score + float(margin):
            payload: CandidatePayload = best_item.payload
            switch_payload = replace(
                payload,
                mux_source=f"{payload.mux_source}+reacquire",
                mux_score=max(float(best_item.score), float(min_emit_score) + 0.1),
                reason=f"target_reacquisition_score_{best_score:.3f}",
            )
            out[frame] = SequenceItem(frame=frame, bbox=switch_payload.bbox, score=switch_payload.mux_score, payload=switch_payload)
            append_reacquisition_trace(
                trace,
                clip=switch_payload.clip,
                frame=frame,
                current=current_payload,
                candidate=payload,
                switch_score=best_score,
                threshold=reacquisition_policy.threshold,
                action="switch_to_candidate",
                reason="score_above_threshold",
                current_trust_score=current_trust_score,
                current_trust_threshold=None if current_trust_policy is None else current_trust_policy.threshold,
            )
        else:
            append_reacquisition_trace(
                trace,
                clip=(current_payload.clip if current_payload else best_item.payload.clip),
                frame=frame,
                current=current_payload,
                candidate=best_item.payload,
                switch_score=best_score,
                threshold=reacquisition_policy.threshold,
                action="no_safe_candidate",
                reason="below_threshold_or_margin",
                current_trust_score=current_trust_score,
                current_trust_threshold=None if current_trust_policy is None else current_trust_policy.threshold,
            )
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
    materialize_prediction: bool = False,
    prediction_score_delta: float = 0.65,
    seed_policy: IdentitySeedPolicy | None = None,
    seed_streak: int = 2,
    identity_trace: list[dict[str, Any]] | None = None,
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
    trusted_history: list[bool] = []
    source_frame_history: dict[str, list[int]] = defaultdict(list)
    missed_since_emit = 0
    all_frames = sorted(set(tubes_by_frame) | set(out))
    for frame in all_frames:
        current = out.get(frame)
        if current is not None and current.score >= min_emit_score:
            src_key = payload_source_key(current.payload)
            prior_source_frames = [f for f in source_frame_history[src_key] if int(frame) - int(f) <= 5]
            seed_score, seed_trusted = score_identity_seed(
                seed_policy,
                current.payload,
                source_reappearance_count_5=len(prior_source_frames),
            )
            streak_count = 1 if seed_trusted else 0
            for prev_trusted in reversed(trusted_history):
                if not (seed_trusted and prev_trusted):
                    break
                streak_count += 1
            append_identity_trace(
                identity_trace,
                current.payload,
                score=seed_score,
                threshold=None if seed_policy is None else seed_policy.threshold,
                trusted=seed_trusted,
                streak=streak_count,
                action="seed_observed",
            )
            source_frame_history[src_key] = prior_source_frames + [int(frame)]
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
                            rec_score, rec_trusted = score_identity_seed(
                                seed_policy,
                                item.payload,
                                source_reappearance_count_5=0,
                            )
                            append_identity_trace(
                                identity_trace,
                                item.payload,
                                score=rec_score,
                                threshold=None if seed_policy is None else seed_policy.threshold,
                                trusted=rec_trusted,
                                streak=count_trusted_streak(trusted_history + [rec_trusted]),
                                action="recovered_candidate_observed",
                            )
                            emitted_history.append(item.payload)
                            trusted_history.append(rec_trusted)
                            emitted_history = emitted_history[-4:]
                            trusted_history = trusted_history[-4:]
                            missed_since_emit = 0
                            continue
                    elif materialize_prediction:
                        if materialization_allowed(trusted_history, seed_streak, seed_policy):
                            pred_payload = target_local_prediction_payload(
                                frame,
                                pred,
                                min_emit_score=min_emit_score,
                                score_delta=prediction_score_delta,
                                reason=f"replace_materialized_prediction_current_error_{current_err:.3f}",
                            )
                            append_identity_trace(
                                identity_trace,
                                pred_payload,
                                score=None,
                                threshold=None if seed_policy is None else seed_policy.threshold,
                                trusted=False,
                                streak=0,
                                action="materialize_allowed",
                                reason=pred_payload.reason,
                            )
                            item = candidate_from_payload(pred_payload)
                            out[frame] = item
                            emitted_history.append(item.payload)
                            trusted_history.append(False)
                            emitted_history = emitted_history[-4:]
                            trusted_history = trusted_history[-4:]
                            missed_since_emit = 0
                            continue
                        append_identity_trace(
                            identity_trace,
                            current.payload,
                            score=seed_score,
                            threshold=None if seed_policy is None else seed_policy.threshold,
                            trusted=seed_trusted,
                            streak=streak_count,
                            action="materialize_blocked_seed_gate",
                            reason=f"replace_current_error_{current_err:.3f}",
                        )
            emitted_history.append(current.payload)
            trusted_history.append(seed_trusted)
            emitted_history = emitted_history[-4:]
            trusted_history = trusted_history[-4:]
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
            rec_score, rec_trusted = score_identity_seed(
                seed_policy,
                item.payload,
                source_reappearance_count_5=0,
            )
            append_identity_trace(
                identity_trace,
                item.payload,
                score=rec_score,
                threshold=None if seed_policy is None else seed_policy.threshold,
                trusted=rec_trusted,
                streak=count_trusted_streak(trusted_history + [rec_trusted]),
                action="recovered_candidate_observed",
            )
            emitted_history.append(best_payload)
            trusted_history.append(rec_trusted)
            emitted_history = emitted_history[-4:]
            trusted_history = trusted_history[-4:]
            missed_since_emit = 0
        elif materialize_prediction:
            if materialization_allowed(trusted_history, seed_streak, seed_policy):
                best_payload = target_local_prediction_payload(
                    frame,
                    pred,
                    min_emit_score=min_emit_score,
                    score_delta=prediction_score_delta,
                    reason="materialized_path_prediction_no_near_candidate",
                )
                append_identity_trace(
                    identity_trace,
                    best_payload,
                    score=None,
                    threshold=None if seed_policy is None else seed_policy.threshold,
                    trusted=False,
                    streak=0,
                    action="materialize_allowed",
                    reason=best_payload.reason,
                )
                item = candidate_from_payload(best_payload)
                out[frame] = item
                emitted_history.append(best_payload)
                trusted_history.append(False)
                emitted_history = emitted_history[-4:]
                trusted_history = trusted_history[-4:]
                missed_since_emit = 0
            else:
                append_identity_trace(
                    identity_trace,
                    emitted_history[-1],
                    score=None,
                    threshold=None if seed_policy is None else seed_policy.threshold,
                    trusted=trusted_history[-1] if trusted_history else False,
                    streak=count_trusted_streak(trusted_history),
                    action="materialize_blocked_seed_gate",
                    reason="no_near_candidate",
                )
    return out


def count_trusted_streak(trusted_history: list[bool]) -> int:
    count = 0
    for trusted in reversed(trusted_history):
        if not trusted:
            break
        count += 1
    return count


def materialization_allowed(
    trusted_history: list[bool],
    seed_streak: int,
    seed_policy: IdentitySeedPolicy | None,
) -> bool:
    if seed_policy is None:
        return True
    streak = max(1, int(seed_streak))
    if len(trusted_history) < streak:
        return False
    return all(trusted_history[-streak:])


def target_local_prediction_payload(
    frame: int,
    pred: BBox,
    *,
    min_emit_score: float,
    score_delta: float,
    reason: str,
) -> CandidatePayload:
    return CandidatePayload(
        clip="",
        frame=int(frame),
        bbox=pred,
        mux_source="target_local_path_prediction",
        mux_score=min_emit_score + score_delta,
        rank="",
        learned_score="",
        verified_score="",
        source="target_local_path_prediction",
        track_id="",
        reason=reason,
        source_path="motion_prediction",
    )


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


def run_clip(
    args: argparse.Namespace,
    clip: str,
    weights: dict[str, float],
    fallbacks: list[tuple[str, str]],
    *,
    seed_policy: IdentitySeedPolicy | None = None,
    reacquisition_policy: ReacquisitionPolicy | None = None,
    current_trust_policy: CurrentTrustPolicy | None = None,
    source_promoter_policy: SourcePromoterPolicy | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_items: dict[int, list[SequenceItem]] = defaultdict(list)
    identity_trace: list[dict[str, Any]] = []
    reacquisition_trace: list[dict[str, Any]] = []
    source_bridge_trace: list[dict[str, Any]] = []
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
            materialize_prediction=args.target_local_recovery_materialize_prediction,
            prediction_score_delta=args.target_local_recovery_prediction_score_delta,
            seed_policy=seed_policy,
            seed_streak=args.target_local_recovery_seed_streak,
            identity_trace=identity_trace if (seed_policy is not None or args.target_local_recovery_seed_trace) else None,
        )
    if args.target_reacquisition_source_bridge:
        add_source_bridge_candidates(
            clip,
            tubes_by_frame,
            frame_items,
            top_k=args.target_reacquisition_source_bridge_top_k,
            min_seed=args.target_reacquisition_source_bridge_min_seed,
            trace=source_bridge_trace if args.target_reacquisition_source_bridge_trace else None,
        )
    if args.target_reacquisition_external_bridge_csv:
        add_external_bridge_candidates(
            clip,
            resolve_root_path(args.target_reacquisition_external_bridge_csv),
            frame_items,
            top_k=args.target_reacquisition_external_bridge_top_k,
            trace=source_bridge_trace if args.target_reacquisition_source_bridge_trace else None,
        )
    if args.target_source_promoter_source_table and source_promoter_policy is not None:
        add_target_source_promoter_candidates(
            clip,
            resolve_root_path(args.target_source_promoter_source_table),
            frame_items,
            policy=source_promoter_policy,
            top_k=args.target_source_promoter_top_k,
            threshold=source_promoter_policy.threshold,
            trace=source_bridge_trace if args.target_source_promoter_trace else None,
        )
    if reacquisition_policy is not None:
        selected = apply_target_reacquisition(
            selected,
            frame_items,
            min_emit_score=args.min_emit_score,
            reacquisition_policy=reacquisition_policy,
            seed_policy=seed_policy,
            top_k=args.target_reacquisition_top_k,
            margin=args.target_reacquisition_margin,
            current_trust_policy=current_trust_policy,
            force_release_current=args.target_current_release_all,
            trace=reacquisition_trace if args.target_reacquisition_trace else None,
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
    return output_rows_for_clip(clip, selected, min_emit_score=args.min_emit_score), identity_trace, reacquisition_trace, source_bridge_trace


def main() -> None:
    args = parse_args()
    clips = sorted(set(args.clip))
    if not clips:
        raise SystemExit("--clip is required at least once")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.source_weight)
    fallbacks = parse_name_templates(args.fallback)
    seed_policy = load_identity_seed_policy(
        args.target_local_recovery_seed_arbiter,
        args.target_local_recovery_seed_threshold,
    )
    reacquisition_policy = load_reacquisition_policy(
        args.target_reacquisition_arbiter,
        args.target_reacquisition_threshold,
    )
    current_trust_policy = load_current_trust_policy(
        args.target_current_trust_model,
        args.target_current_trust_threshold,
    )
    source_promoter_policy = load_source_promoter_policy(
        args.target_source_promoter_model,
        args.target_source_promoter_threshold,
    )
    all_rows: list[dict[str, Any]] = []
    all_identity_trace: list[dict[str, Any]] = []
    all_reacquisition_trace: list[dict[str, Any]] = []
    all_source_bridge_trace: list[dict[str, Any]] = []
    for clip in clips:
        rows, identity_trace, reacquisition_trace, source_bridge_trace = run_clip(
            args,
            clip,
            weights,
            fallbacks,
            seed_policy=seed_policy,
            reacquisition_policy=reacquisition_policy,
            current_trust_policy=current_trust_policy,
            source_promoter_policy=source_promoter_policy,
        )
        write_csv(out_dir / clip / "sequence_selected_tracks.csv", rows)
        if identity_trace:
            write_csv(out_dir / clip / "identity_seed_trace.csv", identity_trace)
        if reacquisition_trace:
            write_csv(out_dir / clip / "target_reacquisition_trace.csv", reacquisition_trace)
        if source_bridge_trace:
            write_csv(out_dir / clip / "target_reacquisition_source_bridge_trace.csv", source_bridge_trace)
            if args.target_source_promoter_trace:
                write_csv(out_dir / clip / "target_source_promoter_trace.csv", source_bridge_trace)
        all_rows.extend(rows)
        all_identity_trace.extend(identity_trace)
        all_reacquisition_trace.extend(reacquisition_trace)
        all_source_bridge_trace.extend(source_bridge_trace)
    write_csv(out_dir / "all_sequence_selected_tracks.csv", all_rows)
    if all_identity_trace:
        write_csv(out_dir / "identity_seed_trace.csv", all_identity_trace)
    if all_reacquisition_trace:
        write_csv(out_dir / "target_reacquisition_trace.csv", all_reacquisition_trace)
    if all_source_bridge_trace:
        write_csv(out_dir / "target_reacquisition_source_bridge_trace.csv", all_source_bridge_trace)
        if args.target_source_promoter_trace:
            write_csv(out_dir / "target_source_promoter_trace.csv", all_source_bridge_trace)
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
        {
            "key": "target_local_recovery_materialize_prediction",
            "value": int(args.target_local_recovery_materialize_prediction),
        },
        {
            "key": "target_local_recovery_prediction_score_delta",
            "value": args.target_local_recovery_prediction_score_delta,
        },
        {
            "key": "target_local_recovery_seed_arbiter",
            "value": args.target_local_recovery_seed_arbiter,
        },
        {
            "key": "target_local_recovery_seed_threshold",
            "value": "" if args.target_local_recovery_seed_threshold is None else args.target_local_recovery_seed_threshold,
        },
        {
            "key": "target_local_recovery_seed_streak",
            "value": args.target_local_recovery_seed_streak,
        },
        {
            "key": "target_local_recovery_seed_policy_model_path",
            "value": "" if seed_policy is None else seed_policy.model_path,
        },
        {
            "key": "target_local_recovery_seed_policy_threshold",
            "value": "" if seed_policy is None else seed_policy.threshold,
        },
        {
            "key": "target_reacquisition_arbiter",
            "value": args.target_reacquisition_arbiter,
        },
        {
            "key": "target_reacquisition_threshold",
            "value": "" if args.target_reacquisition_threshold is None else args.target_reacquisition_threshold,
        },
        {
            "key": "target_reacquisition_policy_model_path",
            "value": "" if reacquisition_policy is None else reacquisition_policy.model_path,
        },
        {
            "key": "target_reacquisition_policy_threshold",
            "value": "" if reacquisition_policy is None else reacquisition_policy.threshold,
        },
        {"key": "target_reacquisition_top_k", "value": args.target_reacquisition_top_k},
        {"key": "target_reacquisition_margin", "value": args.target_reacquisition_margin},
        {"key": "target_reacquisition_trace", "value": int(args.target_reacquisition_trace)},
        {"key": "target_current_trust_model", "value": args.target_current_trust_model},
        {"key": "target_current_release_all", "value": int(args.target_current_release_all)},
        {
            "key": "target_current_trust_threshold",
            "value": "" if args.target_current_trust_threshold is None else args.target_current_trust_threshold,
        },
        {
            "key": "target_current_trust_policy_model_path",
            "value": "" if current_trust_policy is None else current_trust_policy.model_path,
        },
        {
            "key": "target_current_trust_policy_threshold",
            "value": "" if current_trust_policy is None else current_trust_policy.threshold,
        },
        {"key": "target_source_promoter_model", "value": args.target_source_promoter_model},
        {
            "key": "target_source_promoter_threshold",
            "value": "" if args.target_source_promoter_threshold is None else args.target_source_promoter_threshold,
        },
        {
            "key": "target_source_promoter_policy_model_path",
            "value": "" if source_promoter_policy is None else source_promoter_policy.model_path,
        },
        {
            "key": "target_source_promoter_policy_threshold",
            "value": "" if source_promoter_policy is None else source_promoter_policy.threshold,
        },
        {"key": "target_source_promoter_source_table", "value": args.target_source_promoter_source_table},
        {"key": "target_source_promoter_top_k", "value": args.target_source_promoter_top_k},
        {"key": "target_source_promoter_trace", "value": int(args.target_source_promoter_trace)},
        {"key": "target_reacquisition_source_bridge", "value": int(args.target_reacquisition_source_bridge)},
        {"key": "target_reacquisition_source_bridge_top_k", "value": args.target_reacquisition_source_bridge_top_k},
        {"key": "target_reacquisition_source_bridge_min_seed", "value": args.target_reacquisition_source_bridge_min_seed},
        {"key": "target_reacquisition_external_bridge_csv", "value": args.target_reacquisition_external_bridge_csv},
        {"key": "target_reacquisition_external_bridge_top_k", "value": args.target_reacquisition_external_bridge_top_k},
        {"key": "target_reacquisition_source_bridge_trace", "value": int(args.target_reacquisition_source_bridge_trace)},
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
