#!/usr/bin/env python3
"""Evaluate an explicit-state candidate selector.

This is an offline harness for the professor's A/P/T/S/E/C recommendation. It
does not create proposals and does not change runtime behavior. It consumes
exported top-tube rows, including optional CLBA columns, and asks whether an
explicit target/static/attached/null state model improves full-video selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


STATES = ("A", "P", "T", "S", "E", "C")


@dataclass(frozen=True)
class Label:
    visible: bool
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class Candidate:
    frame: int
    rank: int
    bbox: tuple[float, float, float, float]
    target_obs: float
    static_obs: float
    attached_obs: float
    raw_score: float
    row: dict[str, str]
    boundary_obs: float = 0.0
    generic_obs: float = 0.0
    null_obs: float = 0.0
    target_llr: float = 0.0
    static_llr: float = 0.0
    attached_llr: float = 0.0
    boundary_llr: float = 0.0
    generic_llr: float = 0.0
    null_llr: float = 0.0
    target_margin: float = 0.0
    router_bucket: str = "unknown"
    proposal_prior: float = 0.0


@dataclass(frozen=True)
class Hypothesis:
    state: str
    score: float
    bbox: tuple[float, float, float, float] | None = None
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 0
    misses: int = 0
    age: int = 0
    lock_age: int = 0
    quarantine_bbox: tuple[float, float, float, float] | None = None
    selected: Candidate | None = None
    reason: str = ""
    quarantine_kind: str = ""
    router_bucket: str = "unknown"
    target_logodds: float = 0.0
    overrides: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--score_column", default="score")
    p.add_argument("--max_rank", type=int, default=80)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument("--beam_width", type=int, default=96)
    p.add_argument("--state_beam", type=int, default=12)
    p.add_argument("--max_jump_px", default="12,18,24,32,48")
    p.add_argument("--acquire_thresholds", default="0.8,1.0,1.2,1.5,1.8,2.2")
    p.add_argument("--track_thresholds", default="0.2,0.4,0.6,0.8,1.0,1.2")
    p.add_argument("--acquire_hits", default="1,2,3")
    p.add_argument("--max_misses", default="0,1,2")
    p.add_argument("--clutter_margin", default="0.1,0.3,0.6,1.0")
    p.add_argument(
        "--clutter_lock_gap",
        type=float,
        default=0.0,
        help=(
            "Require static/attached clutter evidence to beat target evidence by this margin "
            "before entering S/E quarantine. Set very negative to reproduce the older positive-clutter-only lock."
        ),
    )
    p.add_argument("--quarantine_px", type=float, default=12.0)
    p.add_argument("--static_quarantine_frames", type=int, default=15)
    p.add_argument("--attached_quarantine_frames", type=int, default=30)
    p.add_argument(
        "--quarantine_frames",
        type=int,
        default=0,
        help="Compatibility alias. When >0, overrides both static and attached quarantine TTLs.",
    )
    p.add_argument(
        "--global_quarantine",
        action="store_true",
        help="Apply S/E lock quarantine across the whole beam instead of only the lock path.",
    )
    p.add_argument("--score_weight", type=float, default=0.75)
    p.add_argument("--proposal_clip", type=float, default=2.0)
    p.add_argument(
        "--learned_prior_source",
        choices=("proposal", "raw_score"),
        default="proposal",
        help="Source for the weak old-score prior in learned_logits observation mode.",
    )
    p.add_argument("--learned_target_prior_weight", type=float, default=0.15)
    p.add_argument("--learned_clutter_prior_weight", type=float, default=0.05)
    p.add_argument("--learned_generic_prior_weight", type=float, default=0.02)
    p.add_argument("--learned_prior_clip", type=float, default=2.0)
    p.add_argument(
        "--surface_branch_rank_bonus",
        type=float,
        default=0.0,
        help="Extra target logit for gated surface-branch candidates, decayed by their surface rank.",
    )
    p.add_argument(
        "--surface_branch_rank_decay",
        type=float,
        default=0.35,
        help="Subtract this times log1p(surface rank) from surface_branch_rank_bonus.",
    )
    p.add_argument(
        "--instant_surface_acquire_score",
        type=float,
        default=1.1,
        help=(
            "When <=1, allow A->T immediate acquisition for gated surface-branch candidates "
            "with surface_halo_score at least this value. S/E still release through P."
        ),
    )
    p.add_argument(
        "--observation_mode",
        choices=("js1", "old_score", "oracle", "learned_logits"),
        default="js1",
        help=(
            "Observation override. js1 uses engineered observations; old_score disables clutter terms; "
            "oracle uses labels; learned_logits reads crop_t/s/e/h/g_logit columns."
        ),
    )
    p.add_argument(
        "--proposal_mode",
        choices=("shared", "target_only"),
        default="shared",
        help="Whether old candidate score is a shared non-null prior or target-only objectness.",
    )
    p.add_argument("--clba_weight", type=float, default=0.55)
    p.add_argument("--path_weight", type=float, default=0.25)
    p.add_argument("--static_weight", type=float, default=0.7)
    p.add_argument("--attached_weight", type=float, default=0.7)
    p.add_argument("--rank_weight", type=float, default=0.12)
    p.add_argument("--motion_weight", type=float, default=0.8)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--image_width", type=float, default=320.0)
    p.add_argument("--horizontal_fov_deg", type=float, default=120.0)
    p.add_argument("--vmax_mps", type=float, default=10.0)
    p.add_argument("--amax_mps2", type=float, default=40.0)
    p.add_argument("--registration_sigma_px", type=float, default=1.5)
    p.add_argument("--box_sigma_px", type=float, default=2.0)
    p.add_argument("--motion_prior_weight", type=float, default=0.25)
    p.add_argument("--absurd_jump_px", type=float, default=96.0)
    p.add_argument("--null_priors", default="")
    p.add_argument(
        "--null_calibration_quantile",
        type=float,
        default=-1.0,
        help="When >=0, estimate per-router max-null offsets from invisible frames in this label set.",
    )
    p.add_argument("--min_router_null_samples", type=int, default=3)
    p.add_argument("--null_calibration_margin", type=float, default=0.0)
    p.add_argument(
        "--quarantine_override_margin",
        type=float,
        default=1.0,
        help="Allow a quarantined candidate only when target evidence beats clutter/null by this margin.",
    )
    p.add_argument(
        "--continuity_bonus",
        type=float,
        default=0.0,
        help="Extra target evidence for plausible continuations from T/C states. Offline replay only.",
    )
    p.add_argument(
        "--continuity_max_pred_error_px",
        type=float,
        default=12.0,
        help="Maximum prediction error for applying continuity_bonus.",
    )
    p.add_argument(
        "--continuity_clutter_gap",
        type=float,
        default=0.5,
        help="Require target observation to be within this margin of best clutter observation for continuity rescue.",
    )
    p.add_argument(
        "--continuity_min_raw_score",
        type=float,
        default=0.0,
        help="Minimum raw candidate score for continuity rescue.",
    )
    p.add_argument(
        "--absent_reward",
        type=float,
        default=0.05,
        help="Per-frame score for staying in absent/null state.",
    )
    p.add_argument(
        "--absent_score_cap",
        type=float,
        default=1e9,
        help="Cap accumulated absent-state score so long null runs cannot dominate later births.",
    )
    p.add_argument(
        "--selection_metric",
        choices=(
            "all_frame_accuracy",
            "visible_strict_recall",
            "visible_loose_recall",
            "invisible_no_box_rate",
            "recall_no_box_balance",
        ),
        default="all_frame_accuracy",
        help=(
            "Metric used to choose best_frame_predictions.csv from a sweep. "
            "Default preserves historical balanced/null behavior; ground-focused "
            "experiments can select visible_strict_recall explicitly."
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


def fnum(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def is_high_confidence_surface_branch(cand: Candidate, threshold: float) -> bool:
    if threshold > 1.0:
        return False
    if str(cand.row.get("gated_surface_branch", "")).strip() not in {"1", "true", "True"}:
        return False
    return fnum(cand.row.get("surface_halo_score"), -1.0) >= threshold


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + 0.5 * w, y + 0.5 * h


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def load_labels(path: Path, clip: str = "") -> dict[int, Label]:
    out: dict[int, Label] = {}
    for row in read_csv(path):
        if clip and row.get("clip", "") != clip:
            continue
        frame = int(fnum(row.get("frame"), -1))
        if frame < 0:
            continue
        visible = bool(int(fnum(row.get("visible"), 0)))
        bbox = None
        if visible:
            x = fnum(row.get("det_x", row.get("x")), math.nan)
            y = fnum(row.get("det_y", row.get("y")), math.nan)
            w = fnum(row.get("det_w", row.get("w")), math.nan)
            h = fnum(row.get("det_h", row.get("h")), math.nan)
            if all(math.isfinite(v) for v in (x, y, w, h)):
                bbox = (x, y, w, h)
        out[frame] = Label(visible=visible and bbox is not None, bbox=bbox)
    return out


def row_bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        fnum(row.get("x")),
        fnum(row.get("y")),
        max(1.0, fnum(row.get("w"), 1.0)),
        max(1.0, fnum(row.get("h"), 1.0)),
    )


def max_feature(row: dict[str, str], *names: str) -> float:
    return max(fnum(row.get(name)) for name in names)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clipped(value: float, limit: float = 3.0) -> float:
    return clamp(value, -limit, limit)


def pos(value: float, limit: float = 3.0) -> float:
    return max(0.0, min(limit, value))


def bounded_logit(prob: float) -> float:
    p = clamp(prob, 1e-4, 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def router_bucket(row: dict[str, str]) -> str:
    state = str(row.get("cand_router_state", "")).strip()
    if state == "surface_backed":
        return "surface"
    if state in {"boundary_mixed", "sky_target_near_surface"}:
        return "boundary"
    if state == "line_attached":
        return "line"
    if state == "clean_sky":
        return "clean_sky"
    rates = {
        "surface": fnum(row.get("tube_router_surface_backed_rate")),
        "clean_sky": fnum(row.get("tube_router_clean_sky_rate")),
        "boundary": fnum(row.get("tube_router_boundary_rate")),
        "line": fnum(row.get("tube_router_line_attached_rate")),
    }
    best, val = max(rates.items(), key=lambda item: item[1])
    if val >= 0.35:
        return best
    return "unknown"


def router_priors(bucket: str, explicit_null_priors: dict[str, float] | None = None) -> tuple[float, float, float, float]:
    """Return static/attached/boundary/null priors for a candidate router bucket."""

    defaults = {
        "clean_sky": (-0.15, -0.30, 0.05, -0.25),
        "surface": (0.25, 0.25, 0.00, 0.20),
        "line": (0.20, 0.50, 0.05, 0.25),
        "boundary": (0.10, 0.15, 0.45, 0.25),
        "unknown": (0.00, 0.00, 0.00, 0.10),
    }
    s, e, h, n = defaults.get(bucket, defaults["unknown"])
    if explicit_null_priors and bucket in explicit_null_priors:
        n = explicit_null_priors[bucket]
    elif explicit_null_priors and "global" in explicit_null_priors:
        n = explicit_null_priors["global"]
    return s, e, h, n


def parse_null_priors(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit(f"bad null prior entry: {part!r}; expected bucket=value")
        key, value = part.split("=", 1)
        out[key.strip()] = float(value)
    return out


def proposal_prior(
    row: dict[str, str],
    score_column: str,
    score_weight: float,
    rank_weight: float,
    proposal_clip: float = 2.0,
) -> tuple[float, float]:
    rank = int(fnum(row.get("rank"), 999999))
    raw_score = fnum(row.get(score_column), fnum(row.get("score")))
    if 0.0 <= raw_score <= 1.0:
        score_term = bounded_logit(raw_score)
    else:
        score_term = math.log1p(max(0.0, raw_score))
    prior = score_weight * clipped(score_term, proposal_clip) - rank_weight * math.log1p(max(1, rank))
    return prior, raw_score


def learned_prior_value(candidate: Candidate, args: argparse.Namespace) -> float:
    if getattr(args, "learned_prior_source", "proposal") == "raw_score":
        raw_score = candidate.raw_score
        if 0.0 <= raw_score <= 1.0:
            score_term = bounded_logit(raw_score)
        else:
            score_term = math.log1p(max(0.0, raw_score))
        return clipped(score_term, getattr(args, "learned_prior_clip", 2.0))
    return clipped(candidate.proposal_prior, getattr(args, "learned_prior_clip", 2.0))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "none", "nan"}


def surface_branch_rank_bonus(candidate: Candidate, args: argparse.Namespace | None) -> float:
    if args is None or not truthy(candidate.row.get("gated_surface_branch", "")):
        return 0.0
    bonus = float(getattr(args, "surface_branch_rank_bonus", 0.0))
    if bonus <= 0.0:
        return 0.0
    rank = fnum(candidate.row.get("surface_halo_parent_rank"), fnum(candidate.row.get("rank"), float(candidate.rank)))
    decay = float(getattr(args, "surface_branch_rank_decay", 0.35))
    return max(0.0, bonus - decay * math.log1p(max(1.0, rank)))


def joint_candidate_observations(
    row: dict[str, str],
    score_column: str,
    score_weight: float,
    clba_weight: float,
    path_weight: float,
    static_weight: float,
    attached_weight: float,
    rank_weight: float,
    proposal_clip: float = 2.0,
    proposal_mode: str = "shared",
    explicit_null_priors: dict[str, float] | None = None,
) -> dict[str, float | str]:
    proposal, raw_score = proposal_prior(row, score_column, score_weight, rank_weight, proposal_clip)
    bucket = router_bucket(row)
    static_prior, attached_prior, boundary_prior, null_prior = router_priors(bucket, explicit_null_priors)
    clba_gain = fnum(row.get("clba_gain_norm"), fnum(row.get("tube_mean_align_gain")))
    target_like = fnum(row.get("clba_target_likelihood"), clba_gain)
    bg_static_like = fnum(row.get("clba_bg_static_likelihood"))
    attached_like = fnum(row.get("clba_attached_likelihood"))
    target_q = fnum(row.get("clba_target_q"), fnum(row.get("tube_mean_native_dark_score")))
    bg_q = fnum(row.get("clba_bg_q"))
    path_dist = fnum(row.get("clba_path_bg_dist_mean"), fnum(row.get("tube_mean_bg_dist")))
    line = max_feature(row, "cand_line_context", "tube_mean_line_context")
    support = max_feature(row, "cand_attached_support", "tube_mean_attached_support")
    density = fnum(row.get("tube_log_cand_density"), math.log1p(max_feature(row, "tube_mean_cand_density")))
    pair = max_feature(row, "tube_mean_pair_bg", "tube_mean_pair_score")
    pair_rate = max_feature(row, "tube_positive_pair_bg_rate", "tube_positive_pair_rate")
    bg_dist = fnum(row.get("tube_mean_bg_dist"), path_dist)
    cv_resid = fnum(row.get("tube_mean_cv_resid"))
    bg_minus_cv = fnum(row.get("tube_mean_bg_minus_cv"))
    texture = max_feature(row, "cand_texture", "tube_mean_texture")
    sky_like = max_feature(row, "cand_sky_like", "tube_mean_sky_like")
    bg_anisotropy = fnum(row.get("clba_bg_anisotropy"))
    boundary_rate = fnum(row.get("tube_router_boundary_rate"))

    target_raw = (
        1.10 * clipped(target_like)
        + clba_weight * 1.20 * clipped(clba_gain)
        + 0.18 * clipped(target_q, 4.0)
        + path_weight * min(2.0, path_dist / 8.0)
        + 0.20 * pos(pair)
        + 0.18 * pos(pair_rate * 2.0)
    )
    static_raw = (
        static_weight * 1.10 * pos(bg_static_like)
        + 0.45 * pos(bg_q)
        + 0.45 * max(0.0, 1.0 - min(path_dist, 6.0) / 6.0)
        + 0.25 * max(0.0, 1.0 - min(bg_dist, 6.0) / 6.0)
        + 0.12 * pos(line)
        + 0.08 * pos(density)
        + 0.08 * max(0.0, 1.0 - min(cv_resid, 6.0) / 6.0)
    )
    attached_raw = (
        attached_weight * (1.00 * pos(attached_like) + 0.55 * pos(line) + 0.04 * pos(support))
        + 0.12 * pos(density)
        + 0.20 * pos(texture)
        - 0.15 * pos(clba_gain)
    )
    boundary_raw = (
        0.55 * pos(boundary_rate * 3.0)
        + 0.35 * pos(bg_anisotropy)
        + 0.25 * pos(sky_like * texture)
        + 0.25 * pos(bg_minus_cv)
        + 0.10 * pos(line)
    )
    generic_raw = (
        0.18 * pos(density)
        + 0.12 * pos(texture)
        + 0.08 * max(0.0, 1.0 - min(max(0.0, clba_gain), 2.0) / 2.0)
    )

    clutter_proposal = proposal if proposal_mode == "shared" else 0.0
    o_t = proposal + target_raw
    o_s = clutter_proposal + static_raw + static_prior
    o_e = clutter_proposal + attached_raw + attached_prior
    o_h = clutter_proposal + boundary_raw + boundary_prior
    o_g = 0.35 * clutter_proposal + generic_raw
    o_n = null_prior
    target_llr = o_t - logsumexp([o_s, o_e, o_h, o_g, o_n])
    static_llr = o_s - logsumexp([o_t, o_e, o_h, o_g, o_n])
    attached_llr = o_e - logsumexp([o_t, o_s, o_h, o_g, o_n])
    boundary_llr = o_h - logsumexp([o_t, o_s, o_e, o_g, o_n])
    generic_llr = o_g - logsumexp([o_t, o_s, o_e, o_h, o_n])
    null_llr = o_n - logsumexp([o_t, o_s, o_e, o_h, o_g])
    return {
        "target_obs": o_t,
        "static_obs": o_s,
        "attached_obs": o_e,
        "boundary_obs": o_h,
        "generic_obs": o_g,
        "null_obs": o_n,
        "target_llr": target_llr,
        "static_llr": static_llr,
        "attached_llr": attached_llr,
        "boundary_llr": boundary_llr,
        "generic_llr": generic_llr,
        "null_llr": null_llr,
        "target_margin": o_t - max(o_s, o_e, o_h, o_g, o_n),
        "router_bucket": bucket,
        "proposal_prior": proposal,
        "raw_score": raw_score,
    }


def candidate_observations(
    row: dict[str, str],
    score_column: str,
    score_weight: float,
    clba_weight: float,
    path_weight: float,
    static_weight: float,
    attached_weight: float,
    rank_weight: float,
) -> tuple[float, float, float, float]:
    obs = joint_candidate_observations(
        row,
        score_column,
        score_weight,
        clba_weight,
        path_weight,
        static_weight,
        attached_weight,
        rank_weight,
        2.0,
        "shared",
    )
    return (
        float(obs["target_obs"]),
        float(obs["static_obs"]),
        float(obs["attached_obs"]),
        float(obs["raw_score"]),
    )


def load_candidates(
    path: Path,
    clip: str,
    max_rank: int,
    score_column: str,
    args: argparse.Namespace,
    labels: dict[int, Label] | None = None,
    strict_tol_px: float = 8.0,
) -> dict[int, list[Candidate]]:
    out: dict[int, list[Candidate]] = {}
    null_priors = parse_null_priors(args.null_priors)
    for row in read_csv(path):
        row_clip = str(row.get("clip", "")).strip()
        if clip and row_clip and row_clip != clip:
            continue
        frame = int(fnum(row.get("frame"), -1))
        rank = int(fnum(row.get("rank"), 999999))
        if frame < 0 or rank > max_rank:
            continue
        obs = joint_candidate_observations(
            row,
            score_column,
            args.score_weight,
            args.clba_weight,
            args.path_weight,
            args.static_weight,
            args.attached_weight,
            args.rank_weight,
            args.proposal_clip,
            args.proposal_mode,
            null_priors,
        )
        cand = Candidate(
            frame=frame,
            rank=rank,
            bbox=row_bbox(row),
            target_obs=float(obs["target_obs"]),
            static_obs=float(obs["static_obs"]),
            attached_obs=float(obs["attached_obs"]),
            raw_score=float(obs["raw_score"]),
            row=row,
            boundary_obs=float(obs["boundary_obs"]),
            generic_obs=float(obs["generic_obs"]),
            null_obs=float(obs["null_obs"]),
            target_llr=float(obs["target_llr"]),
            static_llr=float(obs["static_llr"]),
            attached_llr=float(obs["attached_llr"]),
            boundary_llr=float(obs["boundary_llr"]),
            generic_llr=float(obs["generic_llr"]),
            null_llr=float(obs["null_llr"]),
            target_margin=float(obs["target_margin"]),
            router_bucket=str(obs["router_bucket"]),
            proposal_prior=float(obs["proposal_prior"]),
        )
        cand = apply_observation_mode(
            cand,
            getattr(args, "observation_mode", "js1"),
            None if labels is None else labels.get(frame),
            strict_tol_px,
            args,
        )
        out.setdefault(frame, []).append(cand)
    for rows in out.values():
        rows.sort(key=lambda c: c.rank)
    return out


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def recalculated_candidate_with_null(candidate: Candidate, null_offset: float) -> Candidate:
    o_t = candidate.target_obs
    o_s = candidate.static_obs
    o_e = candidate.attached_obs
    o_h = candidate.boundary_obs
    o_g = candidate.generic_obs
    o_n = candidate.null_obs + null_offset
    return replace(
        candidate,
        null_obs=o_n,
        target_llr=o_t - logsumexp([o_s, o_e, o_h, o_g, o_n]),
        static_llr=o_s - logsumexp([o_t, o_e, o_h, o_g, o_n]),
        attached_llr=o_e - logsumexp([o_t, o_s, o_h, o_g, o_n]),
        boundary_llr=o_h - logsumexp([o_t, o_s, o_e, o_g, o_n]),
        generic_llr=o_g - logsumexp([o_t, o_s, o_e, o_h, o_n]),
        null_llr=o_n - logsumexp([o_t, o_s, o_e, o_h, o_g]),
        target_margin=o_t - max(o_s, o_e, o_h, o_g, o_n),
    )


def candidate_with_observations(
    candidate: Candidate,
    target_obs: float,
    static_obs: float,
    attached_obs: float,
    boundary_obs: float,
    null_obs: float,
    generic_obs: float = 0.0,
) -> Candidate:
    return replace(
        candidate,
        target_obs=target_obs,
        static_obs=static_obs,
        attached_obs=attached_obs,
        boundary_obs=boundary_obs,
        generic_obs=generic_obs,
        null_obs=null_obs,
        target_llr=target_obs - logsumexp([static_obs, attached_obs, boundary_obs, generic_obs, null_obs]),
        static_llr=static_obs - logsumexp([target_obs, attached_obs, boundary_obs, generic_obs, null_obs]),
        attached_llr=attached_obs - logsumexp([target_obs, static_obs, boundary_obs, generic_obs, null_obs]),
        boundary_llr=boundary_obs - logsumexp([target_obs, static_obs, attached_obs, generic_obs, null_obs]),
        generic_llr=generic_obs - logsumexp([target_obs, static_obs, attached_obs, boundary_obs, null_obs]),
        null_llr=null_obs - logsumexp([target_obs, static_obs, attached_obs, boundary_obs, generic_obs]),
        target_margin=target_obs - max(static_obs, attached_obs, boundary_obs, generic_obs, null_obs),
    )


def apply_observation_mode(
    candidate: Candidate,
    mode: str,
    label: Label | None,
    strict_tol_px: float,
    args: argparse.Namespace | None = None,
) -> Candidate:
    if mode == "js1":
        return candidate
    if mode == "old_score":
        return candidate_with_observations(candidate, candidate.proposal_prior, -6.0, -6.0, -6.0, candidate.null_obs, -6.0)
    if mode == "oracle":
        if label is not None and label.visible and label.bbox is not None:
            if center_dist(candidate.bbox, label.bbox) <= strict_tol_px:
                return candidate_with_observations(candidate, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0)
            return candidate_with_observations(candidate, -10.0, 0.0, 0.0, 0.0, -2.0, 0.0)
        return candidate_with_observations(candidate, -10.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    if mode == "learned_logits":
        row = candidate.row
        explicit_null_priors = parse_null_priors(getattr(args, "null_priors", "")) if args is not None else None
        static_prior, attached_prior, boundary_prior, null_prior = router_priors(
            candidate.router_bucket,
            explicit_null_priors,
        )
        prior_base = learned_prior_value(candidate, args) if args is not None else clipped(candidate.proposal_prior, 2.0)
        proposal = (getattr(args, "learned_target_prior_weight", 0.15) if args is not None else 0.15) * prior_base
        weak_clutter_proposal = (
            getattr(args, "learned_clutter_prior_weight", 0.05) if args is not None else 0.05
        ) * prior_base
        generic_proposal = (getattr(args, "learned_generic_prior_weight", 0.02) if args is not None else 0.02) * prior_base
        target_obs = (
            clipped(fnum(row.get("crop_t_logit"), candidate.target_obs), 6.0)
            + proposal
            + surface_branch_rank_bonus(candidate, args)
        )
        static_obs = clipped(fnum(row.get("crop_s_logit"), candidate.static_obs), 6.0) + weak_clutter_proposal + static_prior
        attached_obs = clipped(fnum(row.get("crop_e_logit"), candidate.attached_obs), 6.0) + weak_clutter_proposal + attached_prior
        boundary_obs = clipped(fnum(row.get("crop_h_logit"), candidate.boundary_obs), 6.0) + weak_clutter_proposal + boundary_prior
        generic_obs = clipped(fnum(row.get("crop_g_logit"), candidate.generic_obs), 6.0) + generic_proposal
        return candidate_with_observations(
            candidate,
            target_obs,
            static_obs,
            attached_obs,
            boundary_obs,
            null_prior,
            generic_obs,
        )
    raise ValueError(f"unknown observation mode: {mode}")


def calibrate_null_offsets(
    labels: dict[int, Label],
    candidates: dict[int, list[Candidate]],
    q: float,
    min_router_null_samples: int,
    margin: float,
) -> tuple[dict[str, float], dict[int, list[Candidate]], list[dict[str, Any]]]:
    """Calibrate router-specific null offsets from max target LLR on invisible frames."""

    by_bucket: dict[str, list[float]] = defaultdict(list)
    for frame, lab in labels.items():
        if lab.visible:
            continue
        frame_best: dict[str, float] = defaultdict(lambda: -math.inf)
        for cand in candidates.get(frame, []):
            frame_best[cand.router_bucket] = max(frame_best[cand.router_bucket], cand.target_llr)
            frame_best["global"] = max(frame_best["global"], cand.target_llr)
        for bucket, value in frame_best.items():
            if math.isfinite(value):
                by_bucket[bucket].append(value)
    global_thr = quantile(by_bucket.get("global", []), q) + margin
    offsets: dict[str, float] = {"global": max(0.0, global_thr)}
    rows: list[dict[str, Any]] = [
        {"router_bucket": "global", "null_samples": len(by_bucket.get("global", [])), "null_offset": offsets["global"]}
    ]
    for bucket, vals in sorted(by_bucket.items()):
        if bucket == "global":
            continue
        if len(vals) < min_router_null_samples:
            offsets[bucket] = offsets["global"]
        else:
            offsets[bucket] = max(0.0, quantile(vals, q) + margin)
        rows.append({"router_bucket": bucket, "null_samples": len(vals), "null_offset": offsets[bucket]})
    adjusted: dict[int, list[Candidate]] = {}
    for frame, cands in candidates.items():
        adjusted[frame] = [
            recalculated_candidate_with_null(c, offsets.get(c.router_bucket, offsets.get("global", 0.0)))
            for c in cands
        ]
    return offsets, adjusted, rows


def student_t_neglog_radius(radius: float, sigma: float, nu: float = 3.0) -> float:
    # Constant terms cancel in path comparison; keep the heavy-tailed shape.
    s = max(1e-3, sigma)
    return 0.5 * (nu + 1.0) * math.log1p((radius / s) ** 2 / nu)


def range_bin_motion_cost(
    radius: float,
    fps: float,
    image_width: float,
    horizontal_fov_deg: float,
    vmax_mps: float,
    registration_sigma_px: float,
    box_sigma_px: float,
) -> float:
    f_px = image_width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    dt = 1.0 / max(1e-3, fps)
    ranges = [3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0]
    priors = [0.08, 0.12, 0.18, 0.20, 0.18, 0.12, 0.08, 0.04]
    vals: list[float] = []
    for rng, prior in zip(ranges, priors):
        sigma = f_px * vmax_mps * dt / rng + registration_sigma_px + box_sigma_px
        vals.append(math.log(prior) - student_t_neglog_radius(radius, sigma))
    return -logsumexp(vals)


def motion_cost(
    prev: Hypothesis,
    cand: Candidate,
    max_jump_px: float,
    motion_weight: float,
    fps: float,
    image_width: float,
    horizontal_fov_deg: float,
    vmax_mps: float,
    registration_sigma_px: float,
    box_sigma_px: float,
    motion_prior_weight: float,
    absurd_jump_px: float,
) -> float:
    if prev.bbox is None:
        return 0.0
    pred = prev.bbox
    if prev.state in {"T", "C", "P"}:
        x, y, w, h = prev.bbox
        pred = (x + prev.vx, y + prev.vy, w, h)
    jump = center_dist(pred, cand.bbox)
    allowed = max_jump_px + 0.7 * max(prev.bbox[2], prev.bbox[3], cand.bbox[2], cand.bbox[3])
    if jump > max(absurd_jump_px, 3.0 * allowed):
        return 1e6
    range_cost = range_bin_motion_cost(
        jump,
        fps,
        image_width,
        horizontal_fov_deg,
        vmax_mps,
        registration_sigma_px,
        box_sigma_px,
    )
    return motion_weight * (jump / max(1.0, allowed)) ** 2 + motion_prior_weight * range_cost


def predicted_center_error(prev: Hypothesis, cand: Candidate) -> float:
    if prev.bbox is None:
        return float("inf")
    x, y, w, h = prev.bbox
    pred = (x + prev.vx, y + prev.vy, w, h)
    return center_dist(pred, cand.bbox)


def continuity_adjusted_target_llr(
    hyp: Hypothesis,
    cand: Candidate,
    target_llr: float,
    continuity_bonus: float,
    continuity_max_pred_error_px: float,
    continuity_clutter_gap: float,
    continuity_min_raw_score: float,
) -> float:
    if continuity_bonus <= 0.0 or hyp.state not in {"T", "C"} or hyp.bbox is None:
        return target_llr
    if cand.raw_score < continuity_min_raw_score:
        return target_llr
    if predicted_center_error(hyp, cand) > continuity_max_pred_error_px:
        return target_llr
    best_clutter_obs = max(cand.static_obs, cand.attached_obs, cand.boundary_obs, cand.generic_obs)
    if cand.target_obs + continuity_clutter_gap < best_clutter_obs:
        return target_llr
    return target_llr + continuity_bonus


def update_velocity(prev: Hypothesis, cand: Candidate) -> tuple[float, float]:
    if prev.bbox is None:
        return 0.0, 0.0
    pcx, pcy = center(prev.bbox)
    ccx, ccy = center(cand.bbox)
    return ccx - pcx, ccy - pcy


def in_quarantine(hyp: Hypothesis, cand: Candidate, radius: float) -> bool:
    if hyp.quarantine_bbox is None or hyp.lock_age <= 0:
        return False
    return center_dist(hyp.quarantine_bbox, cand.bbox) <= radius


def active_quarantines(hyps: list[Hypothesis]) -> list[tuple[tuple[float, float, float, float], int]]:
    """Collect active S/E clutter anchors across the whole beam.

    Quarantine is a tracker-level memory, not a single path-local preference.
    If one plausible hypothesis identifies a branch/static lock, the next frame
    should not let the independent absent path immediately birth the same
    candidate again.
    """

    anchors: list[tuple[tuple[float, float, float, float], int]] = []
    for hyp in hyps:
        if hyp.quarantine_bbox is not None and hyp.lock_age > 0:
            anchors.append((hyp.quarantine_bbox, hyp.lock_age))
    return anchors


def candidate_in_quarantine(
    cand: Candidate,
    quarantines: list[tuple[tuple[float, float, float, float], int]],
    radius: float,
) -> bool:
    return any(age > 0 and center_dist(anchor, cand.bbox) <= radius for anchor, age in quarantines)


def add_pruned(pool: list[Hypothesis], hyp: Hypothesis, beam_width: int) -> None:
    pool.append(hyp)
    if len(pool) > beam_width * 3:
        pool.sort(key=lambda h: h.score, reverse=True)
        del pool[beam_width:]


def prune_by_state(pool: list[Hypothesis], beam_width: int, state_beam: int) -> list[Hypothesis]:
    pool.sort(key=lambda h: h.score, reverse=True)
    counts: dict[str, int] = {}
    out: list[Hypothesis] = []
    for hyp in pool:
        n = counts.get(hyp.state, 0)
        if n >= state_beam:
            continue
        counts[hyp.state] = n + 1
        out.append(hyp)
        if len(out) >= beam_width:
            break
    return out


def absent_continuation_score(score: float, absent_reward: float, absent_score_cap: float) -> float:
    out = score + absent_reward
    if absent_score_cap >= 0.0 and math.isfinite(absent_score_cap):
        out = min(out, absent_score_cap)
    return out


def step_hypotheses(
    hyps: list[Hypothesis],
    cands: list[Candidate],
    acquire_threshold: float,
    track_threshold: float,
    acquire_hits: int,
    max_misses: int,
    max_jump_px: float,
    clutter_margin: float,
    quarantine_px: float,
    static_quarantine_frames: int,
    attached_quarantine_frames: int,
    global_quarantine: bool,
    quarantine_override_margin: float,
    motion_weight: float,
    fps: float = 30.0,
    image_width: float = 320.0,
    horizontal_fov_deg: float = 120.0,
    vmax_mps: float = 10.0,
    registration_sigma_px: float = 1.5,
    box_sigma_px: float = 2.0,
    motion_prior_weight: float = 0.25,
    absurd_jump_px: float = 96.0,
    continuity_bonus: float = 0.0,
    continuity_max_pred_error_px: float = 12.0,
    continuity_clutter_gap: float = 0.5,
    continuity_min_raw_score: float = 0.0,
    absent_reward: float = 0.05,
    absent_score_cap: float = 1e9,
    beam_width: int = 96,
    state_beam: int = 12,
    clutter_lock_gap: float = 0.0,
    instant_surface_acquire_score: float = 1.1,
) -> list[Hypothesis]:
    pool: list[Hypothesis] = []
    beam_quarantines = active_quarantines(hyps) if global_quarantine else []
    for hyp in hyps:
        q_age = max(0, hyp.lock_age - 1)
        q_bbox = hyp.quarantine_bbox if q_age > 0 else None
        # Null / absent continuation.
        if hyp.state in {"A", "S", "E"}:
            add_pruned(
                pool,
                Hypothesis(
                    "A",
                    absent_continuation_score(hyp.score, absent_reward, absent_score_cap),
                    quarantine_bbox=q_bbox,
                    lock_age=q_age,
                    reason="null",
                ),
                beam_width,
            )
        elif hyp.state == "C":
            if hyp.misses + 1 > max_misses:
                add_pruned(pool, Hypothesis("A", hyp.score - 0.1, quarantine_bbox=q_bbox, lock_age=q_age, reason="drop"), beam_width)
            else:
                add_pruned(
                    pool,
                    replace(hyp, score=hyp.score - 0.25, misses=hyp.misses + 1, lock_age=q_age, quarantine_bbox=q_bbox, reason="coast"),
                    beam_width,
                )
        elif hyp.state in {"P", "T"}:
            add_pruned(
                pool,
                replace(hyp, state="C", score=hyp.score - 0.45, misses=1, lock_age=q_age, quarantine_bbox=q_bbox, selected=None, reason="miss"),
                beam_width,
            )

        for cand in cands:
            clutter_scores = {
                "S": cand.static_llr,
                "E": max(cand.attached_llr, cand.boundary_llr),
            }
            clutter_state, clutter_edge = max(clutter_scores.items(), key=lambda item: item[1])
            target_llr = continuity_adjusted_target_llr(
                hyp,
                cand,
                cand.target_llr,
                continuity_bonus,
                continuity_max_pred_error_px,
                continuity_clutter_gap,
                continuity_min_raw_score,
            )
            if in_quarantine(hyp, cand, quarantine_px) or candidate_in_quarantine(cand, beam_quarantines, quarantine_px):
                target_margin = cand.target_margin
                if (
                    target_margin < quarantine_override_margin
                    or target_llr < acquire_threshold + quarantine_override_margin
                ):
                    continue
            mcost = (
                motion_cost(
                    hyp,
                    cand,
                    max_jump_px,
                    motion_weight,
                    fps,
                    image_width,
                    horizontal_fov_deg,
                    vmax_mps,
                    registration_sigma_px,
                    box_sigma_px,
                    motion_prior_weight,
                    absurd_jump_px,
                )
                if hyp.state in {"P", "T", "C"}
                else 0.0
            )
            if mcost >= 1e5:
                continue

            if clutter_edge >= clutter_margin and clutter_edge >= target_llr + clutter_lock_gap:
                ttl = attached_quarantine_frames if clutter_state == "E" else static_quarantine_frames
                add_pruned(
                    pool,
                    Hypothesis(
                        clutter_state,
                        hyp.score - 0.05 + 0.05 * min(1.0, clutter_edge),
                        bbox=cand.bbox,
                        lock_age=ttl,
                        quarantine_bbox=cand.bbox,
                        quarantine_kind=clutter_state,
                        router_bucket=cand.router_bucket,
                        reason="clutter_lock",
                    ),
                    beam_width,
                )

            if target_llr < acquire_threshold and hyp.state in {"A", "S", "E"}:
                continue
            if hyp.state == "A" or hyp.state in {"S", "E"}:
                hits = 1 if target_llr >= acquire_threshold else 0
                # A/S/E never emit immediately. S/E locks must release through
                # a fresh tentative target state before a box can be emitted.
                instant_surface_acquire = (
                    hyp.state == "A"
                    and hits >= acquire_hits
                    and target_llr >= track_threshold
                    and is_high_confidence_surface_branch(cand, instant_surface_acquire_score)
                )
                new_state = "T" if instant_surface_acquire else "P"
                add_pruned(
                    pool,
                    Hypothesis(
                        new_state,
                        hyp.score + target_llr + (0.15 if new_state == "T" else -0.35),
                        bbox=cand.bbox,
                        hits=hits,
                        age=1,
                        selected=cand if new_state == "T" else None,
                        quarantine_bbox=q_bbox,
                        lock_age=q_age,
                        router_bucket=cand.router_bucket,
                        target_logodds=target_llr,
                        overrides=hyp.overrides
                        + int(hyp.state in {"S", "E"} and cand.target_margin >= quarantine_override_margin),
                        reason="instant_acquire" if new_state == "T" else "birth",
                    ),
                    beam_width,
                )
            elif hyp.state == "P":
                hits = hyp.hits + (1 if target_llr >= acquire_threshold else 0)
                vx, vy = update_velocity(hyp, cand)
                new_state = "T" if hits >= acquire_hits and target_llr >= track_threshold else "P"
                add_pruned(
                    pool,
                    Hypothesis(
                        new_state,
                        hyp.score + target_llr - mcost + (0.35 if new_state == "T" else 0.0),
                        bbox=cand.bbox,
                        vx=vx,
                        vy=vy,
                        hits=hits,
                        age=hyp.age + 1,
                        selected=cand if new_state == "T" else None,
                        quarantine_bbox=q_bbox,
                        lock_age=q_age,
                        router_bucket=cand.router_bucket,
                        target_logodds=target_llr,
                        overrides=hyp.overrides,
                        reason="acquire" if new_state == "T" else "tentative",
                    ),
                    beam_width,
                )
            elif hyp.state in {"T", "C"}:
                if target_llr < track_threshold:
                    continue
                vx, vy = update_velocity(hyp, cand)
                add_pruned(
                    pool,
                    Hypothesis(
                        "T",
                        hyp.score + target_llr - mcost + 0.15,
                        bbox=cand.bbox,
                        vx=vx,
                        vy=vy,
                        hits=hyp.hits + 1,
                        age=hyp.age + 1,
                        selected=cand,
                        quarantine_bbox=q_bbox,
                        lock_age=q_age,
                        router_bucket=cand.router_bucket,
                        target_logodds=target_llr,
                        overrides=hyp.overrides,
                        reason="track",
                    ),
                    beam_width,
                )
    return prune_by_state(pool, beam_width, state_beam) or [Hypothesis("A", 0.0)]


def evaluate_selector(
    labels: dict[int, Label],
    candidates: dict[int, list[Candidate]],
    acquire_threshold: float,
    track_threshold: float,
    acquire_hits: int,
    max_misses: int,
    max_jump_px: float,
    clutter_margin: float,
    strict_tol_px: float,
    loose_tol_px: float,
    quarantine_px: float,
    static_quarantine_frames: int,
    attached_quarantine_frames: int,
    global_quarantine: bool,
    quarantine_override_margin: float,
    motion_weight: float,
    fps: float = 30.0,
    image_width: float = 320.0,
    horizontal_fov_deg: float = 120.0,
    vmax_mps: float = 10.0,
    registration_sigma_px: float = 1.5,
    box_sigma_px: float = 2.0,
    motion_prior_weight: float = 0.25,
    absurd_jump_px: float = 96.0,
    continuity_bonus: float = 0.0,
    continuity_max_pred_error_px: float = 12.0,
    continuity_clutter_gap: float = 0.5,
    continuity_min_raw_score: float = 0.0,
    absent_reward: float = 0.05,
    absent_score_cap: float = 1e9,
    beam_width: int = 96,
    state_beam: int = 12,
    clutter_lock_gap: float = 0.0,
    instant_surface_acquire_score: float = 1.1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hyps = [Hypothesis("A", 0.0)]
    rows: list[dict[str, Any]] = []
    first_visible = next((f for f in sorted(labels) if labels[f].visible), None)
    first_strict = None
    first_lock = None
    processed_frames = sorted(set(labels) | set(candidates))
    for frame in processed_frames:
        hyps = step_hypotheses(
            hyps,
            candidates.get(frame, []),
            acquire_threshold,
            track_threshold,
            acquire_hits,
            max_misses,
            max_jump_px,
            clutter_margin,
            quarantine_px,
            static_quarantine_frames,
            attached_quarantine_frames,
            global_quarantine,
            quarantine_override_margin,
            motion_weight,
            fps,
            image_width,
            horizontal_fov_deg,
            vmax_mps,
            registration_sigma_px,
            box_sigma_px,
            motion_prior_weight,
            absurd_jump_px,
            continuity_bonus,
            continuity_max_pred_error_px,
            continuity_clutter_gap,
            continuity_min_raw_score,
            absent_reward,
            absent_score_cap,
            beam_width,
            state_beam,
            clutter_lock_gap,
            instant_surface_acquire_score,
        )
        best = hyps[0]
        lab = labels.get(frame)
        is_labeled = lab is not None
        if lab is None:
            lab = Label(visible=False, bbox=None)
        selected = best.state == "T" and best.selected is not None
        selected_bbox = best.selected.bbox if selected and best.selected is not None else None
        sx = sy = sw = sh = ""
        if selected_bbox is not None:
            sx, sy, sw, sh = (round(float(v), 3) for v in selected_bbox)
        dist = ""
        strict_hit = False
        loose_hit = False
        if selected_bbox is not None and lab.visible and lab.bbox is not None:
            d = center_dist(selected_bbox, lab.bbox)
            dist = round(d, 3)
            strict_hit = d <= strict_tol_px
            loose_hit = d <= loose_tol_px
            if strict_hit and first_strict is None:
                first_strict = frame
        if selected and first_lock is None:
            first_lock = frame
        rows.append(
            {
                "frame": frame,
                "labeled": int(is_labeled),
                "visible": int(lab.visible),
                "state": best.state,
                "selected": int(selected),
                "x": sx,
                "y": sy,
                "w": sw,
                "h": sh,
                "rank": "" if best.selected is None else best.selected.rank,
                "target_obs": "" if best.selected is None else round(best.selected.target_obs, 6),
                "static_obs": "" if best.selected is None else round(best.selected.static_obs, 6),
                "attached_obs": "" if best.selected is None else round(best.selected.attached_obs, 6),
                "boundary_obs": "" if best.selected is None else round(best.selected.boundary_obs, 6),
                "generic_obs": "" if best.selected is None else round(best.selected.generic_obs, 6),
                "null_obs": "" if best.selected is None else round(best.selected.null_obs, 6),
                "target_llr": "" if best.selected is None else round(best.selected.target_llr, 6),
                "static_llr": "" if best.selected is None else round(best.selected.static_llr, 6),
                "attached_llr": "" if best.selected is None else round(best.selected.attached_llr, 6),
                "boundary_llr": "" if best.selected is None else round(best.selected.boundary_llr, 6),
                "generic_llr": "" if best.selected is None else round(best.selected.generic_llr, 6),
                "target_margin": "" if best.selected is None else round(best.selected.target_margin, 6),
                "router_bucket": best.router_bucket,
                "lock_age": best.lock_age,
                "quarantine_kind": best.quarantine_kind,
                "overrides": best.overrides,
                "raw_score": "" if best.selected is None else round(best.selected.raw_score, 6),
                "reason": best.reason,
                "dist_px": dist,
                "strict_hit": strict_hit,
                "loose_hit": loose_hit,
                "correct_all_frame": strict_hit if lab.visible else not selected,
            }
        )

    labeled_rows = [r for r in rows if r.get("labeled")]
    visible_rows = [r for r in labeled_rows if r["visible"]]
    invisible_rows = [r for r in labeled_rows if not r["visible"]]
    visible_strict = sum(bool(r["strict_hit"]) for r in visible_rows)
    visible_loose = sum(bool(r["loose_hit"]) for r in visible_rows)
    invisible_no_box = sum(not bool(r["selected"]) for r in invisible_rows)
    correct = sum(bool(r["correct_all_frame"]) for r in labeled_rows)
    summary = {
        "acquire_threshold": acquire_threshold,
        "track_threshold": track_threshold,
        "acquire_hits": acquire_hits,
        "max_misses": max_misses,
        "max_jump_px": max_jump_px,
        "clutter_margin": clutter_margin,
        "clutter_lock_gap": clutter_lock_gap,
        "frames_all": len(labeled_rows),
        "processed_frames": len(rows),
        "visible_frames": len(visible_rows),
        "invisible_frames": len(invisible_rows),
        "all_frame_correct": correct,
        "all_frame_accuracy": round(correct / max(1, len(labeled_rows)), 4),
        "visible_strict": visible_strict,
        "visible_strict_recall": round(visible_strict / max(1, len(visible_rows)), 4),
        "visible_loose": visible_loose,
        "visible_loose_recall": round(visible_loose / max(1, len(visible_rows)), 4),
        "invisible_no_box": invisible_no_box,
        "invisible_no_box_rate": round(invisible_no_box / max(1, len(invisible_rows)), 4),
        "selected_frames": sum(bool(r["selected"]) for r in rows),
        "first_lock_frame": "" if first_lock is None else first_lock,
        "first_strict_frame": "" if first_strict is None else first_strict,
        "strict_latency_frames": (
            "" if first_visible is None or first_strict is None else max(0, first_strict - first_visible)
        ),
        "false_boxes_per_min": round(
            (sum(bool(r["selected"]) for r in invisible_rows) / max(1.0, len(labeled_rows) / max(1e-3, fps) / 60.0)),
            4,
        ),
        "state_counts": json.dumps(dict(Counter(str(r["state"]) for r in rows)), sort_keys=True),
        "quarantine_events": sum(1 for r in rows if str(r.get("reason")) == "clutter_lock"),
        "quarantine_overrides": max([int(r.get("overrides") or 0) for r in rows] or [0]),
    }
    return summary, rows


def summarize_rows_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for name, group in sorted(groups.items()):
        visible_rows = [r for r in group if bool(r.get("visible"))]
        invisible_rows = [r for r in group if not bool(r.get("visible"))]
        selected_invisible = sum(bool(r.get("selected")) for r in invisible_rows)
        out.append(
            {
                key: name,
                "frames": len(group),
                "visible_frames": len(visible_rows),
                "invisible_frames": len(invisible_rows),
                "visible_strict": sum(bool(r.get("strict_hit")) for r in visible_rows),
                "visible_strict_recall": round(
                    sum(bool(r.get("strict_hit")) for r in visible_rows) / max(1, len(visible_rows)), 4
                ),
                "visible_loose": sum(bool(r.get("loose_hit")) for r in visible_rows),
                "visible_loose_recall": round(
                    sum(bool(r.get("loose_hit")) for r in visible_rows) / max(1, len(visible_rows)), 4
                ),
                "invisible_no_box": len(invisible_rows) - selected_invisible,
                "invisible_no_box_rate": round((len(invisible_rows) - selected_invisible) / max(1, len(invisible_rows)), 4),
                "selected_frames": sum(bool(r.get("selected")) for r in group),
            }
        )
    return out


def selected_false_rows(rows: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    bad: list[dict[str, Any]] = []
    for row in rows:
        selected = bool(row.get("selected"))
        if selected and (not bool(row.get("visible")) or not bool(row.get("loose_hit"))):
            bad.append(row)
    return bad[:limit]


def missed_visible_rows(rows: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    bad = [row for row in rows if bool(row.get("visible")) and not bool(row.get("strict_hit"))]
    return bad[:limit]


def summary_selection_key(summary: dict[str, Any], metric: str) -> tuple[float, float, float, float, int]:
    """Return a deterministic sweep ranking key for the requested primary metric."""

    strict = float(summary.get("visible_strict_recall", 0.0))
    loose = float(summary.get("visible_loose_recall", 0.0))
    no_box = float(summary.get("invisible_no_box_rate", 0.0))
    all_acc = float(summary.get("all_frame_accuracy", 0.0))
    selected = int(summary.get("selected_frames", 0))
    if metric == "visible_strict_recall":
        return (strict, loose, no_box, all_acc, -selected)
    if metric == "visible_loose_recall":
        return (loose, strict, no_box, all_acc, -selected)
    if metric == "invisible_no_box_rate":
        return (no_box, strict, loose, all_acc, -selected)
    if metric == "recall_no_box_balance":
        return (min(strict, no_box), strict, no_box, loose, -selected)
    return (all_acc, strict, no_box, loose, -selected)


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels), args.clip)
    candidates = load_candidates(
        Path(args.candidates),
        args.clip,
        args.max_rank,
        args.score_column,
        args,
        labels,
        args.strict_tol_px,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    null_offsets: dict[str, float] = {}
    null_rows: list[dict[str, Any]] = []
    if args.null_calibration_quantile >= 0.0:
        null_offsets, candidates, null_rows = calibrate_null_offsets(
            labels,
            candidates,
            args.null_calibration_quantile,
            args.min_router_null_samples,
            args.null_calibration_margin,
        )
        write_csv(out_dir / "null_calibration_thresholds.csv", null_rows)

    summaries: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    static_ttl = args.quarantine_frames or args.static_quarantine_frames
    attached_ttl = args.quarantine_frames or args.attached_quarantine_frames
    best_key = (-1.0, -1.0, -1.0, -1.0, 0)
    for aq in parse_float_list(args.acquire_thresholds):
        for tr in parse_float_list(args.track_thresholds):
            if tr > aq:
                continue
            for hits in parse_int_list(args.acquire_hits):
                for misses in parse_int_list(args.max_misses):
                    for jump in parse_float_list(args.max_jump_px):
                        for margin in parse_float_list(args.clutter_margin):
                            t0 = time.perf_counter()
                            summary, rows = evaluate_selector(
                                labels,
                                candidates,
                                aq,
                                tr,
                                hits,
                                misses,
                                jump,
                                margin,
                                args.strict_tol_px,
                                args.loose_tol_px,
                                args.quarantine_px,
                                static_ttl,
                                attached_ttl,
                                args.global_quarantine,
                                args.quarantine_override_margin,
                                args.motion_weight,
                                args.fps,
                                args.image_width,
                                args.horizontal_fov_deg,
                                args.vmax_mps,
                                args.registration_sigma_px,
                                args.box_sigma_px,
                                args.motion_prior_weight,
                                args.absurd_jump_px,
                                args.continuity_bonus,
                                args.continuity_max_pred_error_px,
                                args.continuity_clutter_gap,
                                args.continuity_min_raw_score,
                                args.absent_reward,
                                args.absent_score_cap,
                                args.beam_width,
                                args.state_beam,
                                args.clutter_lock_gap,
                                args.instant_surface_acquire_score,
                            )
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            summary["eval_ms_total"] = round(elapsed_ms, 3)
                            summary["eval_ms_per_frame"] = round(elapsed_ms / max(1, summary["frames_all"]), 6)
                            summaries.append(summary)
                            summary["selection_metric"] = args.selection_metric
                            key = summary_selection_key(summary, args.selection_metric)
                            if key > best_key:
                                best_key = key
                                best_rows = rows
    summaries.sort(
        key=lambda r: summary_selection_key(r, args.selection_metric),
        reverse=True,
    )
    write_csv(out_dir / "explicit_state_sweep.csv", summaries)
    write_csv(out_dir / "summary.csv", summaries[:1])
    write_csv(out_dir / "by_clip_summary.csv", [{**summaries[0], "clip": args.clip or "all"}] if summaries else [])
    write_csv(out_dir / "best_frame_predictions.csv", best_rows)
    write_csv(out_dir / "state_trace.csv", best_rows)
    write_csv(out_dir / "by_router_summary.csv", summarize_rows_by_key(best_rows, "router_bucket"))
    write_csv(out_dir / "state_occupancy.csv", summarize_rows_by_key(best_rows, "state"))
    write_csv(
        out_dir / "quarantine_events.csv",
        [row for row in best_rows if row.get("lock_age") or row.get("quarantine_kind") or row.get("reason") == "clutter_lock"],
    )
    write_csv(out_dir / "worst_false_locks.csv", selected_false_rows(best_rows))
    write_csv(out_dir / "worst_misses.csv", missed_visible_rows(best_rows))
    write_csv(
        out_dir / "timing_summary.csv",
        [
            {
                "stage": "offline_selector",
                "frames": len(best_rows),
                "timing_note": "per-config timing is reported in explicit_state_sweep.csv as eval_ms_total/eval_ms_per_frame",
            }
        ],
    )
    (out_dir / "best_config.json").write_text(json.dumps(summaries[0] if summaries else {}, indent=2))
    metadata = {
        "labels": args.labels,
        "candidates": args.candidates,
        "clip": args.clip,
        "score_column": args.score_column,
        "max_rank": args.max_rank,
        "proposal_mode": args.proposal_mode,
        "observation_mode": args.observation_mode,
        "proposal_clip": args.proposal_clip,
        "learned_prior_source": args.learned_prior_source,
        "learned_target_prior_weight": args.learned_target_prior_weight,
        "learned_clutter_prior_weight": args.learned_clutter_prior_weight,
        "learned_generic_prior_weight": args.learned_generic_prior_weight,
        "learned_prior_clip": args.learned_prior_clip,
        "surface_branch_rank_bonus": args.surface_branch_rank_bonus,
        "surface_branch_rank_decay": args.surface_branch_rank_decay,
        "global_quarantine": args.global_quarantine,
        "clutter_lock_gap": args.clutter_lock_gap,
        "instant_surface_acquire_score": args.instant_surface_acquire_score,
        "quarantine_override_margin": args.quarantine_override_margin,
        "continuity_bonus": args.continuity_bonus,
        "continuity_max_pred_error_px": args.continuity_max_pred_error_px,
        "continuity_clutter_gap": args.continuity_clutter_gap,
        "continuity_min_raw_score": args.continuity_min_raw_score,
        "absent_reward": args.absent_reward,
        "absent_score_cap": args.absent_score_cap,
        "selection_metric": args.selection_metric,
        "static_quarantine_frames": static_ttl,
        "attached_quarantine_frames": attached_ttl,
        "fps": args.fps,
        "null_calibration_quantile": args.null_calibration_quantile,
        "null_offsets": null_offsets,
        "states": STATES,
        "candidate_frames": len(candidates),
        "label_frames": len(labels),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "README.md").write_text(
        "# CLBA-JS1 Explicit-State Selector\n\n"
        "Offline candidate-state selector implementing the professor's CLBA-JS1 skeleton: "
        "`A/P/T/S/E/C` states, target/static/attached/boundary/null competing observations, "
        "soft quarantine, and range-bin motion cost. This artifact does not change runtime defaults.\n\n"
        "Primary outputs: `summary.csv`, `explicit_state_sweep.csv`, `state_trace.csv`, "
        "`by_router_summary.csv`, `quarantine_events.csv`, `worst_false_locks.csv`, and "
        "`worst_misses.csv`.\n"
    )
    print(out_dir / "explicit_state_sweep.csv")
    if summaries:
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
