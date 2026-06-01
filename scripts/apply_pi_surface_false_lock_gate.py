#!/usr/bin/env python3
"""Causal post-gate for Pi profile surface false locks.

The Pi-light profile is useful because it keeps sky target recall, but on the
new global-camera clips it also acquires terrain/tree texture before any target
is visible. This script is intentionally small and interpretable: it suppresses
selected boxes that look like surface/attached clutter unless they are supported
by a recent target-like selection and plausible continuity.

It is an offline/lab post-processor first. If the rule keeps improving external
Pi-camera clips, the same features can be moved into the live acquisition gate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


SELECTED_FIELDS = [
    "clip",
    "frame",
    "selected",
    "rank",
    "x",
    "y",
    "w",
    "h",
    "learned_score",
    "track_id",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected_tubes", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--decisions_csv", default="")
    p.add_argument("--events_csv", default="")
    p.add_argument("--recent_support_ttl", type=int, default=30)
    p.add_argument("--max_supported_jump_px", type=float, default=70.0)
    p.add_argument("--risk_threshold", type=float, default=1.0)
    p.add_argument("--strong_support_threshold", type=float, default=1.0)
    p.add_argument("--score_floor", type=float, default=-1e9)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def center(row: dict[str, Any]) -> tuple[float, float]:
    return fnum(row.get("x")) + 0.5 * fnum(row.get("w"), 1.0), fnum(row.get("y")) + 0.5 * fnum(row.get("h"), 1.0)


def dist_px(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(math.hypot(ax - bx, ay - by))


def selected_rows(rows: list[dict[str, str]], score_floor: float) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if str(row.get("selected", "1")).strip().lower() in {"0", "false", "no", ""}:
            continue
        if not row.get("frame") or not row.get("x") or not row.get("y"):
            continue
        if fnum(row.get("verified_score", row.get("score")), -1e9) < score_floor:
            continue
        out.append(row)
    out.sort(key=lambda r: (inum(r.get("frame")), inum(r.get("rank"), 1)))
    by_frame: dict[int, dict[str, str]] = {}
    for row in out:
        frame = inum(row.get("frame"))
        # Keep only the chosen selected row per frame. If multiple rows are
        # marked selected, prefer higher verified score then lower rank.
        old = by_frame.get(frame)
        if old is None:
            by_frame[frame] = row
            continue
        old_key = (fnum(old.get("verified_score", old.get("score")), -1e9), -inum(old.get("rank"), 9999))
        new_key = (fnum(row.get("verified_score", row.get("score")), -1e9), -inum(row.get("rank"), 9999))
        if new_key > old_key:
            by_frame[frame] = row
    return [by_frame[f] for f in sorted(by_frame)]


def surface_false_lock_risk(row: dict[str, Any]) -> tuple[float, str]:
    """Return an interpretable risk score for terrain/tree false locks."""
    texture = max(fnum(row.get("cand_texture")), fnum(row.get("tube_mean_texture")))
    sky = max(fnum(row.get("cand_sky_like")), fnum(row.get("tube_mean_sky_like")))
    app_only = fnum(row.get("tube_appearance_only_rate"))
    line = max(fnum(row.get("cand_line_context")), fnum(row.get("tube_mean_line_context")))
    attach = max(fnum(row.get("cand_attached_support")), fnum(row.get("tube_mean_attached_support")))
    pair = fnum(row.get("tube_mean_pair_score"))
    pos_pair = fnum(row.get("tube_positive_pair_rate"))
    y = fnum(row.get("y"))
    source = str(row.get("cand_source", "")).strip().lower()

    risk = 0.0
    reasons: list[str] = []
    if app_only >= 0.75:
        risk += 0.65
        reasons.append("appearance_only")
    if texture >= 55:
        risk += 0.85
        reasons.append("high_texture")
    elif texture >= 42:
        risk += 0.45
        reasons.append("texture")
    if sky <= 0.03:
        risk += 0.35
        reasons.append("no_sky_support")
    if source == "appearance":
        risk += 0.25
        reasons.append("appearance_source")
    if y >= 250:
        risk += 0.45
        reasons.append("low_frame")
    if line >= 0.35:
        risk += 0.20
        reasons.append("line_context")
    if attach >= 4.0:
        risk += 0.25
        reasons.append("attached_support")
    if pair < 0.35 and pos_pair < 0.5:
        risk += 0.25
        reasons.append("weak_pair")
    if sky >= 0.12 and texture <= 35 and app_only <= 0.25:
        risk -= 0.85
        reasons.append("sky_like_relief")
    if source == "map" and texture <= 25 and pair >= 0.75:
        risk -= 0.45
        reasons.append("map_pair_relief")
    return risk, ";".join(reasons)


def target_support_score(row: dict[str, Any]) -> tuple[float, str]:
    """Cheap support score for updating target continuity memory."""
    texture = max(fnum(row.get("cand_texture")), fnum(row.get("tube_mean_texture")))
    sky = max(fnum(row.get("cand_sky_like")), fnum(row.get("tube_mean_sky_like")))
    app_only = fnum(row.get("tube_appearance_only_rate"))
    pair = fnum(row.get("tube_mean_pair_score"))
    pos_pair = fnum(row.get("tube_positive_pair_rate"))
    score = fnum(row.get("verified_score", row.get("score")))
    source = str(row.get("cand_source", "")).strip().lower()

    support = 0.0
    reasons: list[str] = []
    if sky >= 0.08:
        support += 0.55
        reasons.append("sky")
    if texture <= 35:
        support += 0.40
        reasons.append("low_texture")
    if app_only <= 0.25:
        support += 0.35
        reasons.append("not_app_only")
    if pair >= 1.0:
        support += 0.30
        reasons.append("pair")
    if pos_pair >= 0.65:
        support += 0.20
        reasons.append("pos_pair")
    if source == "map":
        support += 0.20
        reasons.append("map")
    if score >= 20:
        support += 0.15
        reasons.append("score")
    if texture >= 55 and sky <= 0.03:
        support -= 0.70
        reasons.append("high_texture_no_sky_penalty")
    return support, ";".join(reasons)


def gate_rows(
    rows: list[dict[str, str]],
    clip: str,
    recent_support_ttl: int = 30,
    max_supported_jump_px: float = 70.0,
    risk_threshold: float = 1.0,
    strong_support_threshold: float = 1.0,
    score_floor: float = -1e9,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    recent_support: dict[str, Any] | None = None
    last_frame = -1

    for row in selected_rows(rows, score_floor):
        frame = inum(row.get("frame"))
        if last_frame >= 0 and frame - last_frame > recent_support_ttl:
            recent_support = None
        last_frame = frame

        risk, risk_reason = surface_false_lock_risk(row)
        support, support_reason = target_support_score(row)
        recent_age = None if recent_support is None else frame - int(recent_support["frame"])
        jump = None if recent_support is None else dist_px(row, recent_support["row"])
        has_recent = recent_support is not None and recent_age is not None and 0 <= recent_age <= recent_support_ttl
        plausible_continuity = has_recent and jump is not None and jump <= max_supported_jump_px
        strong_support = support >= strong_support_threshold

        suppress = False
        reason = "accept"
        if risk >= risk_threshold and not strong_support:
            if not plausible_continuity:
                suppress = True
                reason = "surface_risk_without_recent_continuity"
            elif risk >= risk_threshold + 0.65:
                suppress = True
                reason = "surface_risk_overrides_weak_continuity"

        out = {
            "clip": clip,
            "frame": frame,
            "selected_in": 1,
            "selected_out": 0 if suppress else 1,
            "suppressed": int(suppress),
            "reason": reason,
            "risk_score": round(risk, 4),
            "risk_reason": risk_reason,
            "support_score": round(support, 4),
            "support_reason": support_reason,
            "recent_support_frame": "" if recent_support is None else recent_support["frame"],
            "recent_support_age": "" if recent_age is None else recent_age,
            "recent_jump_px": "" if jump is None else round(jump, 3),
            "x": row.get("x", ""),
            "y": row.get("y", ""),
            "w": row.get("w", ""),
            "h": row.get("h", ""),
            "rank": row.get("rank", 1),
            "track_id": row.get("track_id", ""),
            "verified_score": row.get("verified_score", row.get("score", "")),
            "cand_source": row.get("cand_source", ""),
            "cand_texture": row.get("cand_texture", ""),
            "cand_sky_like": row.get("cand_sky_like", ""),
            "tube_appearance_only_rate": row.get("tube_appearance_only_rate", ""),
            "tube_mean_texture": row.get("tube_mean_texture", ""),
            "tube_mean_sky_like": row.get("tube_mean_sky_like", ""),
            "tube_mean_pair_score": row.get("tube_mean_pair_score", ""),
            "tube_positive_pair_rate": row.get("tube_positive_pair_rate", ""),
        }
        decisions.append(out)

        if suppress:
            events.append(out)
            continue

        accepted.append(
            {
                "clip": clip,
                "frame": frame,
                "selected": 1,
                "rank": row.get("rank", 1),
                "x": row.get("x", ""),
                "y": row.get("y", ""),
                "w": row.get("w", 1),
                "h": row.get("h", 1),
                "learned_score": row.get("verified_score", row.get("score", "")),
                "track_id": row.get("track_id", ""),
            }
        )
        if strong_support:
            recent_support = {"frame": frame, "row": row}

    return accepted, decisions, events


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.selected_tubes))
    accepted, decisions, events = gate_rows(
        rows,
        args.clip,
        recent_support_ttl=args.recent_support_ttl,
        max_supported_jump_px=args.max_supported_jump_px,
        risk_threshold=args.risk_threshold,
        strong_support_threshold=args.strong_support_threshold,
        score_floor=args.score_floor,
    )
    write_csv(Path(args.out_csv), accepted, SELECTED_FIELDS)
    if args.decisions_csv:
        write_csv(Path(args.decisions_csv), decisions)
    if args.events_csv:
        write_csv(Path(args.events_csv), events)
    print(
        f"accepted={len(accepted)} suppressed={len(events)} "
        f"input_selected={len(selected_rows(rows, args.score_floor))}"
    )


if __name__ == "__main__":
    main()
