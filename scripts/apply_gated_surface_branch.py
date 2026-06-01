#!/usr/bin/env python3
"""Apply a narrow gated surface rescue branch to candidate CSVs.

This is an offline production-shape harness. It combines the normal scored
top-tube candidates with a separately scored surface-halo/recenter branch, but
only on frames where the normal JS1 trace says the tracker is low-confidence in
surface-like context.

It does not change detector/runtime defaults.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.selector_core import (
    surface_gate_low_confidence,
    surface_rescue_risk,
    trace_router_bucket as shared_trace_router_bucket,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_candidates", required=True)
    p.add_argument("--surface_candidates", required=True)
    p.add_argument("--state_trace", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--gate_report_csv", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--max_base_rank", type=int, default=30)
    p.add_argument(
        "--base_keep_when_gated",
        type=int,
        default=3,
        help="Keep this many normal/base candidates on gated frames so the rescue branch competes instead of replacing.",
    )
    p.add_argument(
        "--surface_top_per_frame",
        type=int,
        default=5,
        help="Maximum recentered/surface branch candidates to add on a gated frame.",
    )
    p.add_argument("--gate_states", default="A,P,C,S,E")
    p.add_argument("--gate_routers", default="surface,line,boundary,unknown")
    p.add_argument("--gate_rank_min", type=int, default=10)
    p.add_argument("--gate_margin_max", type=float, default=0.8)
    p.add_argument("--gate_raw_score_max", type=float, default=-999.0)
    p.add_argument(
        "--gate_low_conf_frames",
        type=int,
        default=2,
        help="Require this many consecutive low-confidence trace frames before enabling surface rescue.",
    )
    p.add_argument(
        "--surface_risk_min",
        type=float,
        default=1.0,
        help="Minimum local surface-risk score required before enabling surface rescue.",
    )
    p.add_argument(
        "--gate_hold_frames",
        type=int,
        default=0,
        help=(
            "Keep the surface branch eligible for this many frames after an active gate, "
            "as long as local surface risk remains high. Default 0 preserves the strict gate."
        ),
    )
    p.add_argument(
        "--gate_hold_risk_min",
        type=float,
        default=0.5,
        help="Minimum local surface-risk score required while a post-gate hold is active.",
    )
    p.add_argument("--surface_score_min", type=float, default=-1.0)
    p.add_argument("--surface_as_learned_logits", action="store_true")
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


def fnum(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def score_probability(value: Any, default: float = 0.0) -> float:
    """Convert a candidate score/logit-ish value into a bounded probability.

    Surface-halo ranker outputs are already probabilities, but several bounded
    proposal exports only have detector ``score``. Treat values outside [0, 1]
    as logits so downstream gates can compare all surface branches through the
    same ``surface_halo_score`` field without giving raw detector scores a free
    pass.
    """

    raw = fnum(value, None)
    if raw is None:
        return default
    if 0.0 <= raw <= 1.0:
        return raw
    if raw >= 40.0:
        return 1.0 - 1e-9
    if raw <= -40.0:
        return 1e-9
    return 1.0 / (1.0 + math.exp(-raw))


def score_logit(probability: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, probability))
    return math.log(p / (1.0 - p))


def fint(value: Any, default: int = 0) -> int:
    out = fnum(value)
    return default if out is None else int(round(out))


def parse_set(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def trace_router_bucket(row: dict[str, str]) -> str:
    return shared_trace_router_bucket(row)


def should_gate(
    trace: dict[str, str] | None,
    gate_states: set[str],
    gate_routers: set[str],
    gate_rank_min: int,
    gate_margin_max: float,
    gate_raw_score_max: float,
) -> tuple[bool, str]:
    return surface_gate_low_confidence(
        trace,
        gate_states=gate_states,
        gate_routers=gate_routers,
        gate_rank_min=gate_rank_min,
        gate_margin_max=gate_margin_max,
        gate_raw_score_max=gate_raw_score_max,
    )


def max_row_feature(rows: list[dict[str, str]], names: tuple[str, ...], default: float = 0.0) -> float:
    best = default
    for row in rows:
        for name in names:
            value = fnum(row.get(name), None)
            if value is not None:
                best = max(best, value)
    return best


def local_surface_risk(
    trace: dict[str, str] | None,
    base_rows: list[dict[str, str]],
    surface_rows: list[dict[str, str]],
) -> tuple[float, str]:
    """Return a cheap local risk score for when surface rescue is justified.

    The rescue branch is meant for hard surface/edge ambiguity, not for every
    frame whose global router says "surface". This score intentionally uses
    only already-exported trace/candidate fields, so the gate can run as an
    offline production-shape merge step without touching proposal generation.
    """

    return surface_rescue_risk(trace, base_rows, surface_rows)


def bucket_by_frame(rows: list[dict[str, str]], clip: str, max_rank: int) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if clip and row.get("clip", clip) != clip:
            continue
        rank = fint(row.get("rank"), 999999)
        frame = fint(row.get("frame"), -1)
        if frame >= 0 and rank <= max_rank:
            out[frame].append(row)
    for frame_rows in out.values():
        frame_rows.sort(key=lambda r: fint(r.get("rank"), 999999))
    return out


def adapt_surface_row(row: dict[str, str], as_learned_logits: bool) -> dict[str, Any]:
    rec: dict[str, Any] = dict(row)
    rec["gated_surface_branch"] = "1"
    rec["gated_surface_parent_rank"] = rec.get("surface_halo_parent_rank", rec.get("rank", ""))
    if not rec.get("surface_halo_score"):
        score_source = rec.get("crop_t_prob", rec.get("score", "0"))
        rec["surface_halo_score"] = f"{score_probability(score_source):.6f}"
    if not rec.get("surface_halo_logit"):
        rec["surface_halo_logit"] = f"{score_logit(score_probability(rec.get('surface_halo_score'))):.6f}"
    if as_learned_logits:
        score = score_probability(rec.get("surface_halo_score"))
        logit = fnum(rec.get("surface_halo_logit"), None)
        if logit is None:
            logit = score_logit(score)
        rec["crop_t_prob"] = f"{score:.6f}"
        rec["crop_t_logit"] = f"{logit:.6f}"
        rec["crop_g_prob"] = f"{1.0 - min(1.0, max(0.0, score)):.6f}"
        rec["crop_g_logit"] = f"{-logit:.6f}"
        rec.setdefault("crop_s_logit", "-3.000000")
        rec.setdefault("crop_e_logit", "-3.000000")
        rec.setdefault("crop_h_logit", "-3.000000")
    return rec


def annotate_gate_context(
    row: dict[str, Any],
    trace: dict[str, str] | None,
    *,
    low_confidence: bool,
    low_conf_streak: int,
    surface_risk_score: float,
    surface_risk_reason: str,
    gate_reason: str,
) -> None:
    """Attach routing telemetry without making it selector evidence.

    These fields are deliberately namespaced away from the observation columns
    consumed by JS1/rankers. They let us audit why a recentered branch was
    active and train future routers from base-selector context without leaking
    that context into target identity scores.
    """

    row["gate_reason"] = gate_reason
    row["gate_low_confidence"] = int(low_confidence)
    row["gate_low_conf_streak"] = int(low_conf_streak)
    row["gate_surface_risk_score"] = round(float(surface_risk_score), 4)
    row["gate_surface_risk_reason"] = surface_risk_reason
    row["base_trace_state"] = "" if trace is None else trace.get("state", "")
    row["base_trace_selected"] = "" if trace is None else trace.get("selected", "")
    row["base_trace_rank"] = "" if trace is None else trace.get("rank", "")
    row["base_trace_target_margin"] = "" if trace is None else trace.get("target_margin", "")
    row["base_trace_router_bucket"] = "" if trace is None else trace.get("router_bucket", "")
    row["base_trace_raw_score"] = "" if trace is None else trace.get("raw_score", "")


def merge_candidates(
    base_by_frame: dict[int, list[dict[str, str]]],
    surface_by_frame: dict[int, list[dict[str, str]]],
    trace_by_frame: dict[int, dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = sorted(set(base_by_frame) | set(surface_by_frame) | set(trace_by_frame))
    out: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    gate_states = parse_set(args.gate_states)
    gate_routers = parse_set(args.gate_routers)
    low_conf_streak = 0
    hold_remaining = 0
    prev_frame: int | None = None

    for frame in frames:
        contiguous = prev_frame is not None and frame == prev_frame + 1
        if not contiguous:
            hold_remaining = 0
        base_rows = base_by_frame.get(frame, [])
        surface_rows = [
            row for row in surface_by_frame.get(frame, [])
            if score_probability(row.get("surface_halo_score", row.get("crop_t_prob", row.get("score", "0")))) >= args.surface_score_min
        ]
        trace = trace_by_frame.get(frame)
        low_conf, low_conf_reason = should_gate(
            trace,
            gate_states,
            gate_routers,
            args.gate_rank_min,
            args.gate_margin_max,
            args.gate_raw_score_max,
        )
        if low_conf and contiguous:
            low_conf_streak += 1
        elif low_conf:
            low_conf_streak = 1
        else:
            low_conf_streak = 0
        prev_frame = frame

        risk_score, risk_reason = local_surface_risk(trace, base_rows, surface_rows)
        repeated = low_conf_streak >= max(1, int(args.gate_low_conf_frames))
        risky = risk_score >= float(args.surface_risk_min)
        hold_eligible = (
            (not low_conf)
            and hold_remaining > 0
            and risk_score >= float(args.gate_hold_risk_min)
            and bool(surface_rows)
        )
        gated = bool((low_conf and repeated and risky) or hold_eligible)
        if not low_conf:
            if hold_eligible:
                reason = (
                    f"hold_after_gate_{hold_remaining}:surface_risk_{risk_score:.2f}"
                    f"_ge_{args.gate_hold_risk_min:g}:{risk_reason}"
                )
            else:
                reason = low_conf_reason
        elif not repeated:
            reason = f"low_conf_streak_{low_conf_streak}_lt_{max(1, int(args.gate_low_conf_frames))}:{low_conf_reason}"
        elif not risky:
            reason = f"surface_risk_{risk_score:.2f}_lt_{args.surface_risk_min:g}:{risk_reason}"
        else:
            reason = f"{low_conf_reason}|surface_risk_{risk_score:.2f}:{risk_reason}"

        if low_conf and repeated and risky:
            hold_remaining = int(args.gate_hold_frames)
        elif hold_eligible:
            hold_remaining = max(0, hold_remaining - 1)
        else:
            hold_remaining = max(0, hold_remaining - 1)

        frame_rows: list[dict[str, Any]] = []
        if gated and surface_rows:
            for row in surface_rows[: args.surface_top_per_frame]:
                frame_rows.append(adapt_surface_row(row, args.surface_as_learned_logits))
            for row in base_rows[: max(0, args.base_keep_when_gated)]:
                rec: dict[str, Any] = dict(row)
                rec["gated_surface_branch"] = "0"
                frame_rows.append(rec)
        else:
            for row in base_rows:
                rec = dict(row)
                rec["gated_surface_branch"] = "0"
                frame_rows.append(rec)

        for idx, row in enumerate(frame_rows, start=1):
            row["rank"] = str(idx)
            row["gate_active"] = "1" if gated and surface_rows else "0"
            annotate_gate_context(
                row,
                trace,
                low_confidence=low_conf,
                low_conf_streak=low_conf_streak,
                surface_risk_score=risk_score,
                surface_risk_reason=risk_reason,
                gate_reason=reason,
            )
            out.append(row)

        report.append(
            {
                "frame": frame,
                "gate_active": int(gated and bool(surface_rows)),
                "gate_reason": reason,
                "trace_state": "" if trace is None else trace.get("state", ""),
                "trace_selected": "" if trace is None else trace.get("selected", ""),
                "trace_rank": "" if trace is None else trace.get("rank", ""),
                "trace_target_margin": "" if trace is None else trace.get("target_margin", ""),
                "trace_router_bucket": "" if trace is None else trace.get("router_bucket", ""),
                "low_confidence": int(low_conf),
                "low_conf_streak": low_conf_streak,
                "surface_risk_score": round(risk_score, 4),
                "surface_risk_reason": risk_reason,
                "base_rows": len(base_rows),
                "surface_rows": len(surface_rows),
                "output_rows": len(frame_rows),
            }
        )

    return out, report


def main() -> None:
    args = parse_args()
    base_by_frame = bucket_by_frame(read_csv(Path(args.base_candidates)), args.clip, args.max_base_rank)
    surface_by_frame = bucket_by_frame(read_csv(Path(args.surface_candidates)), args.clip, 999999)
    trace_by_frame = {
        fint(row.get("frame"), -1): row
        for row in read_csv(Path(args.state_trace))
        if fint(row.get("frame"), -1) >= 0 and (not args.clip or row.get("clip", args.clip) == args.clip)
    }
    out, report = merge_candidates(base_by_frame, surface_by_frame, trace_by_frame, args)
    write_csv(Path(args.out_csv), out)
    write_csv(Path(args.gate_report_csv), report)
    active = sum(int(row["gate_active"]) for row in report)
    print(f"{Path(args.out_csv)}")
    print(f"gated_frames={active}/{len(report)}")


if __name__ == "__main__":
    main()
