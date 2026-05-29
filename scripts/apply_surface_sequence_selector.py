#!/usr/bin/env python3
"""Apply a learned surface ranker with a continuity/Viterbi selector.

This is an offline/deferred selector for exported ``top_tubes.csv`` rows. It is
meant to test the runtime direction before putting a delayed sliding-window
version into ``tbd_motion_detector.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib

try:
    import evaluate_xy_sequence_ranker as seq
    import sweep_clba_score_adjustment as clba_adjust
    import train_surface_xy_ranker as surface
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import evaluate_xy_sequence_ranker as seq
    from scripts import sweep_clba_score_adjustment as clba_adjust
    from scripts import train_surface_xy_ranker as surface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_tubes", required=True)
    p.add_argument("--model", required=True, help="Surface ranker .joblib bundle.")
    p.add_argument("--clip", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--scored_csv", default="")
    p.add_argument("--max_rank", type=int, default=40)
    p.add_argument("--max_jump_px", type=float, default=12.0)
    p.add_argument("--transition_weight", type=float, default=0.35)
    p.add_argument("--size_jump_weight", type=float, default=0.0)
    p.add_argument(
        "--selector",
        choices=("viterbi", "hmm"),
        default="viterbi",
        help="Selection model. hmm adds explicit absent/coast/null states.",
    )
    p.add_argument(
        "--sequence_window",
        type=int,
        default=0,
        help="Use rolling-window Viterbi over this many frames. 0 keeps the legacy full-video path.",
    )
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument(
        "--acquire_threshold",
        type=float,
        default=None,
        help=(
            "Optional track-acquisition threshold. When set, selections only "
            "start after a candidate reaches this score."
        ),
    )
    p.add_argument(
        "--keep_threshold",
        type=float,
        default=None,
        help="Score threshold for keeping an acquired track. Defaults to --threshold.",
    )
    p.add_argument(
        "--hysteresis_max_jump_px",
        type=float,
        default=None,
        help="Maximum accepted jump while tracking. Defaults to --max_jump_px.",
    )
    p.add_argument(
        "--lost_patience",
        type=int,
        default=0,
        help="Consecutive failed keep frames allowed before returning to acquisition.",
    )
    p.add_argument(
        "--clba_adjustment",
        action="store_true",
        help="Adjust learned_score with CLBA target/background terms before sequence selection.",
    )
    p.add_argument("--clba_gain_weight", type=float, default=0.0)
    p.add_argument("--clba_path_weight", type=float, default=0.0)
    p.add_argument("--clba_target_q_weight", type=float, default=0.0)
    p.add_argument("--clba_bg_weight", type=float, default=0.0)
    p.add_argument("--clba_attached_weight", type=float, default=0.0)
    p.add_argument("--clba_density_weight", type=float, default=0.0)
    p.add_argument("--hmm_beam", type=int, default=128)
    p.add_argument("--hmm_score_mode", choices=("logit", "centered", "raw"), default="logit")
    p.add_argument("--hmm_score_scale", type=float, default=1.0)
    p.add_argument("--hmm_score_center", type=float, default=0.5)
    p.add_argument("--hmm_birth_penalty", type=float, default=1.2)
    p.add_argument("--hmm_track_bonus", type=float, default=0.05)
    p.add_argument("--hmm_miss_penalty", type=float, default=0.65)
    p.add_argument("--hmm_coast_penalty", type=float, default=0.15)
    p.add_argument("--hmm_reacquire_penalty", type=float, default=0.35)
    p.add_argument("--hmm_max_coast", type=int, default=3)
    p.add_argument("--hmm_clutter_weight", type=float, default=0.0)
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


def load_ranked_rows(path: Path, max_rank: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(path):
        rank = surface.int_or_default(row.get("rank"), 999999)
        if rank > max_rank:
            continue
        rows.append(row)
    rows.sort(
        key=lambda r: (
            surface.int_or_default(r.get("frame"), 0),
            surface.int_or_default(r.get("rank"), 999999),
        )
    )
    return rows


def score_rows(rows: list[dict[str, str]], model_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    numeric = list(bundle["numeric_features"])
    sources = list(bundle["source_features"])
    scores = surface.predict_score(model, surface.vectorize(rows, numeric, sources))
    scored: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        out = dict(row)
        out["learned_score"] = float(score)
        scored.append(out)
    meta = {
        "model": str(model_path),
        "numeric_features": numeric,
        "source_features": sources,
        "max_rank_from_model": bundle.get("max_rank", ""),
        "final_exclude_clip": bundle.get("final_exclude_clip", ""),
    }
    return scored, meta


def apply_clba_adjustment(rows: list[dict[str, Any]], weights: clba_adjust.Weights) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        old_score = float(out.get("learned_score", 0.0) or 0.0)
        out["base_learned_score"] = old_score
        # The shared adjustment utility is tolerant of numeric values even
        # though it is usually fed CSV string rows.
        out["learned_score"] = clba_adjust.adjusted_score(out, weights, "learned_score")
        out["clba_adjusted_score"] = out["learned_score"]
        adjusted.append(out)
    return adjusted


def group_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame = surface.int_or_default(row.get("frame"), 0)
        by_frame.setdefault(frame, []).append(row)
    return by_frame


def bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        surface.float_or_default(row.get("x"), 0.0),
        surface.float_or_default(row.get("y"), 0.0),
        surface.float_or_default(row.get("w"), 1.0),
        surface.float_or_default(row.get("h"), 1.0),
    )


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = a[0] + 0.5 * a[2], a[1] + 0.5 * a[3]
    bx, by = b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def score(row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    return float(row.get("learned_score", 0.0) or 0.0)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def logit_score(value: float) -> float:
    p = min(1.0 - 1e-4, max(1e-4, float(value)))
    return math.log(p / (1.0 - p))


def candidate_evidence(
    row: dict[str, Any],
    score_mode: str,
    score_scale: float,
    score_center: float,
    clutter_weight: float,
) -> float:
    """Return log-likelihood-ish evidence for target vs null/clutter.

    The learned score is still the primary observation. Optional clutter terms
    are deliberately conservative: they only subtract when CLBA static/attached
    explanations are stronger than the target likelihood.
    """

    row_score = score(row)
    if score_mode == "logit":
        evidence = score_scale * logit_score(row_score)
    elif score_mode == "centered":
        evidence = score_scale * (row_score - score_center)
    elif score_mode == "raw":
        evidence = score_scale * row_score
    else:
        raise ValueError(f"unknown score mode: {score_mode}")
    if clutter_weight > 0.0:
        target = safe_float(row.get("clba_target_likelihood"))
        static = safe_float(row.get("clba_bg_static_likelihood"))
        attached = safe_float(row.get("clba_attached_likelihood"))
        evidence -= clutter_weight * max(0.0, static - target, attached - target)
    return evidence


def apply_hysteresis_gate(
    selected: dict[int, dict[str, Any]],
    acquire_threshold: float,
    keep_threshold: float,
    max_jump_px: float,
    lost_patience: int = 0,
) -> dict[int, dict[str, Any]]:
    """Gate a selected candidate stream with acquire/keep state.

    This is intentionally simpler than Viterbi: acquisition is score-first, then
    the tracker only keeps boxes that remain plausible by score and frame-to-frame
    jump. It targets the current failure mode where background branches are
    emitted before a real target appears.
    """

    out: dict[int, dict[str, Any]] = {}
    active = False
    lost = 0
    last_frame: int | None = None
    last_row: dict[str, Any] | None = None
    for frame in sorted(selected):
        row = selected[frame]
        row_score = score(row)
        emit = False
        if not active:
            if row_score >= acquire_threshold:
                active = True
                emit = True
                lost = 0
        else:
            gap = max(1, frame - last_frame) if last_frame is not None else 1
            allowed_jump = max_jump_px * gap
            jump_ok = last_row is None or center_dist(bbox(last_row), bbox(row)) <= allowed_jump
            if row_score >= keep_threshold and jump_ok:
                emit = True
                lost = 0
            else:
                lost += 1
                if lost > max(0, lost_patience):
                    active = False
                    last_row = None
                    last_frame = None
        if emit:
            out[frame] = row
            last_row = row
            last_frame = frame
    return out


def select_with_null_hmm(
    by_frame: dict[int, list[dict[str, Any]]],
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float,
    beam: int,
    score_mode: str,
    score_scale: float,
    score_center: float,
    birth_penalty: float,
    track_bonus: float,
    miss_penalty: float,
    coast_penalty: float,
    reacquire_penalty: float,
    max_coast: int,
    clutter_weight: float,
) -> dict[int, dict[str, Any]]:
    """Candidate HMM with explicit absent and coast/null states.

    States:
      A: absent/no target, emits no box.
      T: target acquired, emits a candidate box.
      C: recently lost/coasting from a prior target, emits no box.

    This is intentionally candidate-based, not dense image Viterbi. It tests the
    professor's recommended structure in the existing top-tube harness before we
    move any logic into the runtime detector.
    """

    frames = sorted(by_frame)
    if not frames:
        return {}

    states: list[dict[str, Any]] = [
        {
            "state": "A",
            "score": 0.0,
            "row": None,
            "last_frame": None,
            "misses": 0,
            "selected": {},
        }
    ]
    beam = max(1, int(beam))

    def transition_cost(prev_row: dict[str, Any], row: dict[str, Any], gap: int) -> float | None:
        pb = bbox(prev_row)
        rb = bbox(row)
        size_allowance = size_jump_weight * max(pb[2], pb[3], rb[2], rb[3])
        allowed = max_jump_px * max(1, gap) + size_allowance
        jump = center_dist(pb, rb)
        if jump > allowed:
            return None
        return transition_weight * (jump / max(1e-6, allowed)) ** 2

    for frame in frames:
        rows = by_frame.get(frame, [])
        next_states: list[dict[str, Any]] = []
        for st in states:
            state_name = st["state"]
            base_score = float(st["score"])
            last_row = st.get("row")
            last_frame = st.get("last_frame")
            misses = int(st.get("misses", 0))
            selected = dict(st.get("selected", {}))

            # Emit null/no-box.
            if state_name == "A":
                next_states.append(
                    {
                        "state": "A",
                        "score": base_score,
                        "row": None,
                        "last_frame": None,
                        "misses": 0,
                        "selected": selected,
                    }
                )
            elif state_name == "T":
                next_states.append(
                    {
                        "state": "C",
                        "score": base_score - miss_penalty,
                        "row": last_row,
                        "last_frame": last_frame,
                        "misses": 1,
                        "selected": selected,
                    }
                )
            elif state_name == "C":
                new_misses = misses + 1
                if new_misses <= max_coast:
                    next_states.append(
                        {
                            "state": "C",
                            "score": base_score - coast_penalty,
                            "row": last_row,
                            "last_frame": last_frame,
                            "misses": new_misses,
                            "selected": selected,
                        }
                    )
                next_states.append(
                    {
                        "state": "A",
                        "score": base_score - 0.05 * max(0, new_misses - max_coast),
                        "row": None,
                        "last_frame": None,
                        "misses": 0,
                        "selected": selected,
                    }
                )

            # Emit a real candidate.
            for row in rows:
                obs = candidate_evidence(row, score_mode, score_scale, score_center, clutter_weight)
                new_selected = dict(selected)
                new_selected[frame] = row
                if state_name == "A":
                    next_states.append(
                        {
                            "state": "T",
                            "score": base_score + obs - birth_penalty,
                            "row": row,
                            "last_frame": frame,
                            "misses": 0,
                            "selected": new_selected,
                        }
                    )
                else:
                    if last_row is None or last_frame is None:
                        continue
                    gap = max(1, frame - int(last_frame))
                    cost = transition_cost(last_row, row, gap)
                    if cost is None:
                        continue
                    penalty = reacquire_penalty if state_name == "C" else 0.0
                    next_states.append(
                        {
                            "state": "T",
                            "score": base_score + obs + track_bonus - cost - penalty,
                            "row": row,
                            "last_frame": frame,
                            "misses": 0,
                            "selected": new_selected,
                        }
                    )

        # Collapse equivalent absent/coast states and prune by score. Keeping a
        # few distinct coasting states preserves possible reacquisition anchors.
        next_states.sort(key=lambda s: float(s["score"]), reverse=True)
        deduped: list[dict[str, Any]] = []
        seen_absent = False
        seen_track_keys: set[tuple[str, int, str]] = set()
        for st in next_states:
            if st["state"] == "A":
                if seen_absent:
                    continue
                seen_absent = True
            else:
                row = st.get("row") or {}
                key = (
                    st["state"],
                    int(st.get("last_frame") or -1),
                    str(row.get("track_id", row.get("rank", ""))),
                )
                if key in seen_track_keys:
                    continue
                seen_track_keys.add(key)
            deduped.append(st)
            if len(deduped) >= beam:
                break
        states = deduped

    best = max(states, key=lambda s: float(s["score"]))
    return dict(best.get("selected", {}))


def output_rows(
    clip: str,
    scored_rows: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    frames = sorted({surface.int_or_default(row.get("frame"), 0) for row in scored_rows})
    rows: list[dict[str, Any]] = []
    for frame in frames:
        row = selected.get(frame)
        score = float(row.get("learned_score", 0.0) or 0.0) if row is not None else 0.0
        is_selected = row is not None and score >= threshold
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "selected": int(is_selected),
                "rank": surface.int_or_default(row.get("rank"), 0) if is_selected else "",
                "learned_score": round(score, 6) if row is not None else "",
                "threshold": threshold,
                "x": row.get("x", "") if is_selected else "",
                "y": row.get("y", "") if is_selected else "",
                "w": row.get("w", "") if is_selected else "",
                "h": row.get("h", "") if is_selected else "",
                "verified_score": row.get("verified_score", "") if is_selected else "",
                "source": row.get("cand_source", "") if is_selected else "",
                "track_id": row.get("track_id", "") if is_selected else "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = load_ranked_rows(Path(args.top_tubes), args.max_rank)
    if not rows:
        raise SystemExit("no top-tube rows loaded")
    scored, meta = score_rows(rows, Path(args.model))
    clba_weights = clba_adjust.Weights(
        gain=args.clba_gain_weight,
        path=args.clba_path_weight,
        target_q=args.clba_target_q_weight,
        bg=args.clba_bg_weight,
        attached=args.clba_attached_weight,
        density=args.clba_density_weight,
    )
    if args.clba_adjustment:
        scored = apply_clba_adjustment(scored, clba_weights)
    by_frame = group_by_frame(scored)
    if args.selector == "hmm":
        selected = select_with_null_hmm(
            by_frame,
            max_jump_px=args.max_jump_px,
            transition_weight=args.transition_weight,
            size_jump_weight=args.size_jump_weight,
            beam=args.hmm_beam,
            score_mode=args.hmm_score_mode,
            score_scale=args.hmm_score_scale,
            score_center=args.hmm_score_center,
            birth_penalty=args.hmm_birth_penalty,
            track_bonus=args.hmm_track_bonus,
            miss_penalty=args.hmm_miss_penalty,
            coast_penalty=args.hmm_coast_penalty,
            reacquire_penalty=args.hmm_reacquire_penalty,
            max_coast=args.hmm_max_coast,
            clutter_weight=args.hmm_clutter_weight,
        )
    elif args.sequence_window > 0:
        selected = seq.rolling_viterbi_select(
            by_frame,
            max_jump_px=args.max_jump_px,
            transition_weight=args.transition_weight,
            size_jump_weight=args.size_jump_weight,
            sequence_window=args.sequence_window,
        )
    else:
        selected = seq.viterbi_select(
            by_frame,
            max_jump_px=args.max_jump_px,
            transition_weight=args.transition_weight,
            size_jump_weight=args.size_jump_weight,
        )
    if args.acquire_threshold is not None:
        selected = apply_hysteresis_gate(
            selected,
            acquire_threshold=args.acquire_threshold,
            keep_threshold=args.keep_threshold if args.keep_threshold is not None else args.threshold,
            max_jump_px=args.hysteresis_max_jump_px if args.hysteresis_max_jump_px is not None else args.max_jump_px,
            lost_patience=args.lost_patience,
        )
    out_rows = output_rows(args.clip, scored, selected, args.threshold)
    out_path = Path(args.out_csv)
    write_csv(out_path, out_rows)
    if args.scored_csv:
        write_csv(Path(args.scored_csv), scored)
    summary = {
        "top_tubes": args.top_tubes,
        "output": str(out_path),
        "clip": args.clip,
        "rows": len(rows),
        "frames": len(out_rows),
        "selected_frames": sum(int(r["selected"]) for r in out_rows),
        "selector": args.selector,
        "max_rank": args.max_rank,
        "max_jump_px": args.max_jump_px,
        "transition_weight": args.transition_weight,
        "size_jump_weight": args.size_jump_weight,
        "sequence_window": args.sequence_window,
        "threshold": args.threshold,
        "acquire_threshold": args.acquire_threshold,
        "keep_threshold": args.keep_threshold,
        "hysteresis_max_jump_px": args.hysteresis_max_jump_px,
        "lost_patience": args.lost_patience,
        "clba_adjustment": int(args.clba_adjustment),
        "clba_weights": clba_weights.__dict__,
        "hmm": {
            "beam": args.hmm_beam,
            "score_mode": args.hmm_score_mode,
            "score_scale": args.hmm_score_scale,
            "score_center": args.hmm_score_center,
            "birth_penalty": args.hmm_birth_penalty,
            "track_bonus": args.hmm_track_bonus,
            "miss_penalty": args.hmm_miss_penalty,
            "coast_penalty": args.hmm_coast_penalty,
            "reacquire_penalty": args.hmm_reacquire_penalty,
            "max_coast": args.hmm_max_coast,
            "clutter_weight": args.hmm_clutter_weight,
        },
        **meta,
    }
    (out_path.parent / "sequence_selector_summary.json").write_text(json.dumps(summary, indent=2))
    print(out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
