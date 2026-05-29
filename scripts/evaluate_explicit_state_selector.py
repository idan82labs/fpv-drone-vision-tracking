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
    p.add_argument("--quarantine_px", type=float, default=14.0)
    p.add_argument("--quarantine_frames", type=int, default=18)
    p.add_argument("--score_weight", type=float, default=0.75)
    p.add_argument("--clba_weight", type=float, default=0.55)
    p.add_argument("--path_weight", type=float, default=0.25)
    p.add_argument("--static_weight", type=float, default=0.7)
    p.add_argument("--attached_weight", type=float, default=0.7)
    p.add_argument("--rank_weight", type=float, default=0.12)
    p.add_argument("--motion_weight", type=float, default=0.8)
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
    rank = int(fnum(row.get("rank"), 999999))
    raw_score = fnum(row.get(score_column), fnum(row.get("score")))
    if 0.0 <= raw_score <= 1.0:
        proposal = score_weight * math.log(max(1e-4, raw_score) / max(1e-4, 1.0 - raw_score))
    else:
        proposal = score_weight * math.log1p(max(0.0, raw_score))
    proposal -= rank_weight * math.log1p(max(1, rank))

    clba_gain = fnum(row.get("clba_gain_norm"), fnum(row.get("tube_mean_align_gain")))
    target_q = fnum(row.get("clba_target_q"), fnum(row.get("tube_mean_native_dark_score")))
    bg_q = fnum(row.get("clba_bg_q"))
    path_dist = fnum(row.get("clba_path_bg_dist_mean"), fnum(row.get("tube_mean_bg_dist")))
    line = max_feature(row, "cand_line_context", "tube_mean_line_context")
    support = max_feature(row, "cand_attached_support", "tube_mean_attached_support")
    density = fnum(row.get("tube_log_cand_density"), math.log1p(max_feature(row, "tube_mean_cand_density")))
    pair = fnum(row.get("tube_mean_pair_bg"), fnum(row.get("tube_mean_pair_score")))

    target_obs = (
        proposal
        + clba_weight * clba_gain
        + 0.18 * max(-2.0, min(4.0, target_q))
        + path_weight * min(2.0, path_dist / 8.0)
        + 0.25 * pair
    )
    static_obs = (
        static_weight * max(0.0, bg_q)
        + 0.55 * max(0.0, 1.0 - min(path_dist, 6.0) / 6.0)
        + 0.08 * density
        + 0.12 * line
    )
    attached_obs = (
        attached_weight * (0.9 * line + 0.045 * support)
        + 0.08 * density
        - 0.15 * max(0.0, clba_gain)
    )
    return target_obs, static_obs, attached_obs, raw_score


def load_candidates(
    path: Path,
    clip: str,
    max_rank: int,
    score_column: str,
    args: argparse.Namespace,
) -> dict[int, list[Candidate]]:
    out: dict[int, list[Candidate]] = {}
    for row in read_csv(path):
        if clip and row.get("clip", clip) != clip:
            continue
        frame = int(fnum(row.get("frame"), -1))
        rank = int(fnum(row.get("rank"), 999999))
        if frame < 0 or rank > max_rank:
            continue
        target_obs, static_obs, attached_obs, raw_score = candidate_observations(
            row,
            score_column,
            args.score_weight,
            args.clba_weight,
            args.path_weight,
            args.static_weight,
            args.attached_weight,
            args.rank_weight,
        )
        out.setdefault(frame, []).append(
            Candidate(
                frame=frame,
                rank=rank,
                bbox=row_bbox(row),
                target_obs=target_obs,
                static_obs=static_obs,
                attached_obs=attached_obs,
                raw_score=raw_score,
                row=row,
            )
        )
    for rows in out.values():
        rows.sort(key=lambda c: c.rank)
    return out


def motion_cost(prev: Hypothesis, cand: Candidate, max_jump_px: float, motion_weight: float) -> float:
    if prev.bbox is None:
        return 0.0
    pred = prev.bbox
    if prev.state in {"T", "C", "P"}:
        x, y, w, h = prev.bbox
        pred = (x + prev.vx, y + prev.vy, w, h)
    jump = center_dist(pred, cand.bbox)
    allowed = max_jump_px + 0.7 * max(prev.bbox[2], prev.bbox[3], cand.bbox[2], cand.bbox[3])
    if jump > 3.0 * allowed:
        return 1e6
    return motion_weight * (jump / max(1.0, allowed)) ** 2


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
    quarantine_frames: int,
    motion_weight: float,
    beam_width: int,
    state_beam: int,
) -> list[Hypothesis]:
    pool: list[Hypothesis] = []
    for hyp in hyps:
        q_age = max(0, hyp.lock_age - 1)
        q_bbox = hyp.quarantine_bbox if q_age > 0 else None
        # Null / absent continuation.
        if hyp.state in {"A", "S", "E"}:
            add_pruned(
                pool,
                Hypothesis("A", hyp.score + 0.05, quarantine_bbox=q_bbox, lock_age=q_age, reason="null"),
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
            if in_quarantine(hyp, cand, quarantine_px):
                continue
            clutter_obs = max(cand.static_obs, cand.attached_obs)
            clutter_state = "E" if cand.attached_obs >= cand.static_obs else "S"
            clutter_edge = clutter_obs - cand.target_obs
            target_llr = cand.target_obs - logsumexp([cand.static_obs, cand.attached_obs, 0.0])
            mcost = motion_cost(hyp, cand, max_jump_px, motion_weight) if hyp.state in {"P", "T", "C"} else 0.0
            if mcost >= 1e5:
                continue

            if clutter_edge >= clutter_margin:
                add_pruned(
                    pool,
                    Hypothesis(
                        clutter_state,
                        hyp.score - 0.05 + 0.05 * min(1.0, clutter_edge),
                        bbox=cand.bbox,
                        lock_age=quarantine_frames,
                        quarantine_bbox=cand.bbox,
                        reason="clutter_lock",
                    ),
                    beam_width,
                )

            if target_llr < acquire_threshold and hyp.state in {"A", "S", "E"}:
                continue
            if hyp.state == "A" or hyp.state in {"S", "E"}:
                hits = 1 if target_llr >= acquire_threshold else 0
                new_state = "T" if hits >= acquire_hits else "P"
                add_pruned(
                    pool,
                    Hypothesis(
                        new_state,
                        hyp.score + target_llr - 0.35,
                        bbox=cand.bbox,
                        hits=hits,
                        age=1,
                        selected=cand if new_state == "T" else None,
                        reason="birth" if new_state == "P" else "birth_track",
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
    quarantine_frames: int,
    motion_weight: float,
    beam_width: int,
    state_beam: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hyps = [Hypothesis("A", 0.0)]
    rows: list[dict[str, Any]] = []
    first_visible = next((f for f in sorted(labels) if labels[f].visible), None)
    first_strict = None
    first_lock = None
    for frame in sorted(labels):
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
            quarantine_frames,
            motion_weight,
            beam_width,
            state_beam,
        )
        best = hyps[0]
        lab = labels[frame]
        selected = best.state == "T" and best.selected is not None
        selected_bbox = best.selected.bbox if selected and best.selected is not None else None
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
                "visible": int(lab.visible),
                "state": best.state,
                "selected": int(selected),
                "rank": "" if best.selected is None else best.selected.rank,
                "target_obs": "" if best.selected is None else round(best.selected.target_obs, 6),
                "static_obs": "" if best.selected is None else round(best.selected.static_obs, 6),
                "attached_obs": "" if best.selected is None else round(best.selected.attached_obs, 6),
                "raw_score": "" if best.selected is None else round(best.selected.raw_score, 6),
                "reason": best.reason,
                "dist_px": dist,
                "strict_hit": strict_hit,
                "loose_hit": loose_hit,
                "correct_all_frame": strict_hit if lab.visible else not selected,
            }
        )

    visible_rows = [r for r in rows if r["visible"]]
    invisible_rows = [r for r in rows if not r["visible"]]
    visible_strict = sum(bool(r["strict_hit"]) for r in visible_rows)
    visible_loose = sum(bool(r["loose_hit"]) for r in visible_rows)
    invisible_no_box = sum(not bool(r["selected"]) for r in invisible_rows)
    correct = sum(bool(r["correct_all_frame"]) for r in rows)
    summary = {
        "acquire_threshold": acquire_threshold,
        "track_threshold": track_threshold,
        "acquire_hits": acquire_hits,
        "max_misses": max_misses,
        "max_jump_px": max_jump_px,
        "clutter_margin": clutter_margin,
        "frames_all": len(rows),
        "visible_frames": len(visible_rows),
        "invisible_frames": len(invisible_rows),
        "all_frame_correct": correct,
        "all_frame_accuracy": round(correct / max(1, len(rows)), 4),
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
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels), args.clip)
    candidates = load_candidates(Path(args.candidates), args.clip, args.max_rank, args.score_column, args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    best_key = (-1.0, -1.0, -1.0, 0)
    for aq in parse_float_list(args.acquire_thresholds):
        for tr in parse_float_list(args.track_thresholds):
            if tr > aq:
                continue
            for hits in parse_int_list(args.acquire_hits):
                for misses in parse_int_list(args.max_misses):
                    for jump in parse_float_list(args.max_jump_px):
                        for margin in parse_float_list(args.clutter_margin):
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
                                args.quarantine_frames,
                                args.motion_weight,
                                args.beam_width,
                                args.state_beam,
                            )
                            summaries.append(summary)
                            key = (
                                summary["all_frame_accuracy"],
                                summary["visible_strict_recall"],
                                summary["invisible_no_box_rate"],
                                -summary["selected_frames"],
                            )
                            if key > best_key:
                                best_key = key
                                best_rows = rows
    summaries.sort(
        key=lambda r: (
            r["all_frame_accuracy"],
            r["visible_strict_recall"],
            r["invisible_no_box_rate"],
            -r["selected_frames"],
        ),
        reverse=True,
    )
    write_csv(out_dir / "explicit_state_sweep.csv", summaries)
    write_csv(out_dir / "best_frame_predictions.csv", best_rows)
    (out_dir / "best_config.json").write_text(json.dumps(summaries[0] if summaries else {}, indent=2))
    metadata = {
        "labels": args.labels,
        "candidates": args.candidates,
        "clip": args.clip,
        "score_column": args.score_column,
        "max_rank": args.max_rank,
        "states": STATES,
        "candidate_frames": len(candidates),
        "label_frames": len(labels),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(out_dir / "explicit_state_sweep.csv")
    if summaries:
        print(json.dumps(summaries[0], indent=2))


if __name__ == "__main__":
    main()
