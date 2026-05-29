#!/usr/bin/env python3
"""Batch-evaluate learned surface selector modes across labeled clips."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import apply_surface_sequence_selector as selector
    import evaluate_tracking_run as tracking_eval
    import evaluate_xy_sequence_ranker as seq
    import train_surface_xy_ranker as surface
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import apply_surface_sequence_selector as selector
    from scripts import evaluate_tracking_run as tracking_eval
    from scripts import evaluate_xy_sequence_ranker as seq
    from scripts import train_surface_xy_ranker as surface


DEFAULT_MODES = (
    "score084:window=1,threshold=0.84;"
    "hyst090080:window=1,acquire=0.90,keep=0.80,jump=24,lost=0;"
    "hyst086086:window=1,acquire=0.86,keep=0.86,jump=24,lost=5"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rank", type=int, default=20)
    p.add_argument("--max_jump_px", type=float, default=12.0)
    p.add_argument("--transition_weight", type=float, default=0.35)
    p.add_argument("--size_jump_weight", type=float, default=0.0)
    p.add_argument("--strict_tol_px", type=float, default=8.0)
    p.add_argument("--loose_tol_px", type=float, default=16.0)
    p.add_argument(
        "--modes",
        default=DEFAULT_MODES,
        help=(
            "Semicolon-separated selector modes. Example: "
            "'score084:window=1,threshold=0.84;hyst:window=1,acquire=0.9,keep=0.8,jump=24,lost=0'."
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


def parse_modes(raw: str) -> list[dict[str, Any]]:
    modes: list[dict[str, Any]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"invalid mode, missing ':': {chunk}")
        name, opts_raw = chunk.split(":", 1)
        opts: dict[str, Any] = {"name": name}
        for part in opts_raw.split(","):
            if not part.strip():
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"window", "lost", "hmm_beam", "max_coast", "acquire_hits"}:
                opts[key] = int(float(value))
            elif key in {"selector", "score_mode"}:
                opts[key] = value
            else:
                opts[key] = float(value)
        modes.append(opts)
    if not modes:
        raise SystemExit("no modes parsed")
    return modes


def labels_by_clip(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        clip = row.get("clip", "")
        frame = tracking_eval.fnum(row.get("frame"))
        if clip and frame is not None:
            grouped[clip].append(row)
    return dict(sorted(grouped.items()))


def top_tubes_path(results_dir: Path, clip: str) -> Path | None:
    direct = results_dir / clip / "top_tubes.csv"
    if direct.exists():
        return direct
    for path in results_dir.glob("*/top_tubes.csv"):
        if surface.clip_matches(clip, path.parent.name):
            return path
    return None


def select_mode(
    by_frame: dict[int, list[dict[str, Any]]],
    mode: dict[str, Any],
    max_jump_px: float,
    transition_weight: float,
    size_jump_weight: float,
) -> dict[int, dict[str, Any]]:
    selector_name = str(mode.get("selector", "viterbi"))
    if selector_name == "hmm":
        return selector.select_with_null_hmm(
            by_frame,
            max_jump_px=float(mode.get("jump", max_jump_px)),
            transition_weight=float(mode.get("transition", transition_weight)),
            size_jump_weight=float(mode.get("size_jump", size_jump_weight)),
            beam=int(mode.get("hmm_beam", 128)),
            score_mode=str(mode.get("score_mode", "logit")),
            score_scale=float(mode.get("score_scale", 1.0)),
            score_center=float(mode.get("score_center", 0.5)),
            birth_penalty=float(mode.get("birth", 1.2)),
            track_bonus=float(mode.get("track_bonus", 0.05)),
            miss_penalty=float(mode.get("miss", 0.65)),
            coast_penalty=float(mode.get("coast", 0.15)),
            reacquire_penalty=float(mode.get("reacquire", 0.35)),
            max_coast=int(mode.get("max_coast", 3)),
            clutter_weight=float(mode.get("clutter", 0.0)),
        )
    if selector_name == "joint_hmm":
        return selector.select_with_joint_hmm(
            by_frame,
            max_jump_px=float(mode.get("jump", max_jump_px)),
            transition_weight=float(mode.get("transition", transition_weight)),
            size_jump_weight=float(mode.get("size_jump", size_jump_weight)),
            beam=int(mode.get("hmm_beam", 96)),
            score_mode=str(mode.get("score_mode", "logit")),
            score_scale=float(mode.get("score_scale", 1.0)),
            score_center=float(mode.get("score_center", 0.5)),
            birth_penalty=float(mode.get("birth", 0.3)),
            track_bonus=float(mode.get("track_bonus", 0.15)),
            miss_penalty=float(mode.get("miss", 0.4)),
            coast_penalty=float(mode.get("coast", 0.1)),
            reacquire_penalty=float(mode.get("reacquire", 0.2)),
            max_coast=int(mode.get("max_coast", 1)),
            acquire_hits=int(mode.get("acquire_hits", 2)),
            target_weight=float(mode.get("target_w", 0.35)),
            gain_weight=float(mode.get("gain_w", 0.55)),
            path_weight=float(mode.get("path_w", 0.03)),
            static_weight=float(mode.get("static_w", 0.75)),
            attached_weight=float(mode.get("attached_w", 0.7)),
            rank_weight=float(mode.get("rank_w", 0.08)),
            null_bias=float(mode.get("null_bias", 0.0)),
            static_bias=float(mode.get("static_bias", 0.15)),
            attached_bias=float(mode.get("attached_bias", 0.1)),
            lock_margin=float(mode.get("lock_margin", 0.1)),
            lock_penalty=float(mode.get("lock_penalty", 0.25)),
            release_penalty=float(mode.get("release", 0.1)),
            quarantine_px=float(mode.get("quarantine_px", 12.0)),
            quarantine_frames=int(mode.get("quarantine_frames", 18)),
            quarantine_penalty=float(mode.get("quarantine_penalty", 2.5)),
        )

    window = int(mode.get("window", 1))
    if window > 0:
        selected = seq.rolling_viterbi_select(
            by_frame,
            max_jump_px=max_jump_px,
            transition_weight=transition_weight,
            size_jump_weight=size_jump_weight,
            sequence_window=window,
        )
    else:
        selected = seq.viterbi_select(
            by_frame,
            max_jump_px=max_jump_px,
            transition_weight=transition_weight,
            size_jump_weight=size_jump_weight,
        )
    if "acquire" in mode:
        selected = selector.apply_hysteresis_gate(
            selected,
            acquire_threshold=float(mode["acquire"]),
            keep_threshold=float(mode.get("keep", mode.get("threshold", 0.0))),
            max_jump_px=float(mode.get("jump", max_jump_px)),
            lost_patience=int(mode.get("lost", 0)),
        )
    return selected


def evaluate_rows(
    labels: list[dict[str, str]],
    selected_rows: list[dict[str, Any]],
    strict_tol: float,
    loose_tol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = {
        int(float(str(r["frame"]))): {k: str(v) for k, v in r.items()}
        for r in selected_rows
        if r.get("frame") not in (None, "") and tracking_eval.row_is_selected({k: str(v) for k, v in r.items()})
    }
    frame_rows: list[dict[str, Any]] = []
    visible_rows = 0
    strict_hits = 0
    loose_hits = 0
    invisible_rows = 0
    invisible_no_box = 0
    for lab in labels:
        frame_val = tracking_eval.fnum(lab.get("frame"))
        if frame_val is None:
            continue
        frame = int(frame_val)
        is_visible = tracking_eval.visible(lab)
        sel = selected.get(frame)
        sel_box = tracking_eval.row_bbox(sel) if sel is not None else None
        lab_box = tracking_eval.label_bbox(lab) if is_visible else None
        dist = None
        strict = False
        loose = False
        if lab_box is not None and sel_box is not None:
            dist = tracking_eval.center_dist(sel_box, lab_box)
            strict = dist <= strict_tol
            loose = dist <= loose_tol
        if is_visible:
            visible_rows += 1
            strict_hits += int(strict)
            loose_hits += int(loose)
        else:
            invisible_rows += 1
            invisible_no_box += int(sel is None)
        frame_rows.append(
            {
                "frame": frame,
                "visible": int(is_visible),
                "selected": int(sel is not None),
                "strict_hit": int(strict),
                "loose_hit": int(loose),
                "dist_px": "" if dist is None else round(dist, 3),
            }
        )
    summary = {
        "label_frames": len(frame_rows),
        "visible_frames": visible_rows,
        "strict_hits": strict_hits,
        "strict_recall": round(strict_hits / max(1, visible_rows), 4),
        "loose_hits": loose_hits,
        "loose_recall": round(loose_hits / max(1, visible_rows), 4),
        "visible_misses_strict": visible_rows - strict_hits,
        "invisible_frames": invisible_rows,
        "invisible_no_box": invisible_no_box,
        "invisible_no_box_rate": round(invisible_no_box / max(1, invisible_rows), 4),
        "selected_frames_total": len(selected),
    }
    return summary, frame_rows


def aggregate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    visible = sum(int(r["visible_frames"]) for r in rows)
    strict = sum(int(r["strict_hits"]) for r in rows)
    loose = sum(int(r["loose_hits"]) for r in rows)
    invisible = sum(int(r["invisible_frames"]) for r in rows)
    invisible_no_box = sum(int(r["invisible_no_box"]) for r in rows)
    selected = sum(int(r["selected_frames_total"]) for r in rows)
    return {
        "mode": mode,
        "clip": "__ALL__",
        "label_frames": sum(int(r["label_frames"]) for r in rows),
        "visible_frames": visible,
        "strict_hits": strict,
        "strict_recall": round(strict / max(1, visible), 4),
        "loose_hits": loose,
        "loose_recall": round(loose / max(1, visible), 4),
        "visible_misses_strict": visible - strict,
        "invisible_frames": invisible,
        "invisible_no_box": invisible_no_box,
        "invisible_no_box_rate": round(invisible_no_box / max(1, invisible), 4),
        "selected_frames_total": selected,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = parse_modes(args.modes)
    grouped_labels = labels_by_clip(Path(args.labels))
    results_dir = Path(args.results_dir)
    model_path = Path(args.model)
    per_clip: list[dict[str, Any]] = []
    aggregate_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for clip, clip_labels in grouped_labels.items():
        top_path = top_tubes_path(results_dir, clip)
        if top_path is None:
            continue
        rows = selector.load_ranked_rows(top_path, args.max_rank)
        if not rows:
            continue
        scored, _meta = selector.score_rows(rows, model_path)
        by_frame = selector.group_by_frame(scored)
        write_csv(out_dir / "scored" / f"{clip}.csv", scored)
        for mode in modes:
            mode_name = str(mode["name"])
            selected = select_mode(
                by_frame,
                mode,
                max_jump_px=args.max_jump_px,
                transition_weight=args.transition_weight,
                size_jump_weight=args.size_jump_weight,
            )
            threshold = float(mode.get("threshold", 0.0))
            selected_rows = selector.output_rows(clip, scored, selected, threshold=threshold)
            summary, frame_rows = evaluate_rows(clip_labels, selected_rows, args.strict_tol_px, args.loose_tol_px)
            summary.update({"mode": mode_name, "clip": clip})
            per_clip.append(summary)
            aggregate_by_mode[mode_name].append(summary)
            mode_dir = out_dir / mode_name / clip
            write_csv(mode_dir / "sequence_selected_tracks.csv", selected_rows)
            write_csv(mode_dir / "frame_eval.csv", frame_rows)
            (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    aggregate_rows = [aggregate(rows, mode) for mode, rows in sorted(aggregate_by_mode.items())]
    write_csv(out_dir / "per_clip_summary.csv", per_clip)
    write_csv(out_dir / "aggregate_summary.csv", aggregate_rows)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "labels": args.labels,
                "results_dir": args.results_dir,
                "model": args.model,
                "max_rank": args.max_rank,
                "modes": modes,
                "strict_tol_px": args.strict_tol_px,
                "loose_tol_px": args.loose_tol_px,
            },
            indent=2,
        )
    )
    print(out_dir / "aggregate_summary.csv")
    print(json.dumps(aggregate_rows, indent=2))


if __name__ == "__main__":
    main()
