#!/usr/bin/env python3
"""Evaluate a simple selector-router policy over existing selector outputs.

This is an offline diagnostic for the current algorithm question:

* permissive Viterbi keeps visible continuity but hallucinates through null
  windows;
* conservative HMM/null mode suppresses no-target frames but can miss continuous
  visible clips.

The router score here is intentionally simple and reproducible: for each clip,
estimate how strongly the top candidate stream looks background/static-locked
from CLBA fields, then choose HMM only when that risk is above a threshold. This
is not a runtime implementation. It is a gate-quality probe that tells us
whether CLBA static-vs-target evidence can decide which selector family belongs
on a clip/window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = "-1.0,-0.25,0.0,0.25,0.5,0.75,1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True, help="Directory containing per-clip top_tubes.csv files.")
    p.add_argument("--hmm_dir", required=True, help="Directory containing HMM selector per-clip eval outputs.")
    p.add_argument("--viterbi_dir", required=True, help="Directory containing Viterbi selector per-clip eval outputs.")
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated CLBA risk thresholds. Use HMM when risk > threshold.",
    )
    p.add_argument("--risk_max_rank", type=int, default=1, help="Use rows with rank <= this value for risk.")
    p.add_argument(
        "--risk_stat",
        choices=("median", "mean", "p75", "p90"),
        default="median",
        help="Statistic over per-row risk values: bg_static_likelihood - target_likelihood.",
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


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any, default: int = 0) -> int:
    out = safe_float(value)
    if out is None:
        return default
    return int(out)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize_values(values: list[float], stat: str) -> tuple[float, dict[str, Any]]:
    if not values:
        summary = {
            "risk_rows": 0,
            "risk_mean": "",
            "risk_median": "",
            "risk_p75": "",
            "risk_p90": "",
            "risk_min": "",
            "risk_max": "",
        }
        return 0.0, summary

    mean = statistics.fmean(values)
    median = statistics.median(values)
    p75 = percentile(values, 0.75)
    p90 = percentile(values, 0.90)
    chosen = {
        "mean": mean,
        "median": median,
        "p75": p75,
        "p90": p90,
    }[stat]
    summary = {
        "risk_rows": len(values),
        "risk_mean": round(mean, 6),
        "risk_median": round(median, 6),
        "risk_p75": round(p75, 6),
        "risk_p90": round(p90, 6),
        "risk_min": round(min(values), 6),
        "risk_max": round(max(values), 6),
    }
    return chosen, summary


def clip_from_eval_summary(path: Path) -> str:
    # <selector_dir>/<clip>/eval/summary.json
    return path.parent.parent.name


def read_selector_summaries(selector_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(selector_dir.glob("*/eval/summary.json")):
        summaries[clip_from_eval_summary(path)] = json.loads(path.read_text())
    return summaries


def read_clip_risks(results_dir: Path, max_rank: int, stat: str) -> dict[str, dict[str, Any]]:
    risks: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*/top_tubes.csv")):
        values: list[float] = []
        frames: set[int] = set()
        for row in read_csv(path):
            rank = safe_int(row.get("rank"), default=999999)
            if rank > max_rank:
                continue
            target = safe_float(row.get("clba_target_likelihood"))
            bg_static = safe_float(row.get("clba_bg_static_likelihood"))
            if target is None or bg_static is None:
                continue
            values.append(bg_static - target)
            frames.add(safe_int(row.get("frame"), default=-1))
        chosen, value_summary = summarize_values(values, stat)
        risks[path.parent.name] = {
            "clip": path.parent.name,
            "risk": round(chosen, 6),
            "risk_stat": stat,
            "risk_max_rank": max_rank,
            "risk_frames": len([f for f in frames if f >= 0]),
            **value_summary,
        }
    return risks


def parse_thresholds(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("no thresholds supplied")
    return values


def metric(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


def aggregate(rows: list[dict[str, Any]], mode: str, threshold: float | str) -> dict[str, Any]:
    visible = sum(int(row["visible_frames"]) for row in rows)
    strict = sum(int(row["strict_hits"]) for row in rows)
    loose = sum(int(row["loose_hits"]) for row in rows)
    invisible = sum(int(row["invisible_frames"]) for row in rows)
    invisible_no_box = sum(int(row["invisible_no_box"]) for row in rows)
    selected = sum(int(row["selected_frames_total"]) for row in rows)
    return {
        "mode": mode,
        "threshold": threshold,
        "clip": "__ALL__",
        "selector": "mixed" if mode == "routed" else mode,
        "risk": "",
        "label_frames": sum(int(row["label_frames"]) for row in rows),
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
        "hmm_clips": sum(1 for row in rows if row["selector"] == "hmm"),
        "viterbi_clips": sum(1 for row in rows if row["selector"] == "viterbi"),
    }


def summary_row(
    clip: str,
    threshold: float | str,
    mode: str,
    selector_name: str,
    risk: float | str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    visible = metric(summary, "visible_frames")
    strict = metric(summary, "strict_hits")
    loose = metric(summary, "loose_hits")
    invisible = metric(summary, "invisible_frames")
    invisible_no_box = metric(summary, "invisible_no_box")
    return {
        "mode": mode,
        "threshold": threshold,
        "clip": clip,
        "selector": selector_name,
        "risk": risk,
        "label_frames": metric(summary, "label_frames"),
        "visible_frames": visible,
        "strict_hits": strict,
        "strict_recall": round(strict / max(1, visible), 4),
        "loose_hits": loose,
        "loose_recall": round(loose / max(1, visible), 4),
        "visible_misses_strict": visible - strict,
        "invisible_frames": invisible,
        "invisible_no_box": invisible_no_box,
        "invisible_no_box_rate": round(invisible_no_box / max(1, invisible), 4),
        "selected_frames_total": metric(summary, "selected_frames_total"),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hmm = read_selector_summaries(Path(args.hmm_dir))
    viterbi = read_selector_summaries(Path(args.viterbi_dir))
    risks = read_clip_risks(Path(args.results_dir), args.risk_max_rank, args.risk_stat)
    thresholds = parse_thresholds(args.thresholds)

    clips = sorted(set(hmm) & set(viterbi) & set(risks))
    if not clips:
        raise SystemExit("no clips found in all inputs")

    risk_rows = [risks[clip] for clip in clips]
    write_csv(out_dir / "clip_risk.csv", risk_rows)

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    hmm_rows = [
        summary_row(clip, "all", "hmm", "hmm", risks[clip]["risk"], hmm[clip])
        for clip in clips
    ]
    vit_rows = [
        summary_row(clip, "all", "viterbi", "viterbi", risks[clip]["risk"], viterbi[clip])
        for clip in clips
    ]
    rows.extend(hmm_rows)
    rows.extend(vit_rows)
    summary_rows.append(aggregate(hmm_rows, "hmm", "all"))
    summary_rows.append(aggregate(vit_rows, "viterbi", "all"))

    for threshold in thresholds:
        routed_rows: list[dict[str, Any]] = []
        for clip in clips:
            risk = float(risks[clip]["risk"])
            use_hmm = risk > threshold
            selector_name = "hmm" if use_hmm else "viterbi"
            selected_summary = hmm[clip] if use_hmm else viterbi[clip]
            routed_rows.append(
                summary_row(
                    clip=clip,
                    threshold=threshold,
                    mode="routed",
                    selector_name=selector_name,
                    risk=round(risk, 6),
                    summary=selected_summary,
                )
            )
        rows.extend(routed_rows)
        summary_rows.append(aggregate(routed_rows, "routed", threshold))

    write_csv(out_dir / "router_policy_by_clip.csv", rows)
    write_csv(out_dir / "router_policy_summary.csv", summary_rows)
    metadata = {
        "results_dir": str(Path(args.results_dir)),
        "hmm_dir": str(Path(args.hmm_dir)),
        "viterbi_dir": str(Path(args.viterbi_dir)),
        "risk_stat": args.risk_stat,
        "risk_max_rank": args.risk_max_rank,
        "thresholds": thresholds,
        "clips": clips,
        "note": (
            "Offline clip-level diagnostic. Runtime should replace clip-level "
            "median risk with candidate/window-local risk."
        ),
    }
    (out_dir / "router_policy_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"summary": summary_rows, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
