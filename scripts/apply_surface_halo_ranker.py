#!/usr/bin/env python3
"""Apply a trained surface-halo ranker to candidate CSVs.

The surface-halo proposal branch is useful only if we can bound candidate
pressure before the JS2/state selector sees it. This tool scores candidates,
reranks each frame by the learned surface-halo score, and optionally keeps only
the top N per frame.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--clip", default="")
    p.add_argument("--max_rank", type=int, default=360)
    p.add_argument("--top_per_frame", type=int, default=40)
    p.add_argument("--overwrite_scores", action="store_true")
    p.add_argument(
        "--mark_selected",
        action="store_true",
        help="Mark exported rows as selected. Use when top_per_frame=1 is being exported as a diagnostic selected-track CSV.",
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


def row_matches_clip(row: dict[str, str], clip: str) -> bool:
    if not clip:
        return True
    row_clip = str(row.get("clip", "")).strip()
    return row_clip in {"", clip}


def fnum(value: Any, default: float | None = None) -> float | None:
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


def bounded_logit(prob: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(prob)))
    return float(math.log(p / (1.0 - p)))


def vectorize(
    rows: list[dict[str, str]],
    numeric: list[str],
    sources: list[str],
    variants: list[str],
) -> np.ndarray:
    data: list[list[float]] = []
    for row in rows:
        vals: list[float] = []
        for key in numeric:
            value = fnum(row.get(key))
            vals.append(np.nan if value is None else value)
        src = str(row.get("cand_source", ""))
        vals.extend(1.0 if key == f"src_{src}" else 0.0 for key in sources)
        variant = str(row.get("proposal_variant", ""))
        vals.extend(1.0 if key == f"variant_{variant}" else 0.0 for key in variants)
        data.append(vals)
    return np.asarray(data, dtype=np.float64)


def predict_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def score_rows(rows: list[dict[str, str]], bundle: dict[str, Any], overwrite_scores: bool) -> list[dict[str, Any]]:
    numeric = [str(v) for v in bundle.get("numeric_features", [])]
    sources = [str(v) for v in bundle.get("source_features", [])]
    variants = [str(v) for v in bundle.get("variant_features", [])]
    model = bundle["model"]
    scores = predict_score(model, vectorize(rows, numeric, sources, variants)) if rows else np.asarray([])
    out: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        rec: dict[str, Any] = dict(row)
        score_f = float(score)
        rec["surface_halo_score"] = round(score_f, 6)
        rec["surface_halo_logit"] = round(bounded_logit(score_f), 6) if 0.0 <= score_f <= 1.0 else round(score_f, 6)
        rec["surface_halo_model"] = str(bundle.get("best_model_interleaved", bundle.get("best_model_loco", "")))
        if overwrite_scores:
            rec["base_score"] = rec.get("score", "")
            rec["base_verified_score"] = rec.get("verified_score", "")
            rec["score"] = round(score_f, 6)
            rec["verified_score"] = round(score_f, 6)
        out.append(rec)
    return out


def rerank_by_frame(rows: list[dict[str, Any]], top_per_frame: int, mark_selected: bool = False) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[fint(row.get("frame"), -1)].append(row)
    out: list[dict[str, Any]] = []
    for frame in sorted(by_frame):
        frame_rows = sorted(
            by_frame[frame],
            key=lambda r: (fnum(r.get("surface_halo_score"), -1e9) or -1e9, -fint(r.get("rank"), 999999)),
            reverse=True,
        )
        if top_per_frame > 0:
            frame_rows = frame_rows[:top_per_frame]
        for idx, row in enumerate(frame_rows, start=1):
            rec = dict(row)
            rec["surface_halo_parent_rank"] = rec.get("rank", "")
            rec["rank"] = str(idx)
            if mark_selected:
                rec["selected"] = "1"
            out.append(rec)
    return out


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    rows = []
    for row in read_csv(Path(args.candidates)):
        if not row_matches_clip(row, args.clip):
            continue
        if fint(row.get("rank"), 999999) <= args.max_rank:
            rec = dict(row)
            if args.clip and not str(rec.get("clip", "")).strip():
                rec["clip"] = args.clip
            rows.append(rec)
    scored = score_rows(rows, bundle, args.overwrite_scores)
    ranked = rerank_by_frame(scored, args.top_per_frame, args.mark_selected)
    write_csv(Path(args.out_csv), ranked)
    print(Path(args.out_csv))


if __name__ == "__main__":
    main()
