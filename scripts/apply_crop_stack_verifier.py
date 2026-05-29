#!/usr/bin/env python3
"""Apply a trained crop-stack verifier to exported top-tube rows.

This is an offline bridge between the crop-stack observation probe and the
existing selector harnesses. It writes top_tubes.csv-compatible rows with:

- ``crop_stack_score``: verifier probability/logit-like score;
- ``base_learned_score``: previous learned score, when present;
- optionally ``learned_score`` overwritten with the crop-stack score so existing
  Viterbi/HMM selectors can be evaluated without runtime changes.

Keep this as a lab tool until full-video selected/null metrics improve.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import profile_tube_alignment_features as align
    import train_crop_stack_verifier as crop
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts import profile_tube_alignment_features as align
    from scripts import train_crop_stack_verifier as crop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--top_tubes", help="Single top_tubes.csv input.")
    group.add_argument("--results_dir", help="Directory containing clip/top_tubes.csv files.")
    p.add_argument("--out_csv", help="Single-file output path for --top_tubes.")
    p.add_argument("--out_dir", help="Output directory for --results_dir.")
    p.add_argument("--model", required=True, help="Crop-stack verifier .joblib bundle.")
    p.add_argument("--video_dir", required=True)
    p.add_argument("--clip", default="", help="Clip id for single-file input if rows lack clip.")
    p.add_argument("--max_rank", type=int, default=20)
    p.add_argument("--overwrite_learned_score", action="store_true")
    p.add_argument("--orb_features", type=int, default=900)
    p.add_argument("--min_matches", type=int, default=18)
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


def safe_int(value: Any, default: int = 0) -> int:
    return crop.safe_int(value, default)


def predict_score(model: Any, x: np.ndarray, score_mode: str = "auto") -> np.ndarray:
    if hasattr(crop, "predict_model_score"):
        return crop.predict_model_score(model, x, score_mode)
    if score_mode == "decision_function":
        return model.decision_function(x)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"model", "window_radius", "crop_size", "patch_size", "detector_scale"}
    missing = sorted(required - set(bundle))
    if missing:
        raise SystemExit(f"crop-stack model bundle missing keys: {', '.join(missing)}")
    return bundle


def score_rows(
    rows: list[dict[str, str]],
    clip: str,
    bundle: dict[str, Any],
    frame_cache: align.FrameCache,
    transforms: align.TransformCache,
    tube_rows: dict[tuple[str, str], dict[int, dict[str, str]]],
    max_rank: int,
    overwrite_learned_score: bool,
) -> list[dict[str, Any]]:
    model = bundle["model"]
    score_mode = str(bundle.get("score_mode", "auto"))
    source_geometry_features = bool(bundle.get("source_geometry_features", False))
    out_rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    vector_indexes: list[int] = []
    for row in rows:
        rank = safe_int(row.get("rank"), 999999)
        if rank > max_rank:
            continue
        out = dict(row)
        if clip and not out.get("clip"):
            out["clip"] = clip
        out_rows.append(out)
        vector, meta = crop.extract_stack_features(
            out,
            frame_cache,
            transforms,
            tube_rows,
            int(bundle["window_radius"]),
            int(bundle["crop_size"]),
            int(bundle["patch_size"]),
            source_geometry_features,
        )
        out.update(meta)
        vectors.append(vector)
        vector_indexes.append(len(out_rows) - 1)
    if vectors:
        scores = predict_score(model, np.vstack(vectors).astype(np.float32), score_mode)
        for idx, score in zip(vector_indexes, scores):
            out = out_rows[idx]
            out["crop_stack_score"] = round(float(score), 6)
            if overwrite_learned_score:
                out["base_learned_score"] = out.get("learned_score", "")
                out["learned_score"] = round(float(score), 6)
    return out_rows


def copy_sidecars(in_csv: Path, out_csv: Path) -> None:
    for name in ("clba_augment_metadata.json",):
        src = in_csv.parent / name
        if src.exists():
            shutil.copy2(src, out_csv.parent / name)


def main() -> None:
    args = parse_args()
    bundle = load_bundle(Path(args.model))
    frame_cache = align.FrameCache(Path(args.video_dir), float(bundle["detector_scale"]))
    try:
        if args.top_tubes:
            if not args.out_csv:
                raise SystemExit("--out_csv is required with --top_tubes")
            top_path = Path(args.top_tubes)
            tube_rows = align.load_tube_rows([top_path])
            if args.clip and not any(k[0] == args.clip for k in tube_rows):
                tube_rows.update({(args.clip, tid): by_frame for (_old, tid), by_frame in list(tube_rows.items())})
            transforms = align.TransformCache(frame_cache, args.orb_features, args.min_matches)
            scored = score_rows(
                read_csv(top_path),
                args.clip,
                bundle,
                frame_cache,
                transforms,
                tube_rows,
                args.max_rank,
                args.overwrite_learned_score,
            )
            out_csv = Path(args.out_csv)
            write_csv(out_csv, scored)
            copy_sidecars(top_path, out_csv)
            print(out_csv)
            return

        if not args.out_dir:
            raise SystemExit("--out_dir is required with --results_dir")
        in_root = Path(args.results_dir)
        out_root = Path(args.out_dir)
        input_paths = sorted(in_root.glob("*/top_tubes.csv"))
        if not input_paths:
            raise SystemExit(f"no */top_tubes.csv files found under {in_root}")
        tube_rows = align.load_tube_rows(input_paths)
        transforms = align.TransformCache(frame_cache, args.orb_features, args.min_matches)
        count = 0
        row_count = 0
        for top_path in input_paths:
            rel = top_path.relative_to(in_root)
            clip = top_path.parent.name
            scored = score_rows(
                read_csv(top_path),
                clip,
                bundle,
                frame_cache,
                transforms,
                tube_rows,
                args.max_rank,
                args.overwrite_learned_score,
            )
            out_csv = out_root / rel
            write_csv(out_csv, scored)
            copy_sidecars(top_path, out_csv)
            count += 1
            row_count += len(scored)
        meta = {
            "model": args.model,
            "results_dir": args.results_dir,
            "out_dir": args.out_dir,
            "video_dir": args.video_dir,
            "max_rank": args.max_rank,
            "overwrite_learned_score": args.overwrite_learned_score,
            "files": count,
            "rows": row_count,
            "window_radius": bundle["window_radius"],
            "crop_size": bundle["crop_size"],
            "patch_size": bundle["patch_size"],
            "detector_scale": bundle["detector_scale"],
            "score_mode": bundle.get("score_mode", "auto"),
            "source_geometry_features": bool(bundle.get("source_geometry_features", False)),
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "crop_stack_score_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))
    finally:
        frame_cache.close()


if __name__ == "__main__":
    main()
