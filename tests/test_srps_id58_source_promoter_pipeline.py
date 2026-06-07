from __future__ import annotations

import csv
from pathlib import Path

from scripts import audit_srps_id58_source_promoter_provenance as audit
from scripts import build_srps_id58_runtime_source_promoter_dataset as build
from scripts import train_srps_id58_runtime_source_promoter as train


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows(frames: range = range(654, 700)) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in frames:
        for rank in [1, 2, 3]:
            dist = 4 if rank == 2 else 40 + rank
            rows.append(
                {
                    "dataset": "e271",
                    "clip": "clip",
                    "frame": frame,
                    "candidate_id": f"e271:{frame}:{rank}",
                    "rank": rank,
                    "track_id": rank,
                    "source_family": "large_dark" if rank == 2 else "map",
                    "source": "large_dark" if rank == 2 else "map",
                    "x": 10 * rank,
                    "y": 20 * rank,
                    "w": 12,
                    "h": 12,
                    "cx": 10 * rank + 6,
                    "cy": 20 * rank + 6,
                    "score": 20 - rank,
                    "verified_score": 5 + 10 * int(rank == 2),
                    "cand_texture": 0.2 if rank == 2 else 1.0,
                    "distance_to_reviewed_center": dist,
                    "label_strict_8px": int(dist <= 8),
                    "label_loose_16px": int(dist <= 16),
                    "pi_selected": int(rank == 1),
                    "split_group": "e271_frozen",
                }
            )
    # Add a tiny regression dataset so the builder gate can see non-focus rows.
    for frame in range(500, 530):
        for rank in [1, 2]:
            rows.append(
                {
                    "dataset": "aaf1",
                    "clip": "aaf1",
                    "frame": frame,
                    "candidate_id": f"aaf1:{frame}:{rank}",
                    "rank": rank,
                    "track_id": rank,
                    "source_family": "map",
                    "source": "map",
                    "x": rank,
                    "y": rank,
                    "w": 10,
                    "h": 10,
                    "score": 10 + rank,
                    "verified_score": 4 + rank,
                    "cand_texture": float(rank),
                    "distance_to_reviewed_center": 5 if rank == 2 else 50,
                    "label_strict_8px": int(rank == 2),
                    "label_loose_16px": int(rank == 2),
                    "pi_selected": int(rank == 1),
                    "split_group": "aaf1_block_a",
                }
            )
    return rows


def test_id58_audit_traces_external_positive_to_pi_candidate_rows(tmp_path: Path) -> None:
    rows = candidate_rows(range(654, 657))
    external = [dict(r, review_reason="x", review_role="nearest_gt") for r in rows if r["rank"] == 2]
    runtime = [dict(r, row_type="RUNTIME_ALTERNATIVE") for r in rows if r["rank"] == 1]
    predictions = [dict(r, model="m", model_score=0.9) for r in external]
    write_csv(tmp_path / "external.csv", external)
    write_csv(tmp_path / "candidates.csv", rows)
    write_csv(tmp_path / "runtime.csv", runtime)
    write_csv(tmp_path / "pred.csv", predictions)

    summary = audit.run(
        audit.parse_args(
            [
                "--external_candidates",
                str(tmp_path / "external.csv"),
                "--pi_candidate_rows",
                str(tmp_path / "candidates.csv"),
                "--prediction_rows",
                str(tmp_path / "pred.csv"),
                "--runtime_rows",
                str(tmp_path / "runtime.csv"),
                "--out_dir",
                str(tmp_path / "out"),
            ]
        )
    )

    assert summary["external_positive_rows"] == 3
    assert summary["matched_pi_positive_rows"] == 3
    assert summary["phase_a_gate_pass"] == 1
    assert (tmp_path / "out" / "provenance_matches.csv").exists()


def test_id58_dataset_excludes_label_leakage_from_feature_schema(tmp_path: Path) -> None:
    rows = candidate_rows()
    write_csv(tmp_path / "candidates.csv", rows)
    write_csv(tmp_path / "provenance.csv", [r for r in rows if r["dataset"] == "e271" and r["rank"] == 2])

    summary = build.run(
        build.parse_args(
            [
                "--candidate_rows",
                str(tmp_path / "candidates.csv"),
                "--provenance_matches",
                str(tmp_path / "provenance.csv"),
                "--out_dir",
                str(tmp_path / "dataset"),
            ]
        )
    )

    assert summary["focus_strict_available_frames"] == 45
    assert summary["focus_loose_available_frames"] == 45
    assert summary["leakage_column_count"] == 0
    schema = (tmp_path / "dataset" / "trainable_feature_columns.json").read_text()
    assert "distance_to_reviewed_center" not in schema
    assert "label_strict_8px" not in schema


def test_id58_trainer_writes_policy_json(tmp_path: Path) -> None:
    rows = candidate_rows()
    write_csv(tmp_path / "candidates.csv", rows)
    write_csv(tmp_path / "provenance.csv", [r for r in rows if r["dataset"] == "e271" and r["rank"] == 2])
    build.run(
        build.parse_args(
            [
                "--candidate_rows",
                str(tmp_path / "candidates.csv"),
                "--provenance_matches",
                str(tmp_path / "provenance.csv"),
                "--out_dir",
                str(tmp_path / "dataset"),
            ]
        )
    )

    summary = train.run(
        train.parse_args(
            [
                "--source_rows",
                str(tmp_path / "dataset" / "source_promoter_rows.csv"),
                "--feature_columns",
                str(tmp_path / "dataset" / "trainable_feature_columns.json"),
                "--dataset_summary",
                str(tmp_path / "dataset" / "summary.json"),
                "--out_dir",
                str(tmp_path / "model"),
                "--target",
                "loose",
            ]
        )
    )

    assert summary["rows"] > 0
    assert (tmp_path / "model" / "model_summary.csv").exists()
    assert list((tmp_path / "model" / "models").glob("*_policy.json"))
