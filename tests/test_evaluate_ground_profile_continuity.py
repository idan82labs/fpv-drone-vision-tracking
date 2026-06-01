import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_ground_profile_continuity import (
    DEFAULT_BACKDROP_MANIFEST,
    ProfileSpec,
    discover_gated_js1_profiles,
    evaluate_profile,
    load_ground_labels,
    run_lengths,
)


class GroundProfileContinuityTests(unittest.TestCase):
    def test_load_ground_labels_prefers_manual_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "labels.csv"
            path.write_text(
                "clip,frame,visible,det_x,det_y,det_w,det_h,bg_split,source,confidence\n"
                "c1,10,1,1,1,3,3,textured_non_sky,surface_audit_seed_v2,medium_high\n"
                "c1,10,1,5,5,3,3,textured_non_sky,vision_manual_frame_by_frame,high\n"
                "c1,11,1,5,5,3,3,skyline_surface,vision_manual_frame_by_frame,high\n"
            )
            labels = load_ground_labels([path], "core_ground")
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["det_x"], 5.0)

    def test_true_ground_view_requires_manifest(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            load_ground_labels([], "true_ground")

    def test_true_ground_view_uses_target_backdrop_and_audit_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels_path = root / "labels.csv"
            labels_path.write_text(
                "clip,frame,visible,det_x,det_y,det_w,det_h,bg_split,source,confidence\n"
                "c1,1,1,1,1,3,3,surface_terrain_tree_line,vision_manual,high\n"
                "c1,2,1,2,2,3,3,surface_terrain_tree_line,vision_manual,high\n"
                "c1,3,1,3,3,3,3,surface_terrain_tree_line,vision_manual,high\n"
            )
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(
                "clip_pattern,frame_start,frame_end,target_backdrop,frame_context,audit_status,label_provenance,exclude_from_true_ground\n"
                "c1,1,1,terrain,terrain,visual_confirmed,test_ground,0\n"
                "c1,2,2,skyline_above_terrain,skyline,contact_sheet_reviewed,test_skyline,1\n"
                "c1,3,3,road,road,interpolated,test_interpolated,0\n"
            )

            labels = load_ground_labels([labels_path], "true_ground", [manifest_path])

            self.assertEqual([r["frame"] for r in labels], [1])
            self.assertEqual(labels[0]["target_backdrop"], "terrain")
            self.assertEqual(labels[0]["audit_status"], "visual_confirmed")
            self.assertEqual(labels[0]["label_provenance"], "test_ground")

    def test_skyline_view_uses_only_audited_skyline_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels_path = root / "labels.csv"
            labels_path.write_text(
                "clip,frame,visible,det_x,det_y,det_w,det_h,bg_split,source,confidence\n"
                "c1,1,1,1,1,3,3,surface_terrain_tree_line,vision_manual,high\n"
                "c1,2,1,2,2,3,3,surface_terrain_tree_line,vision_manual,high\n"
                "c1,3,1,3,3,3,3,surface_terrain_tree_line,vision_manual,high\n"
            )
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(
                "clip_pattern,frame_start,frame_end,target_backdrop,frame_context,audit_status,label_provenance,exclude_from_true_ground\n"
                "c1,1,1,terrain,terrain,visual_confirmed,test_ground,0\n"
                "c1,2,2,skyline_above_terrain,skyline,contact_sheet_reviewed,test_skyline,1\n"
                "c1,3,3,skyline_above_terrain,skyline,interpolated,test_weak_skyline,1\n"
            )

            labels = load_ground_labels([labels_path], "skyline_above_terrain", [manifest_path])

            self.assertEqual([r["frame"] for r in labels], [2])
            self.assertEqual(labels[0]["target_backdrop"], "skyline_above_terrain")
            self.assertEqual(labels[0]["audit_status"], "contact_sheet_reviewed")

    def test_default_manifest_marks_aaf1_skyline_rows_out_of_true_ground(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "labels.csv"
            clip = "aaf1eafd-36d7-43e4-a539-fd79029ddf90"
            path.write_text(
                "clip,frame,visible,det_x,det_y,det_w,det_h,bg_split,source,confidence,target_backdrop,audit_status\n"
                f"{clip},288,1,1,1,3,3,surface_terrain_tree_line,vision_manual,high,terrain,visual_confirmed\n"
                f"{clip},640,1,2,2,3,3,surface_terrain_tree_line,vision_manual,high,terrain,visual_confirmed\n"
                f"{clip},641,1,3,3,3,3,surface_terrain_tree_line,vision_manual,high,terrain,visual_confirmed\n"
            )
            manifest_path = Path(__file__).resolve().parents[1] / DEFAULT_BACKDROP_MANIFEST

            legacy_view = load_ground_labels([path], "core_ground", [manifest_path])
            by_frame = {r["frame"]: r for r in legacy_view}
            self.assertEqual(by_frame[288]["target_backdrop"], "skyline_above_terrain")
            self.assertEqual(by_frame[288]["exclude_from_true_ground"], "1")
            self.assertEqual(by_frame[640]["target_backdrop"], "skyline_above_terrain")

            true_ground = load_ground_labels([path], "true_ground", [manifest_path])
            self.assertEqual([r["frame"] for r in true_ground], [641])

    def test_run_lengths_requires_consecutive_frames(self):
        rows = [
            {"frame": 1, "strict_hit": 1},
            {"frame": 2, "strict_hit": 1},
            {"frame": 4, "strict_hit": 1},
            {"frame": 5, "strict_hit": 0},
            {"frame": 6, "strict_hit": 1},
        ]
        self.assertEqual(run_lengths(rows, "strict_hit"), (2, 1, 2))

    def test_evaluate_profile_computes_strict_and_loose_hits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            selected_dir = root / "run" / "c1"
            selected_dir.mkdir(parents=True)
            selected = selected_dir / "selected_tracks.csv"
            selected.write_text(
                "frame,x,y,w,h,score\n"
                "1,10,10,3,3,1\n"
                "2,50,50,3,3,1\n"
            )
            labels = [
                {
                    "clip": "c1",
                    "frame": 1,
                    "det_x": 10.0,
                    "det_y": 10.0,
                    "det_w": 3.0,
                    "det_h": 3.0,
                    "bg_split": "textured_non_sky",
                    "target_backdrop": "terrain",
                    "audit_status": "visual_confirmed",
                    "evidence_class": "unit_seed",
                },
                {
                    "clip": "c1",
                    "frame": 2,
                    "det_x": 41.0,
                    "det_y": 50.0,
                    "det_w": 3.0,
                    "det_h": 3.0,
                    "bg_split": "textured_non_sky",
                    "target_backdrop": "terrain",
                    "audit_status": "visual_confirmed",
                    "evidence_class": "unit_seed",
                },
            ]
            spec = ProfileSpec("test", str(root / "run" / "{clip}" / "selected_tracks.csv"))
            frame_rows, _clip_rows, summary = evaluate_profile(spec, labels, 8.0, 16.0)
            self.assertEqual(len(frame_rows), 2)
            self.assertEqual(summary["evidence_class"], "unit_seed")
            self.assertEqual(summary["target_backdrops"], "terrain")
            self.assertEqual(summary["strict_hits"], 1)
            self.assertEqual(summary["loose_hits"], 2)
            self.assertEqual(summary["longest_strict_run"], 1)
            self.assertEqual(summary["longest_loose_run"], 2)

    def test_discovers_nested_gated_js1_profiles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = (
                root
                / "gated_surface_branch_v2_repeated_risk_sweep"
                / "risk07"
                / "clip-a"
                / "js1_eval"
                / "best_frame_predictions.csv"
            )
            target.parent.mkdir(parents=True)
            target.write_text("frame,selected\n")

            specs = discover_gated_js1_profiles(root)

            self.assertEqual(len(specs), 1)
            self.assertEqual(
                specs[0].template,
                str(root / "gated_surface_branch_v2_repeated_risk_sweep" / "risk07" / "{clip}" / "js1_eval" / "best_frame_predictions.csv"),
            )
            self.assertEqual(specs[0].name, "cs_js2_gated_surface_branch_v2_repeated_risk_sweep_risk07")


if __name__ == "__main__":
    unittest.main()
