import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_oracle_by_backdrop import (
    CandidateSpec,
    candidate_bbox,
    evaluate_stream,
)
from scripts.evaluate_ground_profile_continuity import load_ground_labels


class OracleByBackdropTests(unittest.TestCase):
    def test_candidate_bbox_supports_selected_prefix(self):
        row = {"selected_x": "10", "selected_y": "20", "selected_w": "3", "selected_h": "4"}
        self.assertEqual(candidate_bbox(row), (10.0, 20.0, 3.0, 4.0))

    def test_evaluate_stream_reports_oracle_by_k(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels_path = root / "labels.csv"
            labels_path.write_text(
                "clip,frame,visible,det_x,det_y,det_w,det_h,bg_split,source,confidence\n"
                "c1,10,1,10,10,4,4,surface_terrain_tree_line,vision_manual,high\n"
                "c1,11,1,30,30,4,4,surface_terrain_tree_line,vision_manual,high\n"
            )
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(
                "clip_pattern,frame_start,frame_end,target_backdrop,frame_context,audit_status,label_provenance,exclude_from_true_ground\n"
                "c1,10,10,terrain,terrain,visual_confirmed,test,0\n"
                "c1,11,11,road,road,contact_sheet_reviewed,test,0\n"
            )
            cand_dir = root / "cands" / "c1"
            cand_dir.mkdir(parents=True)
            (cand_dir / "top_tubes.csv").write_text(
                "frame,rank,x,y,w,h,score\n"
                "10,1,100,100,4,4,9\n"
                "10,2,10,10,4,4,1\n"
                "11,1,30,30,4,4,5\n"
            )
            labels = load_ground_labels([labels_path], "true_ground", [manifest_path])
            frames, summary, groups, meta = evaluate_stream(
                CandidateSpec("cand", str(root / "cands" / "{clip}" / "top_tubes.csv")),
                labels,
                [1, 2],
                strict_tol=8.0,
                loose_tol=16.0,
                max_rank=80,
            )

            self.assertEqual(len(frames), 2)
            self.assertEqual(summary[0]["oracle_strict_at_1"], 1)
            self.assertEqual(summary[0]["oracle_strict_at_2"], 2)
            self.assertEqual(summary[0]["oracle_strict_rate_at_1"], 0.5)
            by_backdrop = {(r.get("target_backdrop"), r.get("stream")): r for r in groups if "clip" not in r}
            self.assertEqual(by_backdrop[("terrain", "cand")]["oracle_strict_at_1"], 0)
            self.assertEqual(by_backdrop[("terrain", "cand")]["oracle_strict_at_2"], 1)
            self.assertEqual(by_backdrop[("road", "cand")]["oracle_strict_at_1"], 1)
            self.assertEqual(meta["candidate_rows_by_clip"]["c1"], 3)

    def test_missing_candidate_clip_is_reported(self):
        labels = [
            {
                "clip": "missing",
                "frame": 1,
                "det_x": 1.0,
                "det_y": 1.0,
                "det_w": 3.0,
                "det_h": 3.0,
                "target_backdrop": "terrain",
                "audit_status": "visual_confirmed",
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _frames, summary, _groups, meta = evaluate_stream(
                CandidateSpec("empty", str(root / "{clip}" / "top_tubes.csv")),
                labels,
                [1],
                strict_tol=8.0,
                loose_tol=16.0,
                max_rank=80,
            )
        self.assertEqual(summary[0]["frames_with_candidates"], 0)
        self.assertEqual(meta["missing_clips"], ["missing"])


if __name__ == "__main__":
    unittest.main()
