import tempfile
import unittest
from pathlib import Path

from scripts import train_surface_xy_ranker as surface


class SurfaceXYRankerTests(unittest.TestCase):
    def test_parse_threshold_range_includes_endpoint(self):
        self.assertEqual(surface.parse_thresholds("0.72:0.76:0.02"), [0.72, 0.74, 0.76])

    def test_fallback_rows_use_learned_only_above_threshold(self):
        predictions = [
            {"model": "baseline_verified_score", "clip": "clip-a", "frame": 1, "score": 0.1, "strict_hit": False},
            {"model": "extra_trees", "clip": "clip-a", "frame": 1, "score": 0.9, "strict_hit": True},
            {"model": "baseline_verified_score", "clip": "clip-a", "frame": 2, "score": 0.1, "strict_hit": True},
            {"model": "extra_trees", "clip": "clip-a", "frame": 2, "score": 0.3, "strict_hit": False},
        ]

        rows = surface.fallback_rows(predictions, "extra_trees", threshold=0.76)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["strict_hit"], True)
        self.assertEqual(rows[0]["fallback_used_learned"], True)
        self.assertEqual(rows[1]["strict_hit"], True)
        self.assertEqual(rows[1]["fallback_used_learned"], False)

    def test_visible_parser_accepts_common_false_values(self):
        self.assertFalse(surface.label_visible({"visible": "false"}))
        self.assertFalse(surface.label_visible({"visible": "not_visible"}))
        self.assertTrue(surface.label_visible({"visible": "true"}))
        self.assertTrue(surface.label_visible({"visible": ""}))

    def test_infer_features_excludes_runtime_unavailable_candidate_flags(self):
        numeric, _sources = surface.infer_features(
            [
                {
                    "rank": "1",
                    "cand_frame": "12",
                    "cand_is_current": "1",
                    "candidate_frame": "12",
                    "candidate_is_current": "1",
                    "verified_score": "4.2",
                }
            ]
        )

        self.assertIn("rank", numeric)
        self.assertIn("verified_score", numeric)
        self.assertNotIn("cand_frame", numeric)
        self.assertNotIn("cand_is_current", numeric)
        self.assertNotIn("candidate_frame", numeric)
        self.assertNotIn("candidate_is_current", numeric)

    def test_load_top_tubes_preserves_frame_zero(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_dir = root / "clip-a"
            clip_dir.mkdir()
            (clip_dir / "top_tubes.csv").write_text(
                "frame,rank,x,y,w,h,verified_score\n0,1,0,0,3,3,4.2\n1,1,1,1,3,3,3.1\n"
            )

            by_frame = surface.load_top_tubes(root, "clip-a", max_rank=80)

        self.assertEqual(set(by_frame), {0, 1})
        self.assertEqual(surface.row_bbox(by_frame[0][0]), (0.0, 0.0, 3.0, 3.0))

    def test_fallback_rows_preserves_frame_zero(self):
        predictions = [
            {"model": "baseline_verified_score", "clip": "clip-a", "frame": 0, "score": 0.1, "strict_hit": False},
            {"model": "extra_trees", "clip": "clip-a", "frame": 0, "score": 0.9, "strict_hit": True},
        ]

        rows = surface.fallback_rows(predictions, "extra_trees", threshold=0.5)

        self.assertEqual(rows[0]["frame"], 0)
        self.assertTrue(rows[0]["strict_hit"])

    def test_nested_fallback_selects_threshold_without_heldout_clip(self):
        predictions = [
            {"model": "baseline_verified_score", "clip": "a", "frame": 1, "score": 0.0, "strict_hit": False, "loose_hit": False, "oracle_hit": True, "confidence": "high"},
            {"model": "extra_trees", "clip": "a", "frame": 1, "score": 0.9, "strict_hit": True, "loose_hit": True, "oracle_hit": True, "confidence": "high"},
            {"model": "baseline_verified_score", "clip": "b", "frame": 1, "score": 0.0, "strict_hit": True, "loose_hit": True, "oracle_hit": True, "confidence": "high"},
            {"model": "extra_trees", "clip": "b", "frame": 1, "score": 0.9, "strict_hit": False, "loose_hit": False, "oracle_hit": True, "confidence": "high"},
        ]

        rows, selected = surface.nested_fallback_rows(predictions, ["extra_trees"], [0.5, 0.95], ["a", "b"])

        by_clip = {row["heldout_clip"]: row for row in selected}
        self.assertEqual(by_clip["a"]["selected_threshold"], 0.95)
        self.assertEqual(by_clip["b"]["selected_threshold"], 0.5)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["model"] == "nested_fallback" for row in rows))

    def test_gated_fallback_requires_gate_to_use_learned(self):
        predictions = [
            {"model": "baseline_verified_score", "clip": "a", "frame": 1, "rank": "1", "score": 1.0, "strict_hit": False, "loose_hit": False, "oracle_hit": True},
            {"model": "extra_trees", "clip": "a", "frame": 1, "rank": "2", "score": 0.9, "strict_hit": True, "loose_hit": True, "oracle_hit": True},
            {"model": "baseline_verified_score", "clip": "a", "frame": 2, "rank": "1", "score": 1.0, "strict_hit": True, "loose_hit": True, "oracle_hit": True},
            {"model": "extra_trees", "clip": "a", "frame": 2, "rank": "2", "score": 0.9, "strict_hit": False, "loose_hit": False, "oracle_hit": True},
        ]
        top_by_clip = {
            "a": {
                1: [{"rank": "2", "cand_source": "large_dark", "cand_attached_support": "10", "tube_mean_pair_bg": "-0.2"}],
                2: [{"rank": "2", "cand_source": "map", "cand_attached_support": "0", "tube_mean_pair_bg": "0.2"}],
            }
        }

        rows = surface.gated_fallback_rows(predictions, top_by_clip, "extra_trees", 0.76, "high_support")

        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["fallback_used_learned"])
        self.assertFalse(rows[1]["fallback_used_learned"])
        self.assertTrue(rows[0]["strict_hit"])
        self.assertTrue(rows[1]["strict_hit"])

    def test_load_extra_examples_reads_hard_label_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra.csv"
            path.write_text("frame,hard_label,rank\n10,1,2\n11,0,1\n12,,3\n")

            rows, y = surface.load_extra_examples(path)

            self.assertEqual([row["frame"] for row in rows], ["10", "11"])
            self.assertEqual(y.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
