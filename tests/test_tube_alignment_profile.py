import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import profile_tube_alignment_features as align
from scripts import augment_top_tubes_alignment_features as augment


class TubeAlignmentProfileTests(unittest.TestCase):
    def test_project_applies_affine_homogeneous_matrix(self):
        mat = np.eye(3, dtype=np.float32)
        mat[0, 2] = 3.5
        mat[1, 2] = -2.0
        self.assertEqual(align.project(mat, (10.0, 20.0)), (13.5, 18.0))

    def test_dark_stack_quality_prefers_compact_dark_center(self):
        size = 31
        bright = np.full((size, size), 120.0, dtype=np.float32)
        dark = bright.copy()
        yy, xx = np.mgrid[:size, :size]
        rr = np.hypot(xx - (size - 1) / 2.0, yy - (size - 1) / 2.0)
        dark[rr <= 3.0] = 40.0
        flat_q = align.stack_quality([bright] * 3, size)["q"]
        dark_q = align.stack_quality([dark] * 3, size)["q"]
        self.assertGreater(dark_q, flat_q)

    def test_pairwise_summary_counts_same_frame_positive_over_negative(self):
        rows = [
            {"clip": "c", "frame": "1", "hard_label": "1", "score": "0.9"},
            {"clip": "c", "frame": "1", "hard_label": "0", "score": "0.2"},
            {"clip": "c", "frame": "2", "hard_label": "1", "score": "0.1"},
            {"clip": "c", "frame": "2", "hard_label": "0", "score": "0.3"},
        ]
        wins, total, rate = align.pairwise_summary(rows, "score")
        self.assertEqual((wins, total), (1, 2))
        self.assertEqual(rate, 0.5)

    def test_control_offsets_are_deterministic_annulus_points(self):
        offsets = augment.sample_control_offsets(radius=10.0, count=8)
        self.assertEqual(len(offsets), 8)
        self.assertAlmostEqual(offsets[0][0], 10.0)
        self.assertAlmostEqual(offsets[0][1], 0.0)
        self.assertTrue(all((x * x + y * y) ** 0.5 >= 9.9 for x, y in offsets))

    def test_gain_zscore_uses_robust_local_controls(self):
        z, med, sigma = augment.gain_zscore(3.0, [1.0, 1.0, 1.0, 9.0])
        self.assertEqual(med, 1.0)
        self.assertGreaterEqual(sigma, 0.25)
        self.assertGreater(z, 0.0)

    def test_matched_controls_prefer_same_router_candidates(self):
        row = {
            "x": "10",
            "y": "10",
            "w": "3",
            "h": "3",
            "cand_router_state": "surface_backed",
            "cand_texture": "0.5",
            "cand_sky_like": "0.1",
            "cand_line_context": "0.2",
            "track_id": "a",
        }
        same = {
            **row,
            "x": "30",
            "track_id": "b",
        }
        different = {
            **row,
            "x": "40",
            "cand_router_state": "clean_sky",
            "track_id": "c",
        }

        offsets, matched = augment.matched_control_offsets(row, [row, same, different], [(50.0, 0.0)], 4)

        self.assertEqual(matched, 1)
        self.assertIn((20.0, 0.0), offsets)
        self.assertNotIn((30.0, 0.0), offsets)

    def test_low_matched_control_count_shrinks_gain(self):
        self.assertAlmostEqual(augment.shrink_low_control_gain(3.0, matched_count=0), 0.0)
        self.assertAlmostEqual(augment.shrink_low_control_gain(3.0, matched_count=3), 1.5)
        self.assertAlmostEqual(augment.shrink_low_control_gain(3.0, matched_count=6), 3.0)

    def test_load_rows_can_filter_to_reviewed_frames(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "top.csv"
            path.write_text(
                "frame,rank,x,y,w,h\n"
                "1,1,0,0,2,2\n"
                "2,1,0,0,2,2\n"
                "2,3,0,0,2,2\n"
            )
            rows = augment.load_rows(path, "clip-a", max_rank=2, frame_min=-1, frame_max=-1, frame_filter={2})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], "2")
        self.assertEqual(rows[0]["clip"], "clip-a")


if __name__ == "__main__":
    unittest.main()
