import unittest

import numpy as np

from scripts import profile_tube_alignment_features as align


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


if __name__ == "__main__":
    unittest.main()
