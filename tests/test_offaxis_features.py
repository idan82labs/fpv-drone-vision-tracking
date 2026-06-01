import unittest

import numpy as np

from scripts import augment_top_tubes_offaxis_features as offaxis


class OffAxisFeatureTests(unittest.TestCase):
    def test_predict_points_accepts_3x3_homogeneous_transform(self):
        pts = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        h = np.asarray([[2.0, 0.0, 5.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        out = offaxis.predict_points(h, pts)

        np.testing.assert_allclose(out, np.asarray([[7.0, 3.0], [11.0, 7.0]], dtype=np.float32))

    def test_predict_points_accepts_2x3_affine_transform(self):
        pts = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        aff = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float32)

        out = offaxis.predict_points(aff, pts)

        np.testing.assert_allclose(out, np.asarray([[3.0, 5.0], [5.0, 7.0]], dtype=np.float32))

    def test_offaxis_score_prefers_independent_motion_over_background_aligned_residual(self):
        prev = np.asarray([[50.0, 50.0], [52.0, 50.0], [48.0, 50.0], [50.0, 52.0]], dtype=np.float32)
        h = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        pred = offaxis.predict_points(h, prev)
        independent_cur = pred + np.asarray([0.0, 3.0], dtype=np.float32)
        aligned_cur = pred + np.asarray([1.2, 0.0], dtype=np.float32)

        independent = offaxis.compute_offaxis_signals(
            prev,
            independent_cur,
            h,
            cx=52.0,
            cy=53.0,
            frame_wh=(100, 100),
            radius=8.0,
            move_min_px=0.4,
        )
        aligned = offaxis.compute_offaxis_signals(
            prev,
            aligned_cur,
            h,
            cx=53.2,
            cy=50.0,
            frame_wh=(100, 100),
            radius=8.0,
            move_min_px=0.4,
        )

        self.assertGreater(independent["offaxis_angle"], 70.0)
        self.assertLess(aligned["offaxis_angle"], 20.0)
        self.assertGreater(independent["offaxis_indep_score"], aligned["offaxis_indep_score"])

    def test_no_nearby_flow_returns_neutral_features(self):
        prev = np.asarray([[10.0, 10.0], [12.0, 10.0]], dtype=np.float32)
        cur = prev + np.asarray([2.0, 0.0], dtype=np.float32)

        feat = offaxis.compute_offaxis_signals(prev, cur, None, cx=80.0, cy=80.0, frame_wh=(100, 100), radius=4.0)

        self.assertEqual(feat["offaxis_near_count"], 0)
        self.assertEqual(feat["offaxis_mover_count"], 0)
        self.assertEqual(feat["offaxis_indep_score"], 0.0)

    def test_robust_gain_compares_against_controls(self):
        z, med, sigma = offaxis.robust_gain(1.0, [0.1, 0.12, 0.1, 4.0])

        self.assertAlmostEqual(med, 0.11, places=2)
        self.assertGreater(sigma, 0.0)
        self.assertGreater(z, 5.0)


if __name__ == "__main__":
    unittest.main()
