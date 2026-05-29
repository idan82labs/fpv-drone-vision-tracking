import unittest

from scripts import evaluate_router_null_calibration as calib


class RouterNullCalibrationTests(unittest.TestCase):
    def test_bucket_prefers_candidate_router_state(self):
        self.assertEqual(calib.bucket_for_row({"cand_router_state": "surface_backed"}), "surface")
        self.assertEqual(calib.bucket_for_row({"cand_router_state": "line_attached"}), "line")
        self.assertEqual(calib.bucket_for_row({"cand_router_state": "boundary_mixed"}), "boundary")
        self.assertEqual(calib.bucket_for_row({"cand_router_state": "clean_sky"}), "clean_sky")

    def test_bucket_falls_back_to_tube_router_rates(self):
        row = {
            "cand_router_state": "unrouted",
            "tube_router_surface_backed_rate": "0.2",
            "tube_router_clean_sky_rate": "0.7",
            "tube_router_boundary_rate": "0.1",
            "tube_router_line_attached_rate": "0.0",
        }

        self.assertEqual(calib.bucket_for_row(row), "clean_sky")

    def test_thresholds_fall_back_to_global_when_bucket_is_sparse(self):
        nulls = {
            "global": [0.1, 0.2, 0.9],
            "surface": [0.8],
            "clean_sky": [0.1, 0.2, 0.3],
        }

        out = calib.thresholds_from_nulls(nulls, 0.5, min_bucket_null_samples=2, min_threshold=0.0, margin=0.01)

        self.assertAlmostEqual(out["global"], 0.21)
        self.assertAlmostEqual(out["surface"], out["global"])
        self.assertAlmostEqual(out["clean_sky"], 0.21)

    def test_thresholds_respect_minimum_floor(self):
        nulls = {
            "global": [0.01, 0.02, 0.03],
            "clean_sky": [0.001, 0.002, 0.003],
        }

        out = calib.thresholds_from_nulls(nulls, 0.5, min_bucket_null_samples=2, min_threshold=0.05, margin=0.0)

        self.assertEqual(out["global"], 0.05)
        self.assertEqual(out["clean_sky"], 0.05)

    def test_evaluate_clip_selects_only_candidates_above_their_bucket_threshold(self):
        labels = {
            ("clip", 1): {"visible": False, "bbox": None},
            ("clip", 2): {"visible": True, "bbox": (10.0, 10.0, 3.0, 3.0)},
        }
        cands = {
            ("clip", 1): [{"_score": 0.4, "_bucket": "surface", "rank": "1", "x": "0", "y": "0", "w": "3", "h": "3"}],
            ("clip", 2): [{"_score": 0.9, "_bucket": "clean_sky", "rank": "1", "x": "10", "y": "10", "w": "3", "h": "3"}],
        }

        rows = calib.evaluate_clip(
            labels,
            cands,
            "clip",
            {"surface": 0.5, "clean_sky": 0.5, "global": 0.5},
            strict_tol=8.0,
            loose_tol=16.0,
            model="m",
            quantile=0.9,
        )

        self.assertEqual(rows[0]["selected"], 0)
        self.assertEqual(rows[1]["selected"], 1)
        self.assertEqual(rows[1]["strict_hit"], 1)


if __name__ == "__main__":
    unittest.main()
