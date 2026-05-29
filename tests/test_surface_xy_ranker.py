import unittest

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


if __name__ == "__main__":
    unittest.main()
