import unittest

from scripts import augment_top_tubes_competition_features as comp


class CompetitionFeatureTests(unittest.TestCase):
    def test_target_margin_is_normalized_against_same_frame_controls(self):
        rows = [
            {
                "frame": "10",
                "rank": "1",
                "x": "10",
                "y": "10",
                "w": "4",
                "h": "4",
                "clba_target_likelihood": "4.0",
                "clba_bg_static_likelihood": "0.0",
                "clba_attached_likelihood": "0.0",
                "clba_gain_norm": "2.0",
                "learned_score": "0.8",
                "cand_texture": "80",
            },
            {
                "frame": "10",
                "rank": "2",
                "x": "22",
                "y": "10",
                "w": "4",
                "h": "4",
                "clba_target_likelihood": "0.0",
                "clba_bg_static_likelihood": "3.0",
                "clba_attached_likelihood": "0.0",
                "clba_gain_norm": "-1.0",
                "learned_score": "0.7",
                "cand_texture": "78",
            },
            {
                "frame": "10",
                "rank": "3",
                "x": "40",
                "y": "10",
                "w": "4",
                "h": "4",
                "clba_target_likelihood": "0.2",
                "clba_bg_static_likelihood": "2.0",
                "clba_attached_likelihood": "0.0",
                "clba_gain_norm": "-0.5",
                "learned_score": "0.6",
                "cand_texture": "81",
            },
        ]

        out = comp.add_competition_features(
            rows,
            max_rank=80,
            near_radius_px=36,
            min_context_controls=1,
            sigma_floor=0.25,
        )

        self.assertEqual(out[0]["comp_context_bucket"], "surface_texture")
        self.assertGreater(float(out[0]["comp_target_margin_context_z"]), 0.0)
        self.assertGreater(float(out[0]["comp_target_margin_near_best_margin"]), 0.0)
        self.assertLess(float(out[1]["comp_target_margin_context_z"]), 0.0)

    def test_context_bucket_marks_attached_linear_before_surface_texture(self):
        row = {
            "cand_texture": "100",
            "cand_attached_support": "8",
            "cand_line_context": "0.1",
        }

        self.assertEqual(comp.context_bucket(row), "attached_linear")


if __name__ == "__main__":
    unittest.main()
