import unittest

import numpy as np

from scripts import train_acquisition_null_ranker as null_ranker


class ScoreColumnModel:
    def predict_proba(self, x):
        scores = x[:, 0]
        return np.column_stack([1.0 - scores, scores])


class AcquisitionNullRankerTests(unittest.TestCase):
    def test_apply_score_threshold_turns_low_score_visible_into_miss(self):
        rows = [
            {
                "model": "logistic",
                "clip": "clip-a",
                "frame": 10,
                "visible": True,
                "selected": True,
                "score": 0.2,
                "strict_hit": True,
                "loose_hit": True,
                "all_correct": True,
                "rank": "1",
            }
        ]

        out = null_ranker.apply_score_threshold(rows, "nested_logistic", 0.5)

        self.assertFalse(out[0]["selected"])
        self.assertFalse(out[0]["strict_hit"])
        self.assertFalse(out[0]["all_correct"])
        self.assertEqual(out[0]["rank"], "")
        self.assertEqual(out[0]["model"], "nested_logistic")

    def test_apply_score_threshold_turns_low_score_empty_into_correct_no_select(self):
        rows = [
            {
                "model": "logistic",
                "clip": "clip-a",
                "frame": 10,
                "visible": False,
                "selected": True,
                "score": 0.2,
                "strict_hit": False,
                "loose_hit": False,
                "no_target_correct": False,
                "all_correct": False,
                "rank": "1",
            }
        ]

        out = null_ranker.apply_score_threshold(rows, "nested_logistic", 0.5)

        self.assertFalse(out[0]["selected"])
        self.assertTrue(out[0]["no_target_correct"])
        self.assertTrue(out[0]["all_correct"])

    def test_apply_score_threshold_keeps_high_score_choice(self):
        rows = [
            {
                "model": "logistic",
                "clip": "clip-a",
                "frame": 10,
                "visible": True,
                "selected": True,
                "score": 0.9,
                "strict_hit": True,
                "loose_hit": True,
                "all_correct": True,
                "rank": "3",
            }
        ]

        out = null_ranker.apply_score_threshold(rows, "nested_logistic", 0.5)

        self.assertTrue(out[0]["selected"])
        self.assertTrue(out[0]["strict_hit"])
        self.assertEqual(out[0]["rank"], "3")

    def test_gate_selected_scores_only_detector_selected_row(self):
        labels = [
            {
                "clip": "clip-a",
                "frame": "1",
                "visible": "1",
                "det_x": "0",
                "det_y": "0",
                "det_w": "4",
                "det_h": "4",
            }
        ]
        top_by_clip = {
            "clip-a": {
                1: [
                    {"rank": "1", "selected": "1", "x": "0", "y": "0", "w": "4", "h": "4", "accept_score": "0.4"},
                    {"rank": "2", "selected": "0", "x": "0", "y": "0", "w": "4", "h": "4", "accept_score": "0.9"},
                ]
            }
        }

        gated = null_ranker.score_frames(
            labels,
            top_by_clip,
            "model",
            ScoreColumnModel(),
            ["accept_score"],
            [],
            threshold=0.5,
            strict_tol=8.0,
            loose_tol=16.0,
            decision_mode="gate_selected",
        )
        selected_best = null_ranker.score_frames(
            labels,
            top_by_clip,
            "model",
            ScoreColumnModel(),
            ["accept_score"],
            [],
            threshold=0.5,
            strict_tol=8.0,
            loose_tol=16.0,
            decision_mode="select_best",
        )

        self.assertFalse(gated[0]["selected"])
        self.assertEqual(gated[0]["rank"], "")
        self.assertTrue(selected_best[0]["selected"])
        self.assertEqual(selected_best[0]["rank"], "2")

    def test_frame_zero_is_not_rewritten_to_negative_one(self):
        labels = [
            {
                "clip": "clip-a",
                "frame": "0",
                "visible": "0",
            }
        ]
        top_by_clip = {
            "clip-a": {
                0: [
                    {"rank": "1", "selected": "1", "x": "0", "y": "0", "w": "4", "h": "4", "accept_score": "0.8"}
                ]
            }
        }

        rows = null_ranker.score_frames(
            labels,
            top_by_clip,
            "model",
            ScoreColumnModel(),
            ["accept_score"],
            [],
            threshold=0.5,
            strict_tol=8.0,
            loose_tol=16.0,
            decision_mode="gate_selected",
        )

        self.assertEqual(rows[0]["frame"], 0)
        self.assertEqual(rows[0]["x"], "0")


if __name__ == "__main__":
    unittest.main()
