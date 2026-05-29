import unittest

import numpy as np

from scripts import train_crop_stack_verifier as crop


class CropStackVerifierTests(unittest.TestCase):
    def test_normalize_crop_centers_by_median_mad(self):
        arr = np.full((5, 5), 10.0, dtype=np.float32)
        arr[2, 2] = 1.0
        norm = crop.normalize_crop(arr)

        self.assertLess(norm[2, 2], 0.0)
        self.assertAlmostEqual(float(np.median(norm)), 0.0)

    def test_pairwise_summary_counts_positive_over_false_competitor(self):
        rows = [
            {"clip": "c", "frame": "1", "hard_label": "1", "crop_stack_score": "0.8"},
            {"clip": "c", "frame": "1", "hard_label": "0", "crop_stack_score": "0.2"},
            {"clip": "c", "frame": "2", "hard_label": "1", "crop_stack_score": "0.1"},
            {"clip": "c", "frame": "2", "hard_label": "0", "crop_stack_score": "0.3"},
            {"clip": "c", "frame": "3", "hard_label": "0", "crop_stack_score": "0.9"},
        ]

        summary = crop.pairwise_summary(rows, "crop_stack_score")

        self.assertEqual(summary["pairwise_wins"], 1)
        self.assertEqual(summary["pairwise_total"], 2)
        self.assertEqual(summary["positive_frames_with_negatives"], 2)
        self.assertEqual(summary["pairwise_win_rate"], 0.5)

    def test_build_examples_prefers_selected_false_competitor(self):
        label = {
            "clip": "c",
            "frame": "10",
            "det_x": "10",
            "det_y": "10",
            "det_w": "4",
            "det_h": "4",
        }
        rows = [
            {"rank": "4", "x": "10", "y": "10", "w": "4", "h": "4"},
            {"rank": "1", "x": "100", "y": "100", "w": "4", "h": "4", "selected": "0", "verified_score": "99"},
            {"rank": "8", "x": "120", "y": "100", "w": "4", "h": "4", "selected": "1", "verified_score": "1"},
        ]

        scored = [(crop.dist_to_label(row, label), row) for row in rows]
        negatives = [(d, row) for d, row in scored if d >= 24.0]
        negatives_sorted = sorted(
            negatives,
            key=lambda item: (
                str(item[1].get("selected", "0")) != "1",
                -crop.safe_float(item[1].get("learned_score"), crop.safe_float(item[1].get("verified_score"), crop.safe_float(item[1].get("score")))),
                crop.safe_int(item[1].get("rank"), 999999),
            ),
        )

        self.assertEqual(negatives_sorted[0][1]["selected"], "1")

    def test_pairwise_training_data_builds_symmetric_differences(self):
        rows = [
            {"clip": "c", "frame": "1", "hard_label": "1"},
            {"clip": "c", "frame": "1", "hard_label": "0"},
            {"clip": "c", "frame": "2", "hard_label": "0"},
        ]
        x = np.asarray([[3.0, 1.0], [1.0, 2.0], [9.0, 9.0]], dtype=np.float32)

        diffs, labels = crop.pairwise_training_data(rows, x)

        self.assertEqual(diffs.shape, (2, 2))
        np.testing.assert_array_equal(diffs[0], np.asarray([2.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(diffs[1], np.asarray([-2.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(labels, np.asarray([1, 0], dtype=np.int32))

    def test_pairwise_model_uses_decision_function_scores(self):
        self.assertEqual(crop.score_mode_for_model("pairwise_logistic"), "decision_function")
        self.assertEqual(crop.score_mode_for_model("hist_gbdt"), "auto")


if __name__ == "__main__":
    unittest.main()
