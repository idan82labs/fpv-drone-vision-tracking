import unittest

from scripts import evaluate_xy_sequence_ranker as seq


class XYSequenceRankerTests(unittest.TestCase):
    def test_viterbi_rejects_single_frame_jump(self):
        by_frame = {
            1: [
                {"frame": "1", "x": "10", "y": "10", "w": "4", "h": "4", "learned_score": 0.90, "rank": "1"},
                {"frame": "1", "x": "80", "y": "80", "w": "4", "h": "4", "learned_score": 0.10, "rank": "2"},
            ],
            2: [
                {"frame": "2", "x": "12", "y": "10", "w": "4", "h": "4", "learned_score": 0.70, "rank": "2"},
                {"frame": "2", "x": "80", "y": "80", "w": "4", "h": "4", "learned_score": 0.99, "rank": "1"},
            ],
            3: [
                {"frame": "3", "x": "14", "y": "10", "w": "4", "h": "4", "learned_score": 0.90, "rank": "1"},
                {"frame": "3", "x": "80", "y": "80", "w": "4", "h": "4", "learned_score": 0.10, "rank": "2"},
            ],
        }

        selected = seq.viterbi_select(by_frame, max_jump_px=8.0, transition_weight=0.25)

        self.assertEqual(selected[1]["x"], "10")
        self.assertEqual(selected[2]["x"], "12")
        self.assertEqual(selected[3]["x"], "14")

    def test_summarize_selection_counts_strict_and_loose(self):
        labels = [
            {"frame": "1", "det_x": "10", "det_y": "10", "det_w": "4", "det_h": "4"},
            {"frame": "2", "det_x": "20", "det_y": "20", "det_w": "4", "det_h": "4"},
        ]
        selected = {
            1: {"x": "10", "y": "10", "w": "4", "h": "4"},
            2: {"x": "28", "y": "20", "w": "4", "h": "4"},
        }

        summary = seq.summarize_selection(labels, selected, center_tol=4.0, loose_tol=10.0, name="test")

        self.assertEqual(summary["strict_hit"], 1)
        self.assertEqual(summary["loose_hit"], 2)


if __name__ == "__main__":
    unittest.main()
