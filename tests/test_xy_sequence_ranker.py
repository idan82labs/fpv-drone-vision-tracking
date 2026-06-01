import unittest

from scripts import evaluate_xy_sequence_ranker as seq


class XYSequenceRankerTests(unittest.TestCase):
    def test_viterbi_legacy_backfills_unreachable_late_restart_prefix(self):
        by_frame = {
            1: [{"frame": "1", "x": "0", "y": "0", "w": "4", "h": "4", "learned_score": 9.0}],
            2: [{"frame": "2", "x": "100", "y": "100", "w": "4", "h": "4", "learned_score": 20.0}],
            3: [{"frame": "3", "x": "102", "y": "100", "w": "4", "h": "4", "learned_score": 20.0}],
        }

        selected = seq.viterbi_select(by_frame, max_jump_px=5.0, transition_weight=1.5)

        self.assertEqual(selected[1]["x"], "0")
        self.assertEqual(selected[2]["x"], "100")
        self.assertEqual(selected[3]["x"], "102")

    def test_viterbi_no_backfill_omits_unreachable_late_restart_prefix(self):
        by_frame = {
            1: [{"frame": "1", "x": "0", "y": "0", "w": "4", "h": "4", "learned_score": 9.0}],
            2: [{"frame": "2", "x": "100", "y": "100", "w": "4", "h": "4", "learned_score": 20.0}],
            3: [{"frame": "3", "x": "102", "y": "100", "w": "4", "h": "4", "learned_score": 20.0}],
        }

        selected = seq.viterbi_select(
            by_frame,
            max_jump_px=5.0,
            transition_weight=1.5,
            backfill_unreachable=False,
        )

        self.assertNotIn(1, selected)
        self.assertEqual(selected[2]["x"], "100")
        self.assertEqual(selected[3]["x"], "102")

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

    def test_constant_velocity_selector_rejects_smooth_score_lure(self):
        by_frame = {
            1: [{"frame": "1", "x": "0", "y": "10", "w": "4", "h": "4", "learned_score": 0.9}],
            2: [{"frame": "2", "x": "10", "y": "10", "w": "4", "h": "4", "learned_score": 0.9}],
            3: [
                {"frame": "3", "x": "20", "y": "10", "w": "4", "h": "4", "learned_score": 0.6, "rank": "2"},
                {"frame": "3", "x": "10", "y": "10", "w": "4", "h": "4", "learned_score": 0.99, "rank": "1"},
            ],
        }

        selected = seq.viterbi_select_constant_velocity(
            by_frame,
            max_jump_px=20.0,
            transition_weight=0.0,
            size_jump_weight=0.0,
            accel_weight=10.0,
            state_beam=32,
        )

        self.assertEqual(selected[3]["x"], "20")

    def test_size_scaled_jump_allows_larger_close_candidate_motion(self):
        by_frame = {
            1: [{"frame": "1", "x": "10", "y": "10", "w": "20", "h": "20", "learned_score": 0.9}],
            2: [
                {"frame": "2", "x": "23", "y": "10", "w": "20", "h": "20", "learned_score": 0.9},
                {"frame": "2", "x": "18", "y": "18", "w": "4", "h": "4", "learned_score": 0.85},
            ],
        }

        strict = seq.viterbi_select(by_frame, max_jump_px=4.0, transition_weight=0.0, size_jump_weight=0.0)
        scaled = seq.viterbi_select(by_frame, max_jump_px=4.0, transition_weight=0.0, size_jump_weight=0.5)

        self.assertEqual(strict[2]["x"], "18")
        self.assertEqual(scaled[1]["x"], "10")
        self.assertEqual(scaled[2]["x"], "23")


if __name__ == "__main__":
    unittest.main()
