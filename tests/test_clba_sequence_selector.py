import unittest

from scripts import sweep_clba_sequence_selector as sweep


class ClbaSequenceSelectorTests(unittest.TestCase):
    def test_framewise_best_uses_adjusted_learned_score(self):
        rows = [
            {"frame": 1, "rank": 1, "learned_score": 0.2},
            {"frame": 1, "rank": 2, "learned_score": 0.9},
        ]

        selected = sweep.framewise_best(sweep.group_by_frame(rows))

        self.assertEqual(selected[1]["rank"], 2)

    def test_prune_by_frame_keeps_top_adjusted_scores(self):
        rows = [
            {"frame": 1, "rank": 1, "learned_score": 0.2},
            {"frame": 1, "rank": 2, "learned_score": 0.9},
            {"frame": 1, "rank": 3, "learned_score": 0.4},
        ]

        pruned = sweep.prune_by_frame(sweep.group_by_frame(rows), 2)

        self.assertEqual([r["rank"] for r in pruned[1]], [2, 3])

    def test_evaluate_selection_counts_visible_and_null_frames(self):
        labels = [
            {"frame": 1, "visible": 1, "bbox": (10.0, 10.0, 4.0, 4.0)},
            {"frame": 2, "visible": 0, "bbox": None},
        ]
        selected = {
            1: {"x": "10", "y": "10", "w": "4", "h": "4", "learned_score": 0.8},
            2: {"x": "30", "y": "30", "w": "4", "h": "4", "learned_score": 0.2},
        }

        summary, rows = sweep.evaluate_selection(labels, selected, 0.5, 8.0, 16.0)

        self.assertEqual(summary["visible_strict"], 1)
        self.assertEqual(summary["invisible_no_box"], 1)
        self.assertEqual(summary["all_frame_correct"], 2)
        self.assertEqual(rows[1]["selected"], 0)

    def test_clba_scoring_can_flip_two_candidates(self):
        rows = [
            {
                "frame": 1,
                "rank": 1,
                "score": "0.8",
                "clba_bg_q": "5",
                "clba_bg_static_likelihood": "5",
            },
            {
                "frame": 1,
                "rank": 2,
                "score": "0.75",
                "clba_gain_norm": "2",
                "clba_path_bg_dist_mean": "12",
                "clba_target_q": "2",
            },
        ]

        scored = sweep.score_rows(
            rows,
            sweep.clba_adjust.Weights(gain=0.3, path=0.1, target_q=0.05, bg=0.3),
            "score",
        )

        self.assertGreater(scored[1]["learned_score"], scored[0]["learned_score"])

    def test_hysteresis_gate_requires_acquire_and_rejects_jump(self):
        selected = {
            1: {"frame": 1, "x": "0", "y": "0", "w": "4", "h": "4", "learned_score": 0.7},
            2: {"frame": 2, "x": "1", "y": "0", "w": "4", "h": "4", "learned_score": 0.95},
            3: {"frame": 3, "x": "2", "y": "0", "w": "4", "h": "4", "learned_score": 0.75},
            4: {"frame": 4, "x": "50", "y": "0", "w": "4", "h": "4", "learned_score": 0.95},
        }

        gated = sweep.apply_hysteresis_gate(selected, 0.9, 1, 0.7, 8.0, 0)

        self.assertNotIn(1, gated)
        self.assertIn(2, gated)
        self.assertIn(3, gated)
        self.assertNotIn(4, gated)

    def test_hysteresis_gate_can_require_multiple_acquire_hits(self):
        selected = {
            1: {"frame": 1, "x": "0", "y": "0", "w": "4", "h": "4", "learned_score": 0.8},
            2: {"frame": 2, "x": "1", "y": "0", "w": "4", "h": "4", "learned_score": 0.82},
            3: {"frame": 3, "x": "2", "y": "0", "w": "4", "h": "4", "learned_score": 0.7},
        }

        gated = sweep.apply_hysteresis_gate(selected, 0.75, 2, 0.65, 8.0, 0)

        self.assertNotIn(1, gated)
        self.assertIn(2, gated)
        self.assertIn(3, gated)


if __name__ == "__main__":
    unittest.main()
