import unittest

from scripts import sweep_clba_score_adjustment as sweep


class ClbaScoreAdjustmentTests(unittest.TestCase):
    def test_background_and_attached_terms_lower_adjusted_score(self):
        clean = {
            "score": "0.7",
            "clba_gain_norm": "1.5",
            "clba_path_bg_dist_mean": "10",
            "clba_target_q": "1.0",
            "clba_bg_q": "0.1",
            "clba_bg_static_likelihood": "0.1",
            "clba_attached_likelihood": "0.0",
        }
        clutter = dict(clean)
        clutter.update(
            {
                "clba_bg_q": "4.0",
                "clba_bg_static_likelihood": "4.0",
                "clba_attached_likelihood": "4.0",
                "cand_line_context": "1.0",
                "cand_attached_support": "20",
            }
        )
        weights = sweep.Weights(gain=0.2, path=0.1, target_q=0.05, bg=0.2, attached=0.2)

        self.assertGreater(
            sweep.adjusted_score(clean, weights, "score"),
            sweep.adjusted_score(clutter, weights, "score"),
        )

    def test_candidate_map_uses_adjusted_score_not_raw_rank(self):
        rows = [
            {
                "frame": "1",
                "rank": "1",
                "score": "0.8",
                "x": "0",
                "y": "0",
                "w": "3",
                "h": "3",
                "clba_bg_q": "5",
                "clba_bg_static_likelihood": "5",
            },
            {
                "frame": "1",
                "rank": "2",
                "score": "0.75",
                "x": "20",
                "y": "0",
                "w": "3",
                "h": "3",
                "clba_gain_norm": "2",
                "clba_path_bg_dist_mean": "12",
                "clba_target_q": "2",
            },
        ]
        weights = sweep.Weights(gain=0.3, path=0.1, target_q=0.05, bg=0.3)

        cands = sweep.candidate_map_from_rows(rows, weights, "score")

        self.assertEqual(cands[1].rank, 2)
        self.assertEqual(cands[1].bbox[0], 20.0)

    def test_best_per_frame_rows_keeps_highest_adjusted_candidate(self):
        rows = [
            {"frame": "2", "rank": "1", "adjusted_score": "0.1"},
            {"frame": "2", "rank": "2", "adjusted_score": "0.9"},
            {"frame": "3", "rank": "1", "adjusted_score": "0.3"},
        ]

        best = sweep.best_per_frame_rows(rows)

        self.assertEqual([r["rank"] for r in best], ["2", "1"])


if __name__ == "__main__":
    unittest.main()
