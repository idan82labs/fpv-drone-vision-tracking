import unittest

from scripts import apply_surface_sequence_selector as selector
from scripts import evaluate_xy_sequence_ranker as seq


class ApplySurfaceSequenceSelectorTests(unittest.TestCase):
    def test_group_by_frame_preserves_zero_frame(self):
        rows = [
            {"frame": "0", "rank": "1", "learned_score": 0.2},
            {"frame": "1", "rank": "1", "learned_score": 0.3},
        ]

        grouped = selector.group_by_frame(rows)

        self.assertEqual(set(grouped), {0, 1})
        self.assertEqual(grouped[0][0]["rank"], "1")

    def test_output_rows_applies_threshold_without_dropping_frames(self):
        scored = [
            {"frame": "0", "rank": "1", "x": "10", "y": "20", "w": "4", "h": "4", "learned_score": 0.4},
            {"frame": "1", "rank": "2", "x": "12", "y": "20", "w": "4", "h": "4", "learned_score": 0.8},
        ]
        selected = {0: scored[0], 1: scored[1]}

        rows = selector.output_rows("clip-a", scored, selected, threshold=0.5)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["frame"], 0)
        self.assertEqual(rows[0]["selected"], 0)
        self.assertEqual(rows[1]["selected"], 1)
        self.assertEqual(rows[1]["x"], "12")

    def test_rolling_viterbi_can_restart_after_false_background_lock(self):
        # A full-video path stays on the early false branch because the true
        # target appears too far away to connect. The rolling selector should
        # recover once the true target has enough local support.
        by_frame = {}
        for frame in range(10):
            by_frame[frame] = [
                {"frame": frame, "x": 0, "y": 0, "w": 4, "h": 4, "learned_score": 0.9},
            ]
        for frame in range(10, 20):
            by_frame[frame] = [
                {"frame": frame, "x": 0, "y": 0, "w": 4, "h": 4, "learned_score": 0.82},
                {"frame": frame, "x": 100 + frame - 10, "y": 0, "w": 4, "h": 4, "learned_score": 0.88},
            ]

        full = seq.viterbi_select(by_frame, max_jump_px=10, transition_weight=0.5)
        rolling = seq.rolling_viterbi_select(by_frame, max_jump_px=10, transition_weight=0.5, sequence_window=5)

        self.assertEqual(full[19]["x"], 0)
        self.assertEqual(rolling[19]["x"], 109)

    def test_hysteresis_gate_requires_acquire_then_keeps_lower_scores(self):
        selected = {
            0: {"frame": "0", "x": "0", "y": "0", "w": "4", "h": "4", "learned_score": 0.83},
            1: {"frame": "1", "x": "1", "y": "0", "w": "4", "h": "4", "learned_score": 0.91},
            2: {"frame": "2", "x": "2", "y": "0", "w": "4", "h": "4", "learned_score": 0.82},
            3: {"frame": "3", "x": "40", "y": "0", "w": "4", "h": "4", "learned_score": 0.95},
            4: {"frame": "4", "x": "41", "y": "0", "w": "4", "h": "4", "learned_score": 0.95},
        }

        gated = selector.apply_hysteresis_gate(
            selected,
            acquire_threshold=0.9,
            keep_threshold=0.8,
            max_jump_px=5,
            lost_patience=0,
        )

        self.assertNotIn(0, gated)
        self.assertIn(1, gated)
        self.assertIn(2, gated)
        self.assertNotIn(3, gated)
        self.assertIn(4, gated)

    def test_clba_adjustment_rewrites_learned_score_and_preserves_base(self):
        rows = [
            {
                "learned_score": 0.8,
                "clba_bg_q": "5",
                "clba_bg_static_likelihood": "5",
            },
            {
                "learned_score": 0.75,
                "clba_gain_norm": "2",
                "clba_path_bg_dist_mean": "12",
                "clba_target_q": "2",
            },
        ]

        adjusted = selector.apply_clba_adjustment(
            rows,
            selector.clba_adjust.Weights(gain=0.3, path=0.1, target_q=0.05, bg=0.3),
        )

        self.assertEqual(adjusted[0]["base_learned_score"], 0.8)
        self.assertEqual(adjusted[1]["base_learned_score"], 0.75)
        self.assertLess(adjusted[0]["learned_score"], adjusted[1]["learned_score"])
        self.assertIn("clba_adjusted_score", adjusted[1])

    def test_hmm_candidate_evidence_score_modes(self):
        row = {"learned_score": 0.75}

        self.assertAlmostEqual(
            selector.candidate_evidence(row, "logit", 2.0, 0.5, 0.0),
            2.0 * selector.logit_score(0.75),
        )
        self.assertAlmostEqual(selector.candidate_evidence(row, "centered", 2.0, 0.5, 0.0), 0.5)
        self.assertAlmostEqual(selector.candidate_evidence(row, "raw", 2.0, 0.5, 0.0), 1.5)

    def test_static_lock_risk_uses_rank_limited_clba_median(self):
        rows = [
            {"rank": "1", "clba_bg_static_likelihood": "3", "clba_target_likelihood": "1"},
            {"rank": "2", "clba_bg_static_likelihood": "-5", "clba_target_likelihood": "1"},
            {"rank": "9", "clba_bg_static_likelihood": "100", "clba_target_likelihood": "0"},
            {"rank": "1", "clba_bg_static_likelihood": "", "clba_target_likelihood": "0"},
        ]

        self.assertEqual(selector.frame_static_lock_risk(rows, max_rank=1), 2.0)
        self.assertEqual(selector.frame_static_lock_risk(rows, max_rank=2), 2.0)

    def test_rolling_static_lock_risk_is_causal(self):
        by_frame = {
            10: [{"rank": "1", "clba_bg_static_likelihood": "2", "clba_target_likelihood": "0"}],
            11: [{"rank": "1", "clba_bg_static_likelihood": "-2", "clba_target_likelihood": "0"}],
            12: [{"rank": "1", "clba_bg_static_likelihood": "4", "clba_target_likelihood": "0"}],
        }

        risks = selector.rolling_static_lock_risk(by_frame, window=2, max_rank=1)

        self.assertEqual(risks[10], 2.0)
        self.assertEqual(risks[11], 2.0)
        self.assertEqual(risks[12], 4.0)


if __name__ == "__main__":
    unittest.main()
