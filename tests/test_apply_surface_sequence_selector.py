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

    def test_joint_candidate_terms_compete_with_static_explanation(self):
        target = {
            "rank": "1",
            "learned_score": 0.9,
            "clba_target_likelihood": "4",
            "clba_gain_norm": "3",
            "clba_bg_static_likelihood": "0",
            "clba_attached_likelihood": "0",
        }
        static = {
            "rank": "1",
            "learned_score": 0.9,
            "clba_target_likelihood": "0",
            "clba_gain_norm": "0",
            "clba_bg_static_likelihood": "6",
            "clba_attached_likelihood": "0",
        }

        target_terms = selector.joint_candidate_terms(
            target, "logit", 1.0, 0.5, 0.35, 0.55, 0.03, 0.75, 0.7, 0.08, 0.0, 0.15, 0.1
        )
        static_terms = selector.joint_candidate_terms(
            static, "logit", 1.0, 0.5, 0.35, 0.55, 0.03, 0.75, 0.7, 0.08, 0.0, 0.15, 0.1
        )

        self.assertGreater(target_terms["target_llr"], target_terms["static_llr"])
        self.assertGreater(static_terms["static_llr"], static_terms["target_llr"])

    def test_joint_hmm_static_lock_can_release_to_fresh_target(self):
        static = {
            "frame": "0",
            "rank": "1",
            "x": "0",
            "y": "0",
            "w": "4",
            "h": "4",
            "learned_score": 0.95,
            "clba_target_likelihood": "0",
            "clba_gain_norm": "0",
            "clba_bg_static_likelihood": "7",
            "clba_attached_likelihood": "0",
        }
        target1 = {
            "frame": "1",
            "rank": "1",
            "x": "80",
            "y": "0",
            "w": "4",
            "h": "4",
            "learned_score": 0.95,
            "clba_target_likelihood": "5",
            "clba_gain_norm": "3",
            "clba_path_bg_dist_mean": "8",
            "clba_bg_static_likelihood": "0",
            "clba_attached_likelihood": "0",
        }
        target2 = dict(target1, frame="2", x="82")
        by_frame = {0: [static], 1: [target1], 2: [target2]}

        selected = selector.select_with_joint_hmm(
            by_frame,
            max_jump_px=8,
            transition_weight=0.2,
            size_jump_weight=0.0,
            beam=32,
            score_mode="logit",
            score_scale=1.0,
            score_center=0.5,
            birth_penalty=0.3,
            track_bonus=0.2,
            miss_penalty=0.4,
            coast_penalty=0.1,
            reacquire_penalty=0.2,
            max_coast=1,
            acquire_hits=2,
            target_weight=0.35,
            gain_weight=0.55,
            path_weight=0.03,
            static_weight=0.75,
            attached_weight=0.7,
            rank_weight=0.08,
            null_bias=0.0,
            static_bias=0.15,
            attached_bias=0.1,
            lock_margin=0.1,
            lock_penalty=0.25,
            release_penalty=0.1,
            quarantine_px=12.0,
            quarantine_frames=10,
            quarantine_penalty=2.5,
        )

        self.assertNotIn(0, selected)
        self.assertIn(2, selected)
        self.assertEqual(selected[2]["x"], "82")

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

    def test_adaptive_hmm_frame_mask_requires_sustained_risk(self):
        risks = {1: 0.2, 2: 1.2, 3: 0.4, 4: 1.3, 5: 1.4, 6: 0.8, 7: 0.2}

        mask = selector.adaptive_hmm_frame_mask(
            risks,
            acquire_threshold=1.0,
            keep_threshold=0.5,
            hits_required=2,
            release_required=2,
        )

        self.assertFalse(mask[2])
        self.assertFalse(mask[4])
        self.assertTrue(mask[5])
        self.assertTrue(mask[6])
        self.assertTrue(mask[7])


if __name__ == "__main__":
    unittest.main()
