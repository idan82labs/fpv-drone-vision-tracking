import unittest

from scripts import evaluate_explicit_state_selector as explicit


class ExplicitStateSelectorTests(unittest.TestCase):
    def cand(self, frame=1, x=0, y=0, target=2.0, static=0.0, attached=0.0):
        return explicit.Candidate(
            frame=frame,
            rank=1,
            bbox=(float(x), float(y), 3.0, 3.0),
            target_obs=target,
            static_obs=static,
            attached_obs=attached,
            raw_score=target,
            row={},
        )

    def test_acquire_then_track_emits_only_after_required_hits(self):
        labels = {
            1: explicit.Label(True, (0.0, 0.0, 3.0, 3.0)),
            2: explicit.Label(True, (1.0, 0.0, 3.0, 3.0)),
        }
        candidates = {1: [self.cand(1, 0, 0)], 2: [self.cand(2, 1, 0)]}
        summary, rows = explicit.evaluate_selector(
            labels,
            candidates,
            acquire_threshold=0.5,
            track_threshold=0.4,
            acquire_hits=2,
            max_misses=0,
            max_jump_px=12.0,
            clutter_margin=0.4,
            strict_tol_px=8.0,
            loose_tol_px=16.0,
            quarantine_px=10.0,
            quarantine_frames=5,
            motion_weight=0.2,
            beam_width=24,
            state_beam=6,
        )

        self.assertEqual(rows[0]["state"], "P")
        self.assertEqual(rows[0]["selected"], 0)
        self.assertEqual(rows[1]["state"], "T")
        self.assertEqual(summary["visible_strict"], 1)

    def test_attached_lock_quarantines_same_branch_and_allows_birth_elsewhere(self):
        labels = {
            1: explicit.Label(False, None),
            2: explicit.Label(True, (50.0, 0.0, 3.0, 3.0)),
            3: explicit.Label(True, (51.0, 0.0, 3.0, 3.0)),
        }
        candidates = {
            1: [self.cand(1, 0, 0, target=0.2, attached=2.2)],
            2: [self.cand(2, 0, 0, target=3.0, attached=4.2), self.cand(2, 50, 0, target=2.0)],
            3: [self.cand(3, 51, 0, target=2.0)],
        }

        summary, rows = explicit.evaluate_selector(
            labels,
            candidates,
            acquire_threshold=0.5,
            track_threshold=0.4,
            acquire_hits=1,
            max_misses=0,
            max_jump_px=12.0,
            clutter_margin=0.4,
            strict_tol_px=8.0,
            loose_tol_px=16.0,
            quarantine_px=12.0,
            quarantine_frames=5,
            motion_weight=0.2,
            beam_width=32,
            state_beam=8,
        )

        self.assertIn(rows[0]["state"], {"A", "E"})
        self.assertEqual(rows[1]["selected"], 1)
        self.assertEqual(rows[1]["rank"], 1)
        self.assertEqual(summary["visible_strict"], 2)

    def test_clutter_lock_does_not_outscore_absent_by_reward_itself(self):
        hyp = explicit.Hypothesis("A", 0.0)
        cand = self.cand(1, 0, 0, target=0.1, attached=5.0)

        out = explicit.step_hypotheses(
            [hyp],
            [cand],
            acquire_threshold=0.5,
            track_threshold=0.4,
            acquire_hits=1,
            max_misses=0,
            max_jump_px=12.0,
            clutter_margin=0.4,
            quarantine_px=10.0,
            quarantine_frames=5,
            motion_weight=0.2,
            beam_width=16,
            state_beam=8,
        )

        absent = max(h.score for h in out if h.state == "A")
        clutter = max(h.score for h in out if h.state == "E")
        self.assertLessEqual(clutter, absent)

    def test_candidate_observations_include_static_and_attached_alternatives(self):
        row = {
            "rank": "2",
            "score": "20",
            "clba_gain_norm": "0.5",
            "clba_target_q": "1.0",
            "clba_bg_q": "2.0",
            "clba_path_bg_dist_mean": "1.0",
            "cand_line_context": "0.8",
            "cand_attached_support": "12",
            "tube_log_cand_density": "3",
        }

        target, static, attached, raw = explicit.candidate_observations(
            row,
            score_column="score",
            score_weight=0.5,
            clba_weight=0.5,
            path_weight=0.2,
            static_weight=0.7,
            attached_weight=0.7,
            rank_weight=0.1,
        )

        self.assertEqual(raw, 20.0)
        self.assertGreater(target, 0.0)
        self.assertGreater(static, 0.0)
        self.assertGreater(attached, 0.0)


if __name__ == "__main__":
    unittest.main()
