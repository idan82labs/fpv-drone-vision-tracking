import unittest

from scripts import apply_gated_surface_branch as gated


class ApplyGatedSurfaceBranchTests(unittest.TestCase):
    def test_should_gate_surface_low_confidence_rank(self):
        trace = {
            "state": "T",
            "selected": "1",
            "rank": "18",
            "target_margin": "2.0",
            "raw_score": "1.0",
            "router_bucket": "surface",
        }

        active, reason = gated.should_gate(
            trace,
            gate_states={"A", "P", "C", "S", "E"},
            gate_routers={"surface", "line", "boundary", "unknown"},
            gate_rank_min=10,
            gate_margin_max=0.8,
            gate_raw_score_max=-999.0,
        )

        self.assertTrue(active)
        self.assertIn("rank_ge_10", reason)

    def test_should_not_gate_clean_confident_track(self):
        trace = {
            "state": "T",
            "selected": "1",
            "rank": "1",
            "target_margin": "2.0",
            "raw_score": "1.0",
            "router_bucket": "surface",
        }

        active, reason = gated.should_gate(
            trace,
            gate_states={"A", "P", "C", "S", "E"},
            gate_routers={"surface", "line", "boundary", "unknown"},
            gate_rank_min=10,
            gate_margin_max=0.8,
            gate_raw_score_max=-999.0,
        )

        self.assertFalse(active)
        self.assertEqual(reason, "confident_base")

    def test_merge_requires_repeated_low_confidence_and_surface_risk(self):
        base = {
            1: [{"frame": "1", "rank": "1", "x": "1", "y": "1", "cand_line_context": "0.7"}],
            2: [{"frame": "2", "rank": "1", "x": "2", "y": "2", "cand_line_context": "0.7"}],
            3: [{"frame": "3", "rank": "1", "x": "3", "y": "3"}],
        }
        surface = {
            1: [{"frame": "1", "rank": "1", "x": "9", "y": "9", "surface_halo_score": "0.8", "surface_halo_logit": "1.2"}],
            2: [{"frame": "2", "rank": "1", "x": "8", "y": "8", "surface_halo_score": "0.8", "surface_halo_logit": "1.2"}],
            3: [{"frame": "3", "rank": "1", "x": "7", "y": "7", "surface_halo_score": "0.8", "surface_halo_logit": "1.2"}],
        }
        trace = {
            1: {"frame": "1", "state": "T", "selected": "1", "rank": "20", "target_margin": "2", "router_bucket": "surface"},
            2: {"frame": "2", "state": "T", "selected": "1", "rank": "20", "target_margin": "2", "router_bucket": "surface"},
            3: {"frame": "3", "state": "T", "selected": "1", "rank": "1", "target_margin": "2", "router_bucket": "surface"},
        }
        args = type(
            "Args",
            (),
            {
                "gate_states": "A,P,C,S,E",
                "gate_routers": "surface,line,boundary,unknown",
                "gate_rank_min": 10,
                "gate_margin_max": 0.8,
                "gate_raw_score_max": -999.0,
                "gate_low_conf_frames": 2,
                "surface_risk_min": 1.0,
                "gate_hold_frames": 0,
                "gate_hold_risk_min": 0.5,
                "surface_score_min": -1.0,
                "surface_top_per_frame": 3,
                "base_keep_when_gated": 0,
                "surface_as_learned_logits": True,
            },
        )()

        rows, report = gated.merge_candidates(base, surface, trace, args)

        self.assertEqual(rows[0]["x"], "1")
        self.assertEqual(rows[0]["gate_active"], "0")
        self.assertIn("low_conf_streak_1_lt_2", rows[0]["gate_reason"])
        self.assertEqual(rows[1]["x"], "8")
        self.assertEqual(rows[1]["gate_active"], "1")
        self.assertEqual(rows[1]["crop_t_logit"], "1.200000")
        self.assertEqual(rows[1]["base_trace_state"], "T")
        self.assertEqual(rows[1]["base_trace_rank"], "20")
        self.assertEqual(rows[1]["gate_low_confidence"], 1)
        self.assertEqual(rows[1]["gate_low_conf_streak"], 2)
        self.assertGreater(float(rows[1]["gate_surface_risk_score"]), 1.0)
        self.assertEqual(rows[2]["x"], "3")
        self.assertEqual(rows[2]["gate_active"], "0")
        self.assertEqual([r["gate_active"] for r in report], [0, 1, 0])

    def test_merge_blocks_easy_surface_when_risk_is_low(self):
        base = {
            1: [{"frame": "1", "rank": "1", "x": "1", "y": "1"}],
            2: [{"frame": "2", "rank": "1", "x": "2", "y": "2"}],
        }
        surface = {
            1: [{"frame": "1", "rank": "1", "x": "9", "y": "9", "surface_halo_score": "0.8"}],
            2: [{"frame": "2", "rank": "1", "x": "8", "y": "8", "surface_halo_score": "0.8"}],
        }
        trace = {
            1: {"frame": "1", "state": "T", "selected": "1", "rank": "11", "target_margin": "2.0", "router_bucket": "surface"},
            2: {"frame": "2", "state": "T", "selected": "1", "rank": "11", "target_margin": "2.0", "router_bucket": "surface"},
        }
        args = type(
            "Args",
            (),
            {
                "gate_states": "A,P,C,S,E",
                "gate_routers": "surface,line,boundary,unknown",
                "gate_rank_min": 10,
                "gate_margin_max": 0.8,
                "gate_raw_score_max": -999.0,
                "gate_low_conf_frames": 2,
                "surface_risk_min": 1.0,
                "gate_hold_frames": 0,
                "gate_hold_risk_min": 0.5,
                "surface_score_min": -1.0,
                "surface_top_per_frame": 3,
                "base_keep_when_gated": 0,
                "surface_as_learned_logits": False,
            },
        )()

        rows, report = gated.merge_candidates(base, surface, trace, args)

        self.assertEqual([r["gate_active"] for r in report], [0, 0])
        self.assertEqual([r["x"] for r in rows], ["1", "2"])
        self.assertIn("surface_risk", report[1]["gate_reason"])

    def test_merge_can_hold_surface_rescue_after_recent_gate(self):
        base = {
            1: [{"frame": "1", "rank": "1", "x": "1", "y": "1", "cand_line_context": "0.7"}],
            2: [{"frame": "2", "rank": "1", "x": "2", "y": "2", "cand_line_context": "0.7"}],
            3: [{"frame": "3", "rank": "1", "x": "3", "y": "3", "cand_line_context": "0.7"}],
        }
        surface = {
            1: [{"frame": "1", "rank": "1", "x": "9", "y": "9", "surface_halo_score": "0.8"}],
            2: [{"frame": "2", "rank": "1", "x": "8", "y": "8", "surface_halo_score": "0.8"}],
            3: [{"frame": "3", "rank": "1", "x": "7", "y": "7", "surface_halo_score": "0.8"}],
        }
        trace = {
            1: {"frame": "1", "state": "T", "selected": "1", "rank": "20", "target_margin": "2", "router_bucket": "surface"},
            2: {"frame": "2", "state": "T", "selected": "1", "rank": "20", "target_margin": "2", "router_bucket": "surface"},
            3: {"frame": "3", "state": "T", "selected": "1", "rank": "1", "target_margin": "1.0", "router_bucket": "boundary"},
        }
        args = type(
            "Args",
            (),
            {
                "gate_states": "A,P,C,S,E",
                "gate_routers": "surface,line,boundary,unknown",
                "gate_rank_min": 10,
                "gate_margin_max": 0.8,
                "gate_raw_score_max": -999.0,
                "gate_low_conf_frames": 2,
                "surface_risk_min": 1.0,
                "gate_hold_frames": 2,
                "gate_hold_risk_min": 0.5,
                "surface_score_min": -1.0,
                "surface_top_per_frame": 3,
                "base_keep_when_gated": 0,
                "surface_as_learned_logits": False,
            },
        )()

        rows, report = gated.merge_candidates(base, surface, trace, args)

        self.assertEqual([r["gate_active"] for r in report], [0, 1, 1])
        self.assertEqual(rows[-1]["x"], "7")
        self.assertIn("hold_after_gate", report[-1]["gate_reason"])

    def test_adapt_surface_row_normalizes_generic_detector_score(self):
        row = {"frame": "1", "rank": "4", "score": "2.0"}

        out = gated.adapt_surface_row(row, as_learned_logits=True)

        self.assertEqual(out["gated_surface_branch"], "1")
        self.assertAlmostEqual(float(out["surface_halo_score"]), 0.880797, places=5)
        self.assertAlmostEqual(float(out["crop_t_prob"]), 0.880797, places=5)
        self.assertAlmostEqual(float(out["crop_t_logit"]), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
