import unittest
import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

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
            boundary_obs=0.0,
            null_obs=0.0,
            target_llr=target,
            static_llr=static - target,
            attached_llr=attached - target,
            boundary_llr=-target,
            null_llr=-target,
            target_margin=target - max(static, attached, 0.0),
            router_bucket="unknown",
            proposal_prior=0.0,
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
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=False,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
            beam_width=24,
            state_beam=6,
        )

        self.assertEqual(rows[0]["state"], "P")
        self.assertEqual(rows[0]["selected"], 0)
        self.assertEqual(rows[1]["state"], "T")
        self.assertEqual(rows[1]["x"], 1.0)
        self.assertEqual(rows[1]["w"], 3.0)
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
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=True,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
            beam_width=32,
            state_beam=8,
        )

        self.assertIn(rows[0]["state"], {"A", "E"})
        self.assertEqual(rows[1]["selected"], 0)
        self.assertEqual(rows[2]["selected"], 1)
        self.assertEqual(summary["visible_strict"], 1)

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
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=False,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
            beam_width=16,
            state_beam=8,
        )

        absent = max(h.score for h in out if h.state == "A")
        clutter = max(h.score for h in out if h.state == "E")
        self.assertLessEqual(clutter, absent)

    def test_clutter_lock_requires_clutter_to_beat_target(self):
        hyp = explicit.Hypothesis("T", 0.0, bbox=(0.0, 0.0, 3.0, 3.0), hits=4)
        cand = self.cand(2, 1, 0, target=3.0, attached=3.2)
        cand = replace(cand, attached_llr=0.2, target_llr=2.4)

        out = explicit.step_hypotheses(
            [hyp],
            [cand],
            acquire_threshold=0.5,
            track_threshold=0.4,
            acquire_hits=1,
            max_misses=0,
            max_jump_px=12.0,
            clutter_margin=0.1,
            quarantine_px=10.0,
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=False,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
            beam_width=16,
            state_beam=8,
            clutter_lock_gap=0.0,
        )

        self.assertFalse(any(h.state == "E" for h in out))
        self.assertTrue(any(h.state == "T" for h in out))

    def test_high_confidence_surface_branch_can_instant_acquire_from_absent(self):
        hyp = explicit.Hypothesis("A", 0.0)
        cand = replace(
            self.cand(1, 10, 10, target=3.0, attached=0.0),
            row={"gated_surface_branch": "1", "surface_halo_score": "0.96"},
        )

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
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=False,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
            beam_width=16,
            state_beam=8,
            instant_surface_acquire_score=0.95,
        )

        instant = [h for h in out if h.state == "T" and h.selected is cand]
        self.assertTrue(instant)
        self.assertEqual(instant[0].reason, "instant_acquire")

    def test_high_confidence_surface_branch_does_not_instant_acquire_from_lock(self):
        hyp = explicit.Hypothesis("E", 0.0, quarantine_bbox=(0.0, 0.0, 3.0, 3.0), lock_age=1)
        cand = replace(
            self.cand(1, 20, 10, target=3.0, attached=0.0),
            row={"gated_surface_branch": "1", "surface_halo_score": "0.96"},
        )

        out = explicit.step_hypotheses(
            [hyp],
            [cand],
            acquire_threshold=0.5,
            track_threshold=0.4,
            acquire_hits=1,
            max_misses=0,
            max_jump_px=60.0,
            clutter_margin=0.4,
            quarantine_px=10.0,
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=False,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
            beam_width=16,
            state_beam=8,
            instant_surface_acquire_score=0.95,
        )

        self.assertFalse(any(h.state == "T" for h in out))
        self.assertTrue(any(h.state == "P" for h in out))

    def test_quarantine_applies_across_beam_not_only_same_path(self):
        hyps = [
            explicit.Hypothesis("A", 10.0),
            explicit.Hypothesis("E", 1.0, quarantine_bbox=(0.0, 0.0, 3.0, 3.0), lock_age=3),
        ]
        cands = [
            self.cand(2, 0, 0, target=5.0),
            self.cand(2, 40, 0, target=2.0),
        ]

        out = explicit.step_hypotheses(
            hyps,
            cands,
            acquire_threshold=0.5,
            track_threshold=0.4,
            acquire_hits=1,
            max_misses=0,
            max_jump_px=60.0,
            clutter_margin=0.4,
            quarantine_px=12.0,
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=True,
            quarantine_override_margin=99.0,
            motion_weight=0.2,
            beam_width=32,
            state_beam=8,
        )

        target_candidates = [hyp for hyp in out if hyp.bbox is not None and hyp.state in {"P", "T"}]
        self.assertTrue(target_candidates)
        self.assertTrue(all(hyp.bbox is None or hyp.bbox[0] != 0.0 for hyp in target_candidates))

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

    def test_joint_observation_true_candidate_beats_static_attached_null(self):
        row = {
            "rank": "1",
            "score": "0.8",
            "clba_target_likelihood": "1.4",
            "clba_gain_norm": "1.6",
            "clba_target_q": "1.5",
            "clba_bg_q": "0.1",
            "clba_bg_static_likelihood": "0.1",
            "clba_attached_likelihood": "0.0",
            "clba_path_bg_dist_mean": "8.0",
            "tube_positive_pair_rate": "0.8",
            "cand_router_state": "clean_sky",
        }

        obs = explicit.joint_candidate_observations(
            row,
            score_column="score",
            score_weight=0.3,
            clba_weight=0.55,
            path_weight=0.25,
            static_weight=0.7,
            attached_weight=0.7,
            rank_weight=0.1,
        )

        self.assertGreater(obs["target_llr"], obs["static_llr"])
        self.assertGreater(obs["target_llr"], obs["attached_llr"])
        self.assertGreater(obs["target_llr"], obs["null_llr"])

    def test_target_only_proposal_can_carry_objectness_against_clba_false(self):
        true_row = {
            "rank": "1",
            "score": "40",
            "clba_target_likelihood": "1.0",
            "clba_gain_norm": "1.0",
            "clba_path_bg_dist_mean": "2.0",
        }
        false_row = {
            "rank": "2",
            "score": "0",
            "clba_target_likelihood": "4.0",
            "clba_gain_norm": "4.0",
            "clba_path_bg_dist_mean": "8.0",
        }

        true_obs = explicit.joint_candidate_observations(
            true_row,
            score_column="score",
            score_weight=1.0,
            clba_weight=0.35,
            path_weight=0.1,
            static_weight=0.7,
            attached_weight=0.7,
            rank_weight=0.1,
            proposal_clip=4.0,
            proposal_mode="target_only",
        )
        false_obs = explicit.joint_candidate_observations(
            false_row,
            score_column="score",
            score_weight=1.0,
            clba_weight=0.35,
            path_weight=0.1,
            static_weight=0.7,
            attached_weight=0.7,
            rank_weight=0.1,
            proposal_clip=4.0,
            proposal_mode="target_only",
        )

        self.assertGreater(true_obs["target_llr"], false_obs["target_llr"])

    def test_old_score_observation_mode_disables_clutter_terms(self):
        cand = replace(
            self.cand(target=-3.0, static=9.0, attached=8.0),
            proposal_prior=2.5,
            null_obs=-0.5,
        )

        out = explicit.apply_observation_mode(cand, "old_score", None, strict_tol_px=8.0)

        self.assertEqual(out.target_obs, 2.5)
        self.assertEqual(out.static_obs, -6.0)
        self.assertEqual(out.attached_obs, -6.0)
        self.assertGreater(out.target_llr, out.static_llr)
        self.assertGreater(out.target_llr, out.null_llr)

    def test_oracle_observation_mode_marks_true_false_and_null_cases(self):
        visible = explicit.Label(True, (0.0, 0.0, 3.0, 3.0))
        true_cand = self.cand(x=0, y=0)
        false_cand = self.cand(x=40, y=0)

        true_out = explicit.apply_observation_mode(true_cand, "oracle", visible, strict_tol_px=8.0)
        false_out = explicit.apply_observation_mode(false_cand, "oracle", visible, strict_tol_px=8.0)
        null_out = explicit.apply_observation_mode(false_cand, "oracle", explicit.Label(False, None), strict_tol_px=8.0)

        self.assertGreater(true_out.target_llr, 10.0)
        self.assertLess(false_out.target_llr, -9.0)
        self.assertLess(null_out.target_llr, -15.0)
        self.assertGreater(null_out.null_llr, 8.0)

    def test_learned_logits_observation_mode_uses_class_columns(self):
        cand = self.cand(target=0.0, static=0.0, attached=0.0)
        cand = replace(
            cand,
            row={
                "crop_t_logit": "3.0",
                "crop_s_logit": "-2.0",
                "crop_e_logit": "-1.0",
                "crop_h_logit": "-3.0",
                "crop_g_logit": "-4.0",
            },
            proposal_prior=1.0,
            router_bucket="clean_sky",
        )

        out = explicit.apply_observation_mode(cand, "learned_logits", None, strict_tol_px=8.0)

        self.assertGreater(out.target_obs, out.static_obs)
        self.assertGreater(out.target_llr, out.static_llr)
        self.assertGreater(out.target_llr, out.boundary_llr)

    def test_learned_logits_uses_explicit_surface_null_prior(self):
        cand = replace(
            self.cand(target=0.0, static=0.0, attached=0.0),
            row={
                "crop_t_logit": "1.0",
                "crop_s_logit": "-2.0",
                "crop_e_logit": "-2.0",
                "crop_h_logit": "-2.0",
                "crop_g_logit": "-2.0",
            },
            proposal_prior=0.0,
            router_bucket="surface",
        )
        args = argparse.Namespace(
            null_priors="surface=3.5",
            learned_prior_source="proposal",
            learned_prior_clip=2.0,
            learned_target_prior_weight=0.0,
            learned_clutter_prior_weight=0.0,
            learned_generic_prior_weight=0.0,
        )

        default_out = explicit.apply_observation_mode(cand, "learned_logits", None, strict_tol_px=8.0)
        explicit_out = explicit.apply_observation_mode(cand, "learned_logits", None, strict_tol_px=8.0, args=args)

        self.assertAlmostEqual(default_out.null_obs, 0.20)
        self.assertAlmostEqual(explicit_out.null_obs, 3.5)
        self.assertLess(explicit_out.target_margin, default_out.target_margin)
        self.assertAlmostEqual(explicit_out.target_margin, explicit_out.target_obs - explicit_out.null_obs)

    def test_learned_logits_can_use_raw_score_prior_without_double_scaling(self):
        cand = replace(
            self.cand(target=0.0, static=0.0, attached=0.0),
            row={
                "crop_t_logit": "0.0",
                "crop_s_logit": "-1.0",
                "crop_e_logit": "-1.0",
                "crop_h_logit": "-1.0",
                "crop_g_logit": "-1.0",
            },
            raw_score=0.9,
            proposal_prior=0.0,
            router_bucket="clean_sky",
        )
        args = argparse.Namespace(
            learned_prior_source="raw_score",
            learned_prior_clip=3.0,
            learned_target_prior_weight=0.25,
            learned_clutter_prior_weight=0.0,
            learned_generic_prior_weight=0.0,
        )

        out = explicit.apply_observation_mode(cand, "learned_logits", None, strict_tol_px=8.0, args=args)

        self.assertGreater(out.target_obs, 0.4)
        self.assertGreater(out.target_obs, out.static_obs)

    def test_learned_logits_generic_clutter_competes_with_target(self):
        cand = replace(
            self.cand(target=0.0, static=0.0, attached=0.0),
            row={
                "crop_t_logit": "0.5",
                "crop_s_logit": "-2.0",
                "crop_e_logit": "-2.0",
                "crop_h_logit": "-2.0",
                "crop_g_logit": "4.0",
            },
            proposal_prior=0.0,
            router_bucket="unknown",
        )

        out = explicit.apply_observation_mode(cand, "learned_logits", None, strict_tol_px=8.0)

        self.assertGreater(out.generic_obs, out.target_obs)
        self.assertLess(out.target_llr, 0.0)
        self.assertGreater(out.generic_llr, out.static_llr)

    def test_surface_branch_rank_bonus_lifts_gated_surface_candidate(self):
        cand = replace(
            self.cand(target=0.0, static=0.0, attached=0.0),
            row={
                "gated_surface_branch": "1",
                "surface_halo_parent_rank": "1",
                "crop_t_logit": "-1.0",
                "crop_s_logit": "-3.0",
                "crop_e_logit": "-3.0",
                "crop_h_logit": "-3.0",
                "crop_g_logit": "0.0",
            },
            proposal_prior=0.0,
            router_bucket="surface",
        )
        args = argparse.Namespace(
            learned_prior_source="proposal",
            learned_prior_clip=3.0,
            learned_target_prior_weight=0.0,
            learned_clutter_prior_weight=0.0,
            learned_generic_prior_weight=0.0,
            surface_branch_rank_bonus=2.0,
            surface_branch_rank_decay=0.0,
        )

        out = explicit.apply_observation_mode(cand, "learned_logits", None, strict_tol_px=8.0, args=args)

        self.assertGreater(out.target_obs, 0.9)
        self.assertGreater(out.target_llr, 0.0)

    def test_range_bin_motion_rejects_absurd_but_not_plausible_jump(self):
        prev = explicit.Hypothesis("T", 0.0, bbox=(0.0, 0.0, 3.0, 3.0), vx=1.0, vy=0.0)
        plausible = self.cand(2, 4, 0)
        absurd = self.cand(2, 400, 0)

        plausible_cost = explicit.motion_cost(
            prev,
            plausible,
            max_jump_px=12.0,
            motion_weight=0.2,
            fps=30.0,
            image_width=320.0,
            horizontal_fov_deg=120.0,
            vmax_mps=10.0,
            registration_sigma_px=1.5,
            box_sigma_px=2.0,
            motion_prior_weight=0.25,
            absurd_jump_px=96.0,
        )
        absurd_cost = explicit.motion_cost(
            prev,
            absurd,
            max_jump_px=12.0,
            motion_weight=0.2,
            fps=30.0,
            image_width=320.0,
            horizontal_fov_deg=120.0,
            vmax_mps=10.0,
            registration_sigma_px=1.5,
            box_sigma_px=2.0,
            motion_prior_weight=0.25,
            absurd_jump_px=96.0,
        )

        self.assertLess(plausible_cost, 1e5)
        self.assertGreaterEqual(absurd_cost, 1e5)

    def test_continuity_bonus_only_applies_to_plausible_existing_track(self):
        prev = explicit.Hypothesis("T", 0.0, bbox=(0.0, 0.0, 3.0, 3.0), vx=1.0, vy=0.0)
        good = replace(self.cand(2, 1, 0, target=0.0, static=0.2, attached=0.1), raw_score=0.8)
        far = replace(self.cand(2, 50, 0, target=0.0, static=0.2, attached=0.1), raw_score=0.8)
        clutter_wins = replace(self.cand(2, 1, 0, target=0.0, static=3.0, attached=0.1), raw_score=0.8)

        rescued = explicit.continuity_adjusted_target_llr(
            prev,
            good,
            target_llr=-0.5,
            continuity_bonus=0.7,
            continuity_max_pred_error_px=8.0,
            continuity_clutter_gap=0.5,
            continuity_min_raw_score=0.1,
        )
        far_out = explicit.continuity_adjusted_target_llr(
            prev,
            far,
            target_llr=-0.5,
            continuity_bonus=0.7,
            continuity_max_pred_error_px=8.0,
            continuity_clutter_gap=0.5,
            continuity_min_raw_score=0.1,
        )
        clutter_out = explicit.continuity_adjusted_target_llr(
            prev,
            clutter_wins,
            target_llr=-0.5,
            continuity_bonus=0.7,
            continuity_max_pred_error_px=8.0,
            continuity_clutter_gap=0.5,
            continuity_min_raw_score=0.1,
        )

        self.assertAlmostEqual(rescued, 0.2)
        self.assertAlmostEqual(far_out, -0.5)
        self.assertAlmostEqual(clutter_out, -0.5)

    def test_null_calibration_offsets_reduce_target_llr_on_null_bucket(self):
        labels = {
            1: explicit.Label(False, None),
            2: explicit.Label(True, (10.0, 0.0, 3.0, 3.0)),
        }
        c1 = self.cand(1, 0, 0, target=2.0)
        c2 = self.cand(2, 10, 0, target=2.0)
        offsets, adjusted, rows = explicit.calibrate_null_offsets(
            labels,
            {1: [c1], 2: [c2]},
            q=1.0,
            min_router_null_samples=1,
            margin=0.0,
        )

        self.assertGreaterEqual(offsets["global"], 2.0)
        self.assertLess(adjusted[1][0].target_llr, c1.target_llr)
        self.assertTrue(rows)

    def test_load_candidates_keeps_blank_clip_rows_for_per_clip_exports(self):
        args = argparse.Namespace(
            null_priors="",
            score_weight=0.75,
            clba_weight=0.55,
            path_weight=0.25,
            static_weight=0.7,
            attached_weight=0.7,
            rank_weight=0.12,
            proposal_clip=2.0,
            proposal_mode="shared",
            observation_mode="old_score",
            learned_prior_source="proposal",
            learned_target_prior_weight=0.15,
            learned_clutter_prior_weight=0.05,
            learned_generic_prior_weight=0.02,
            learned_prior_clip=2.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            path.write_text(
                "clip,frame,rank,x,y,w,h,score\n"
                ",1,1,10,10,3,3,0.9\n"
                "wanted,1,2,20,10,3,3,0.8\n"
                "other,1,3,30,10,3,3,0.7\n"
            )

            loaded = explicit.load_candidates(path, "wanted", 10, "score", args)

        self.assertEqual([c.rank for c in loaded[1]], [1, 2])

    def test_absent_score_cap_bounds_null_path(self):
        out = explicit.step_hypotheses(
            [explicit.Hypothesis("A", 10.0)],
            [],
            acquire_threshold=0.5,
            track_threshold=0.4,
            acquire_hits=1,
            max_misses=0,
            max_jump_px=12.0,
            clutter_margin=0.4,
            quarantine_px=10.0,
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=False,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
            absent_reward=0.05,
            absent_score_cap=2.0,
            beam_width=16,
            state_beam=8,
        )

        self.assertLessEqual(max(h.score for h in out if h.state == "A"), 2.0)

    def test_summary_selection_key_can_prioritize_ground_recall(self):
        recall_focused = {
            "all_frame_accuracy": 0.80,
            "visible_strict_recall": 0.92,
            "visible_loose_recall": 0.94,
            "invisible_no_box_rate": 0.30,
            "selected_frames": 450,
        }
        balanced = {
            "all_frame_accuracy": 0.88,
            "visible_strict_recall": 0.89,
            "visible_loose_recall": 0.91,
            "invisible_no_box_rate": 0.70,
            "selected_frames": 360,
        }

        self.assertGreater(
            explicit.summary_selection_key(balanced, "all_frame_accuracy"),
            explicit.summary_selection_key(recall_focused, "all_frame_accuracy"),
        )
        self.assertGreater(
            explicit.summary_selection_key(recall_focused, "visible_strict_recall"),
            explicit.summary_selection_key(balanced, "visible_strict_recall"),
        )

    def test_selector_processes_unlabeled_candidate_frames_between_labels(self):
        labels = {
            1: explicit.Label(True, (0.0, 0.0, 3.0, 3.0)),
            3: explicit.Label(True, (2.0, 0.0, 3.0, 3.0)),
        }
        candidates = {
            1: [self.cand(1, 0, 0)],
            2: [self.cand(2, 1, 0)],
            3: [self.cand(3, 2, 0)],
        }

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
            static_quarantine_frames=5,
            attached_quarantine_frames=5,
            global_quarantine=False,
            quarantine_override_margin=1.0,
            motion_weight=0.2,
        )

        self.assertEqual(summary["frames_all"], 2)
        self.assertEqual(summary["processed_frames"], 3)
        self.assertEqual([r["labeled"] for r in rows], [1, 0, 1])
        self.assertEqual(rows[2]["state"], "T")


if __name__ == "__main__":
    unittest.main()
