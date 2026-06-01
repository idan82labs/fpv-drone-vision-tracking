import argparse
import math
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

from scripts import motion_detector_v2 as base
from scripts import tbd_motion_detector as tbd


def args(**overrides):
    vals = {
        "candidate_router": "off",
        "runtime_mode": "baseline",
        "top_k_candidates": 80,
        "scenario_balance": True,
        "scenario_sky_top_k": 24,
        "scenario_surface_top_k": 34,
        "scenario_boundary_top_k": 20,
        "scenario_large_top_k": 20,
        "scenario_coast_top_k": 18,
        "obs_weight": 1.0,
        "line_weight": 0.0,
        "router_surface_source_penalty": 2.5,
        "router_line_penalty": 1.5,
        "router_surface_bonus": 0.0,
        "support_penalty_weight": 0.0,
        "support_penalty_threshold": 3.5,
        "native_roi_score": False,
        "native_roi_weight": 0.35,
        "native_roi_neutral": 1.0,
        "app_low_residual_penalty": 0.0,
        "app_low_residual_px": 4.0,
        "surface_branch_allow_acquisition": False,
        "surface_branch_min_candidates": 4,
        "surface_branch_min_score": 2.0,
        "surface_branch_track_min_hits": 6,
        "surface_branch_track_min_score": 24.0,
        "surface_branch_track_rate": 0.55,
        "surface_ranker_scope": "all",
        "surface_ranker_min_rate": 0.45,
        "surface_ranker_model": "",
        "surface_ranker_policy": "off",
        "surface_ranker_gate": "none",
        "surface_ranker_threshold": 0.76,
        "surface_ranker_top_n": 80,
        "beam_width": 90,
        "max_selected_misses": 1,
        "min_path_hits": 1,
        "selected_score": 6.0,
        "tube_verifier": "off",
        "tube_verifier_floor": -999.0,
        "tube_verifier_weight": 1.0,
        "sky_bonus_weight": 0.0,
        "density_penalty_weight": 0.0,
        "selection_margin": 0.0,
        "sky_rescue": False,
        "hybrid_coast_highres_recenter": False,
        "hybrid_coast_recenter_radius_det_px": 6.0,
        "hybrid_coast_recenter_radii": "2,3,4",
        "hybrid_coast_recenter_peaks": 3,
        "hybrid_coast_recenter_texture_weight": 0.010,
        "hybrid_coast_recenter_shift_penalty": 0.20,
        "hybrid_coast_recenter_score_weight": 0.70,
        "target_local_recovery_proposals": False,
        "target_local_recovery_top_k": 8,
        "target_local_recovery_max_seed_gap": 5,
        "target_local_recovery_min_hits": 3,
        "target_local_recovery_min_verified_score": 6.0,
        "target_local_recovery_search_radius_det_px": 18.0,
        "target_local_recovery_radii": "2,3,4",
        "target_local_recovery_texture_weight": 0.010,
        "target_local_recovery_shift_penalty": 0.020,
        "target_local_recovery_score_weight": 0.95,
        "target_local_recovery_state_score_weight": 0.10,
        "target_local_recovery_box_det_px": 4.0,
        "target_local_recovery_predictor": "clamped_velocity",
        "target_local_recovery_max_velocity_px": 3.0,
        "target_local_state_select": False,
        "target_local_state_select_error_px": 9.0,
        "target_local_state_select_improvement_px": 6.0,
        "target_local_state_select_max_pred_error_px": 6.0,
        "target_local_state_select_anchor_px": 18.0,
        "target_local_state_select_min_side": 7.0,
        "target_local_state_select_max_side": 18.0,
        "target_local_state_select_top_n": 80,
        "target_local_state_select_sources": "target_local_recovery",
        "target_local_state_select_allow_missed_anchor": False,
        "target_local_state_select_missed_max_misses": 1,
        "candidate_local_recenter_proposals": False,
        "candidate_local_recenter_seed_top_k": 24,
        "candidate_local_recenter_top_k": 48,
        "candidate_local_recenter_radius_det_px": 16.0,
        "candidate_local_recenter_radii": "2,3,4",
        "candidate_local_recenter_texture_weight": 0.010,
        "candidate_local_recenter_shift_penalty": 0.025,
        "candidate_local_recenter_score_weight": 1.10,
        "candidate_local_recenter_seed_score_weight": 0.035,
        "candidate_local_recenter_box_det_px": 4.0,
        "candidate_local_recenter_router_scope": "surface_context",
    }
    vals.update(overrides)
    return argparse.Namespace(**vals)


def cand(score, state="clean_sky", source="map", bbox=(10, 10, 3, 3)):
    out = base.Candidate(
        source=source,
        bbox=bbox,
        area=bbox[2] * bbox[3],
        fill=1.0,
        aspect=1.0,
        mean_residual=10.0,
        mean_appearance=10.0,
        local_contrast=10.0,
        texture=10.0,
        line_context=0.0,
        isolation=0.0,
        score=score,
    )
    out.router_state = state
    out.router_confidence = 1.0
    return out


class HybridCoastHighresTests(unittest.TestCase):
    def test_motion_model_fallback_identity_is_opt_in(self):
        with mock.patch.object(sys, "argv", ["tbd_motion_detector.py", "clip.mp4"]):
            self.assertFalse(tbd.parse_args().motion_model_fallback_identity)

        with mock.patch.object(sys, "argv", ["tbd_motion_detector.py", "--motion_model_fallback_identity", "clip.mp4"]):
            self.assertTrue(tbd.parse_args().motion_model_fallback_identity)

    def test_target_local_state_select_is_opt_in(self):
        with mock.patch.object(sys, "argv", ["tbd_motion_detector.py", "clip.mp4"]):
            parsed = tbd.parse_args()
            self.assertFalse(parsed.target_local_state_select)
            self.assertEqual(parsed.target_local_state_select_sources, "target_local_recovery")

        with mock.patch.object(
            sys,
            "argv",
            [
                "tbd_motion_detector.py",
                "--target_local_state_select",
                "--target_local_state_select_error_px",
                "11",
                "--target_local_state_select_top_n",
                "12",
                "clip.mp4",
            ],
        ):
            parsed = tbd.parse_args()
            self.assertTrue(parsed.target_local_state_select)
            self.assertEqual(parsed.target_local_state_select_error_px, 11.0)
            self.assertEqual(parsed.target_local_state_select_top_n, 12)

    def test_local_native_peaks_near_prefers_dark_peak_close_to_prediction(self):
        gray = np.full((80, 80), 150, dtype=np.uint8)
        yy, xx = np.mgrid[:80, :80]
        gray[np.hypot(xx - 42, yy - 39) <= 2.0] = 25
        score_map = tbd.compact_dark_map_native(gray, 2, texture_weight=0.0)

        peaks = tbd.local_native_peaks_near({2: score_map}, (40.0, 40.0), 8.0, 3)

        self.assertGreaterEqual(len(peaks), 1)
        _score, x, y, radius = peaks[0]
        self.assertEqual(radius, 2)
        self.assertLess(math.hypot(x - 42.0, y - 39.0), 3.0)

    def test_target_local_recovery_requires_recent_seed(self):
        opts = args(target_local_recovery_proposals=True)
        gray = np.full((80, 80, 3), 160, dtype=np.uint8)
        det = np.full((40, 40), 160, dtype=np.uint8)
        empty = np.zeros((40, 40), dtype=np.float32)

        old_seed = tbd.TargetLocalRecoverySeed(
            sid=1,
            bbox=(18, 18, 4, 4),
            vx=0.0,
            vy=0.0,
            frame_no=0,
            hits=5,
            verified_score=10.0,
        )

        self.assertEqual(
            tbd.target_local_recovery_candidates(
                old_seed,
                10,
                40,
                40,
                gray,
                0.5,
                empty,
                empty,
                det,
                opts,
            ),
            [],
        )

    def test_target_local_recovery_finds_dark_peak_near_predicted_track(self):
        opts = args(
            target_local_recovery_proposals=True,
            target_local_recovery_radii="2",
            target_local_recovery_texture_weight=0.0,
            target_local_recovery_shift_penalty=0.0,
            target_local_recovery_top_k=3,
        )
        full = np.full((80, 80, 3), 160, dtype=np.uint8)
        yy, xx = np.mgrid[:80, :80]
        full[np.hypot(xx - 42, yy - 39) <= 2.0] = 25
        det = cv2.resize(full, (40, 40), interpolation=cv2.INTER_AREA)
        det_g = base.ensure_gray(det)
        empty = np.zeros((40, 40), dtype=np.float32)
        seed = tbd.TargetLocalRecoverySeed(
            sid=1,
            bbox=(18, 18, 4, 4),
            vx=1.0,
            vy=0.0,
            frame_no=1,
            hits=5,
            verified_score=10.0,
        )

        cands = tbd.target_local_recovery_candidates(
            seed,
            2,
            40,
            40,
            full,
            0.5,
            empty,
            empty,
            det_g,
            opts,
        )

        self.assertGreaterEqual(len(cands), 1)
        self.assertEqual(cands[0].source, "target_local_recovery")
        cx, cy = base.bbox_center(cands[0].bbox)
        self.assertLess(math.hypot(cx - 21.0, cy - 19.5), 3.0)

    def test_target_local_recovery_clamps_poisoned_seed_velocity(self):
        opts = args(
            target_local_recovery_predictor="clamped_velocity",
            target_local_recovery_max_velocity_px=3.0,
        )
        seed = tbd.TargetLocalRecoverySeed(
            sid=1,
            bbox=(18, 18, 4, 4),
            vx=0.0,
            vy=-30.0,
            frame_no=1,
            hits=5,
            verified_score=10.0,
        )

        predicted = tbd.target_local_seed_prediction_bbox(seed, 2, 80, 80, opts)

        _cx, cy = base.bbox_center(predicted)
        self.assertGreater(cy, 16.0)

    def test_target_local_recovery_state_velocity_predictor_preserves_old_behavior(self):
        opts = args(target_local_recovery_predictor="state_velocity")
        seed = tbd.TargetLocalRecoverySeed(
            sid=1,
            bbox=(18, 18, 4, 4),
            vx=0.0,
            vy=-10.0,
            frame_no=1,
            hits=5,
            verified_score=10.0,
        )

        predicted = tbd.target_local_seed_prediction_bbox(seed, 2, 80, 80, opts)

        _cx, cy = base.bbox_center(predicted)
        self.assertLess(cy, 14.0)

    def test_candidate_local_recenter_finds_peak_near_loose_seed(self):
        opts = args(
            candidate_local_recenter_proposals=True,
            candidate_local_recenter_radii="2",
            candidate_local_recenter_texture_weight=0.0,
            candidate_local_recenter_shift_penalty=0.0,
            candidate_local_recenter_top_k=4,
            candidate_local_recenter_seed_top_k=2,
        )
        full = np.full((80, 80, 3), 160, dtype=np.uint8)
        yy, xx = np.mgrid[:80, :80]
        full[np.hypot(xx - 42, yy - 39) <= 2.0] = 25
        det = cv2.resize(full, (40, 40), interpolation=cv2.INTER_AREA)
        det_g = base.ensure_gray(det)
        empty = np.zeros((40, 40), dtype=np.float32)
        seed = cand(10.0, state="surface_backed", source="large_dark", bbox=(26, 18, 4, 4))

        cands = tbd.candidate_local_recenter_candidates(
            [seed],
            full,
            0.5,
            empty,
            empty,
            det_g,
            opts,
        )

        self.assertGreaterEqual(len(cands), 1)
        self.assertEqual(cands[0].source, "candidate_local_recenter")
        cx, cy = base.bbox_center(cands[0].bbox)
        self.assertLess(math.hypot(cx - 21.0, cy - 19.5), 3.0)


class RuntimeRouterTests(unittest.TestCase):
    def test_router_applies_only_when_explicitly_apply(self):
        self.assertFalse(tbd.router_applies(args(candidate_router="off", runtime_mode="surface")))
        self.assertFalse(tbd.router_applies(args(candidate_router="log", runtime_mode="auto")))
        self.assertTrue(tbd.router_applies(args(candidate_router="apply", runtime_mode="auto")))

    def test_scenario_balance_respects_effective_cap(self):
        cands = [
            cand(10 - i * 0.1, "surface_backed", bbox=(i * 3, 10, 3, 3))
            for i in range(20)
        ]
        cands += [
            cand(9 - i * 0.1, "clean_sky", bbox=(i * 3, 25, 3, 3))
            for i in range(20)
        ]
        kept = tbd.scenario_balanced_candidates(cands, args(candidate_router="apply"), True, max_n=12)
        self.assertLessEqual(len(kept), 12)

    def test_auto_surface_branch_needs_explicit_acquisition_or_track(self):
        decision = tbd.FrameRouterDecision("surface", True, 80, 1.0, {})
        routed = [cand(5, "surface_backed", bbox=(i * 4, 10, 3, 3)) for i in range(5)]
        self.assertFalse(
            tbd.surface_branch_needed(
                routed,
                [],
                decision,
                args(candidate_router="apply", runtime_mode="auto"),
            )
        )
        self.assertTrue(
            tbd.surface_branch_needed(
                routed,
                [],
                decision,
                args(candidate_router="apply", runtime_mode="auto", surface_branch_allow_acquisition=True),
            )
        )

    def test_native_roi_score_can_affect_preselection_without_router(self):
        plain = cand(10.0, bbox=(10, 10, 3, 3))
        native_confirmed = cand(9.0, bbox=(30, 10, 3, 3))
        native_confirmed.native_dark_score = 5.0
        opts = args(native_roi_score=True, native_roi_weight=1.0, native_roi_neutral=1.0)

        kept = tbd.dedupe_candidates(
            [plain, native_confirmed],
            1,
            lambda item: tbd.candidate_obs(item, opts),
        )
        self.assertIs(kept[0], native_confirmed)

        balanced = tbd.scenario_balanced_candidates(
            [plain, native_confirmed],
            opts,
            use_router=False,
            max_n=1,
        )
        self.assertIs(balanced[0], native_confirmed)

    def test_surface_ranker_confidence_fallback_overrides_baseline_only_when_confident(self):
        class FakeRanker:
            def __init__(self, high_score):
                self.high_score = high_score

            def scores(self, rows):
                return [0.10 if row["rank"] == 1 else self.high_score for row in rows]

        opts = args(surface_ranker_policy="confidence_fallback", surface_ranker_threshold=0.76)
        tracker = tbd.BeamTBD(opts, px_per_frame=10.0)
        baseline = tbd.PathState(1, (0, 0, 3, 3), contribs=[10.0], hit_flags=[True])
        learned = tbd.PathState(2, (10, 0, 3, 3), contribs=[9.0], hit_flags=[True])
        tracker.states = [baseline, learned]

        tracker.surface_ranker = FakeRanker(high_score=0.90)
        self.assertEqual(tracker.best().sid, 2)

        tracker.surface_ranker = FakeRanker(high_score=0.50)
        self.assertEqual(tracker.best().sid, 1)

    def test_surface_ranker_respects_top_n_cap(self):
        class FakeRanker:
            def scores(self, rows):
                return [0.10 if row["track_id"] != 3 else 0.99 for row in rows]

        opts = args(surface_ranker_policy="confidence_fallback", surface_ranker_threshold=0.76, surface_ranker_top_n=2)
        tracker = tbd.BeamTBD(opts, px_per_frame=10.0)
        tracker.surface_ranker = FakeRanker()
        tracker.states = [
            tbd.PathState(1, (0, 0, 3, 3), contribs=[10.0], hit_flags=[True]),
            tbd.PathState(2, (10, 0, 3, 3), contribs=[9.0], hit_flags=[True]),
            tbd.PathState(3, (20, 0, 3, 3), contribs=[8.0], hit_flags=[True]),
        ]

        self.assertEqual(tracker.best().sid, 1)
        self.assertEqual(tracker.last_surface_ranker_rows, 2)

    def test_surface_ranker_cannot_override_selected_score_floor(self):
        class FakeRanker:
            def scores(self, rows):
                return [0.10 if row["track_id"] == 1 else 0.99 for row in rows]

        opts = args(surface_ranker_policy="confidence_fallback", surface_ranker_threshold=0.76, selected_score=6.0)
        tracker = tbd.BeamTBD(opts, px_per_frame=10.0)
        tracker.surface_ranker = FakeRanker()
        tracker.states = [
            tbd.PathState(1, (0, 0, 3, 3), contribs=[10.0], hit_flags=[True]),
            tbd.PathState(2, (10, 0, 3, 3), contribs=[5.0], hit_flags=[True]),
        ]

        self.assertEqual(tracker.best().sid, 1)
        self.assertFalse(tracker.last_surface_ranker_used)
        self.assertEqual(tracker.last_surface_ranker_sid, 2)

    def test_state_feature_row_schema_matches_top_tube_missing_margin(self):
        opts = args()
        tracker = tbd.BeamTBD(opts, px_per_frame=10.0)
        st = tbd.PathState(1, (0, 0, 3, 3), contribs=[10.0], hit_flags=[True])

        runtime_row = tbd.state_feature_row(st, opts, 1, 10.0, 0.0, None)
        _payload, export_row = tbd.tube_state_payload(1, 1, st, tracker, opts, st, None)

        self.assertEqual(runtime_row["competitor_margin"], "")
        self.assertEqual(export_row["competitor_margin"], "")

    def test_learned_surface_ranker_vectorize_matches_source_bits_and_missing_values(self):
        ranker = object.__new__(tbd.LearnedSurfaceRanker)
        ranker.numeric_features = ["rank", "competitor_margin", "cand_score"]
        ranker.source_features = ["src_map", "src_temporal_stack"]

        x = ranker.vectorize(
            [
                {
                    "rank": 2,
                    "competitor_margin": "",
                    "cand_score": 5.5,
                    "cand_source": "temporal_stack",
                }
            ]
        )

        self.assertEqual(x.shape, (1, 5))
        self.assertEqual(x[0, 0], 2.0)
        self.assertTrue(math.isnan(x[0, 1]))
        self.assertEqual(x[0, 2], 5.5)
        self.assertEqual(x[0, 3], 0.0)
        self.assertEqual(x[0, 4], 1.0)

    def test_surface_ranker_gate_requires_support_when_enabled(self):
        self.assertTrue(
            tbd.surface_ranker_gate_allows(
                {"cand_source": "large_dark", "cand_attached_support": 10.0, "tube_mean_attached_support": 1.0},
                "high_support",
            )
        )

    def test_surface_ranker_extra_features_are_scoped_to_surface_paths(self):
        opts = args(window=5, surface_ranker_scope="surface_backed", surface_ranker_min_rate=0.5)
        st = tbd.PathState(
            1,
            (0, 0, 3, 3),
            candidate_history=[
                {"router_state": "clean_sky"},
                {"router_state": "surface_backed"},
            ],
        )

        self.assertTrue(tbd.surface_ranker_transition_features_needed(st, cand(5.0, "surface_backed"), opts))
        self.assertFalse(tbd.surface_ranker_transition_features_needed(st, cand(5.0, "clean_sky"), opts))
        self.assertTrue(
            tbd.surface_ranker_transition_features_needed(
                st,
                cand(5.0, "clean_sky"),
                args(surface_ranker_scope="all", surface_ranker_min_rate=0.9),
            )
        )

    def test_surface_context_scope_includes_boundary_tubes(self):
        features = {"router_surface_backed_rate": 0.0, "router_boundary_rate": 0.5}
        opts = args(surface_ranker_scope="surface_context", surface_ranker_min_rate=0.45)

        self.assertTrue(tbd.surface_ranker_scope_allows(features, opts))
        self.assertFalse(
            tbd.surface_ranker_scope_allows(
                features,
                args(surface_ranker_scope="surface_backed", surface_ranker_min_rate=0.45),
            )
        )

    def test_raw_best_for_mask_avoids_learned_ranker_path(self):
        class ExplodingRanker:
            def scores(self, rows):
                raise AssertionError("ranker should not run for mask-only best")

        opts = args(surface_ranker_policy="confidence_fallback", surface_ranker_threshold=0.0)
        tracker = tbd.BeamTBD(opts, px_per_frame=10.0)
        tracker.surface_ranker = ExplodingRanker()
        tracker.states = [
            tbd.PathState(1, (0, 0, 3, 3), contribs=[10.0], hit_flags=[True]),
            tbd.PathState(2, (10, 0, 3, 3), contribs=[9.0], hit_flags=[True]),
        ]

        self.assertEqual(tracker.raw_best_for_mask().sid, 1)

    def test_surface_ranker_gate_applies_after_best_learned_choice(self):
        class FakeRanker:
            def scores(self, rows):
                return [0.10 if row["track_id"] == 1 else 0.99 if row["track_id"] == 2 else 0.80 for row in rows]

        opts = args(
            surface_ranker_policy="confidence_fallback",
            surface_ranker_threshold=0.0,
            surface_ranker_gate="high_support",
        )
        tracker = tbd.BeamTBD(opts, px_per_frame=10.0)
        tracker.surface_ranker = FakeRanker()
        baseline = tbd.PathState(1, (0, 0, 3, 3), contribs=[10.0], hit_flags=[True])
        ungated_high_score = tbd.PathState(2, (10, 0, 3, 3), contribs=[9.0], hit_flags=[True])
        gated_lower_score = tbd.PathState(3, (20, 0, 3, 3), contribs=[8.0], hit_flags=[True])
        gated_lower_score.last_candidate = cand(8.0, source="large_dark")
        gated_lower_score.last_candidate.attached_support = 10.0
        tracker.states = [baseline, ungated_high_score, gated_lower_score]

        selected = tracker.best()

        self.assertEqual(selected.sid, 1)
        self.assertEqual(tracker.last_surface_ranker_sid, 2)
        self.assertFalse(tracker.last_surface_ranker_used)
        self.assertFalse(
            tbd.surface_ranker_gate_allows(
                {"cand_source": "large_dark", "cand_attached_support": 2.0, "tube_mean_attached_support": 1.0},
                "high_support",
            )
        )

    def test_default_best_fast_path_does_not_compute_tube_features(self):
        opts = args(tube_verifier="off", surface_ranker_policy="off", selection_margin=0.0, sky_rescue=False)
        tracker = tbd.BeamTBD(opts, px_per_frame=10.0)
        tracker.states = [
            tbd.PathState(1, (0, 0, 3, 3), contribs=[10.0], hit_flags=[True]),
            tbd.PathState(2, (10, 0, 3, 3), contribs=[9.0], hit_flags=[True]),
        ]

        with mock.patch.object(tbd, "tube_features", side_effect=AssertionError("slow feature path used")):
            self.assertEqual(tracker.best().sid, 1)


if __name__ == "__main__":
    unittest.main()
