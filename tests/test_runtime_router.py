import argparse
import unittest

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
        "beam_width": 90,
        "max_selected_misses": 1,
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


if __name__ == "__main__":
    unittest.main()
