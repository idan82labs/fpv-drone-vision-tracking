#!/usr/bin/env python3
"""
Candidate-level track-before-detect prototype.

This keeps the current cheap proposal generator from motion_detector_v2, then
selects one box using a short-window beam search over plausible trajectories.
It is deliberately candidate-based first: cheap, debuggable, and comparable to
the existing per-frame/track selector before attempting dense x-y-t Viterbi.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import motion_detector_v2 as base  # noqa: E402
from selector_core import SequenceItem, StreamingViterbiSelector  # noqa: E402
from stabilized_residual_path_source import (  # noqa: E402
    BaseState as SRPSBaseState,
    ResidualCandidate as SRPSResidualCandidate,
    SRPSCandidate,
    SRPSConfig,
    SourceCandidate as SRPSSourceCandidate,
    StabilizedResidualPathSource,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--output_dir", default="results_tbd")
    p.add_argument("--downscale", type=float, default=0.5)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--save_every", type=int, default=30)
    p.add_argument("--write_video", action="store_true")
    p.add_argument(
        "--report_mode",
        choices=("full", "summary"),
        default="full",
        help="full stores every frame record; summary stores bounded aggregate stats for long-running onboard use.",
    )
    p.add_argument(
        "--stream_only",
        action="store_true",
        help="Do not accumulate selected/top-tube CSV rows in memory; use JSONL telemetry for live output.",
    )
    p.add_argument("--stats_window", type=int, default=4096, help="Rolling sample window for p90/p95 in summary report mode.")

    p.add_argument("--model", choices=("partial_affine", "full_affine", "homography", "auto"), default="auto")
    p.add_argument("--max_corners", type=int, default=900)
    p.add_argument("--quality", type=float, default=0.008)
    p.add_argument("--min_distance", type=int, default=7)
    p.add_argument("--ransac_px", type=float, default=2.0)
    p.add_argument(
        "--mask_selected_for_motion_model",
        action="store_true",
        help="Mask the previous selected box out of LK/RANSAC ego-motion estimation.",
    )
    p.add_argument(
        "--motion_model_fallback_identity",
        action="store_true",
        help=(
            "When LK/RANSAC cannot estimate frame motion, keep processing the frame "
            "with an identity warp so appearance/large-dark proposals are still exported."
        ),
    )

    p.add_argument("--threshold_sigma", type=float, default=5.5)
    p.add_argument("--threshold_percentile", type=float, default=96.5)
    p.add_argument("--min_threshold", type=float, default=9.0)
    p.add_argument("--appearance", choices=("off", "dark", "bright", "both"), default="dark")
    p.add_argument("--appearance_sigma", type=float, default=7.0)
    p.add_argument("--appearance_percentile", type=float, default=99.5)
    p.add_argument("--appearance_blur", type=float, default=6.0)
    p.add_argument("--min_app_residual", type=float, default=1.5)
    p.add_argument("--strong_appearance", type=float, default=45.0)
    p.add_argument("--min_appearance_mean", type=float, default=0.0)
    p.add_argument("--max_appearance_isolation", type=float, default=999.0)

    p.add_argument("--min_area", type=int, default=3)
    p.add_argument("--max_area", type=int, default=650)
    p.add_argument("--min_fill", type=float, default=0.08)
    p.add_argument("--max_aspect", type=float, default=5.0)
    p.add_argument("--border_frac", type=float, default=0.025)
    p.add_argument("--temporal_radius", type=int, default=7)

    p.add_argument("--kinematic_gate", action="store_true")
    p.add_argument("--horizontal_fov_deg", type=float, default=120.0)
    p.add_argument("--max_relative_speed_mps", type=float, default=10.0)
    p.add_argument("--min_range_m", type=float, default=2.0)
    p.add_argument("--kinematic_slack_px", type=float, default=20.0)
    p.add_argument("--max_match_dist", type=float, default=48.0)

    p.add_argument("--beam_width", type=int, default=90)
    p.add_argument("--top_k_candidates", type=int, default=80)
    p.add_argument("--map_peaks", action="store_true")
    p.add_argument("--map_radii", default="2,3,5")
    p.add_argument("--map_top_k", type=int, default=120)
    p.add_argument("--map_score_floor", type=float, default=1.8)
    p.add_argument("--map_nms_px", type=int, default=5)
    p.add_argument("--map_dark_weight", type=float, default=1.15)
    p.add_argument("--map_residual_weight", type=float, default=0.65)
    p.add_argument("--map_line_weight", type=float, default=0.35)
    p.add_argument("--map_texture_weight", type=float, default=0.08)
    p.add_argument("--map_score_weight", type=float, default=0.9)
    p.add_argument("--micro_map_score_floor", type=float, default=0.95)
    p.add_argument("--micro_map_edge_score_floor", type=float, default=1.55)
    p.add_argument("--micro_map_max_radius", type=int, default=1)
    p.add_argument("--micro_map_max_line", type=float, default=0.35)
    p.add_argument("--micro_map_edge_max_line", type=float, default=0.65)
    p.add_argument("--micro_map_max_texture", type=float, default=0.95)
    p.add_argument("--native_micro_peaks", action="store_true")
    p.add_argument("--native_micro_radii", default="2,3")
    p.add_argument("--native_micro_top_k", type=int, default=40)
    p.add_argument("--native_micro_score_floor", type=float, default=1.25)
    p.add_argument("--native_micro_nms_px", type=int, default=9)
    p.add_argument("--native_micro_line_weight", type=float, default=0.25)
    p.add_argument("--native_micro_texture_weight", type=float, default=0.05)
    p.add_argument("--native_micro_score_weight", type=float, default=1.0)
    p.add_argument("--native_roi_score", action="store_true")
    p.add_argument("--native_roi_weight", type=float, default=0.35)
    p.add_argument("--native_roi_neutral", type=float, default=1.0)
    p.add_argument("--large_dark_peaks", action="store_true")
    p.add_argument("--large_dark_top_k", type=int, default=40)
    p.add_argument("--large_dark_score_floor", type=float, default=35.0)
    p.add_argument("--large_dark_nms_px", type=float, default=18.0)
    p.add_argument("--large_dark_box_full", type=float, default=28.0)
    p.add_argument("--large_dark_bg_sigma", type=float, default=5.0)
    p.add_argument("--large_dark_score_weight", type=float, default=0.08)
    p.add_argument("--hybrid_coast_proposals", action="store_true")
    p.add_argument("--hybrid_coast_top_k", type=int, default=18)
    p.add_argument("--hybrid_coast_max_misses", type=int, default=8)
    p.add_argument("--hybrid_coast_min_hits", type=int, default=5)
    p.add_argument("--hybrid_coast_min_verified_score", type=float, default=18.0)
    p.add_argument("--hybrid_coast_min_evidence", type=float, default=0.1)
    p.add_argument("--hybrid_coast_offsets", default="0:0,-4:0,4:0,0:-4,0:4,-7:0,7:0,0:-7,0:7")
    p.add_argument("--hybrid_coast_score_weight", type=float, default=0.18)
    p.add_argument("--hybrid_coast_highres_recenter", action="store_true")
    p.add_argument("--hybrid_coast_recenter_radius_det_px", type=float, default=6.0)
    p.add_argument("--hybrid_coast_recenter_radii", default="2,3,4")
    p.add_argument("--hybrid_coast_recenter_peaks", type=int, default=3)
    p.add_argument("--hybrid_coast_recenter_texture_weight", type=float, default=0.010)
    p.add_argument("--hybrid_coast_recenter_shift_penalty", type=float, default=0.20)
    p.add_argument("--hybrid_coast_recenter_score_weight", type=float, default=0.70)
    p.add_argument(
        "--target_local_recovery_proposals",
        action="store_true",
        help="Opt-in recent-lock local high-res recovery branch around the last emitted track prediction.",
    )
    p.add_argument("--target_local_recovery_top_k", type=int, default=8)
    p.add_argument("--target_local_recovery_max_seed_gap", type=int, default=5)
    p.add_argument("--target_local_recovery_min_hits", type=int, default=3)
    p.add_argument("--target_local_recovery_min_verified_score", type=float, default=6.0)
    p.add_argument("--target_local_recovery_search_radius_det_px", type=float, default=18.0)
    p.add_argument("--target_local_recovery_radii", default="2,3,4")
    p.add_argument("--target_local_recovery_texture_weight", type=float, default=0.010)
    p.add_argument("--target_local_recovery_shift_penalty", type=float, default=0.020)
    p.add_argument("--target_local_recovery_score_weight", type=float, default=0.95)
    p.add_argument("--target_local_recovery_state_score_weight", type=float, default=0.10)
    p.add_argument("--target_local_recovery_box_det_px", type=float, default=4.0)
    p.add_argument(
        "--target_local_recovery_predictor",
        choices=("previous", "state_velocity", "clamped_velocity", "path_bank"),
        default="clamped_velocity",
        help="Prediction used to center target-local recovery. Clamped avoids poisoned beam velocity jumps.",
    )
    p.add_argument("--target_local_recovery_max_velocity_px", type=float, default=3.0)
    p.add_argument(
        "--target_local_recovery_path_bank_offsets",
        default="0:0,4:0,-4:0,0:4,0:-4,4:4,4:-4,-4:4,-4:-4",
        help="Detector-pixel dx:dy offsets for path-bank target-local recovery.",
    )
    p.add_argument(
        "--target_local_recovery_path_bank_velocity_scales",
        default="0,0.5,1",
        help="Velocity multipliers for path-bank target-local recovery.",
    )
    p.add_argument(
        "--target_local_anchor_bank_proposals",
        action="store_true",
        help="Opt-in bounded target-local anchor-bank proposal source decoupled from selected-state feedback.",
    )
    p.add_argument("--target_local_anchor_bank_max_anchors", type=int, default=2)
    p.add_argument("--target_local_anchor_bank_max_predictions", type=int, default=10)
    p.add_argument("--target_local_anchor_bank_peaks_per_prediction", type=int, default=2)
    p.add_argument("--target_local_anchor_bank_top_k", type=int, default=8)
    p.add_argument("--target_local_anchor_bank_max_ms", type=float, default=4.0)
    p.add_argument("--target_local_anchor_bank_max_pending", type=int, default=8)
    p.add_argument("--target_local_anchor_bank_max_promotion_candidates", type=int, default=24)
    p.add_argument(
        "--target_local_anchor_bank_allowed_sources",
        default="candidate_local_recenter,target_local_recovery",
        help=(
            "Comma-separated candidate sources allowed to create/refresh runtime target-local anchors. "
            "Default excludes target_local_anchor_bank so anchors cannot self-promote."
        ),
    )
    p.add_argument("--target_local_anchor_bank_promotion_hits", type=int, default=3)
    p.add_argument("--target_local_anchor_bank_offsets", default="0:0,4:0,-4:0,0:4,0:-4")
    p.add_argument("--target_local_anchor_bank_radii", default="2,3,4")
    p.add_argument("--target_local_anchor_bank_texture_weight", type=float, default=0.010)
    p.add_argument("--target_local_anchor_bank_shift_penalty", type=float, default=0.020)
    p.add_argument("--target_local_anchor_bank_score_weight", type=float, default=0.95)
    p.add_argument("--target_local_anchor_bank_anchor_trust_weight", type=float, default=0.20)
    p.add_argument("--target_local_anchor_bank_min_map_score", type=float, default=0.20)
    p.add_argument("--target_local_anchor_bank_min_side", type=float, default=4.0)
    p.add_argument("--target_local_anchor_bank_max_side", type=float, default=12.0)
    p.add_argument("--target_local_anchor_bank_max_line_context", type=float, default=0.85)
    p.add_argument("--target_local_anchor_bank_max_attached_support", type=float, default=16.0)
    p.add_argument("--target_local_anchor_bank_continuity_px", type=float, default=8.0)
    p.add_argument("--target_local_anchor_bank_pending_ttl", type=int, default=3)
    p.add_argument("--target_local_anchor_bank_anchor_ttl", type=int, default=12)
    p.add_argument("--target_local_anchor_bank_quarantine_ttl", type=int, default=18)
    p.add_argument("--target_local_anchor_bank_quarantine_radius_px", type=float, default=12.0)
    p.add_argument(
        "--stabilized_residual_path_source",
        action="store_true",
        help="Opt-in seed-gated stabilized residual path source. Emits candidates only; no selected-output override.",
    )
    p.add_argument(
        "--srps_recovery_mode",
        choices=("terrain_no_sky_only", "surface_or_low_confidence", "all"),
        default="terrain_no_sky_only",
    )
    p.add_argument("--srps_source_top_k", type=int, default=80)
    p.add_argument(
        "--srps_source_candidates",
        default="large_dark,map,candidate,motion,appearance,native_map,candidate_local_recenter,target_local_recovery",
        help="Comma-separated candidate sources allowed to create SRPS seeds.",
    )
    p.add_argument(
        "--srps_residual_source",
        choices=("temporal_combo", "temporal_dark_fullres", "residual_blur"),
        default="temporal_combo",
        help=(
            "SRPS residual peak source. temporal_combo keeps the legacy detector-space "
            "resized temporal map; temporal_dark_fullres extracts causal temporal-dark "
            "peaks at full resolution before projecting to detector coordinates."
        ),
    )
    p.add_argument("--srps_residual_top", type=int, default=200)
    p.add_argument("--srps_residual_nms_px", type=float, default=4.0)
    p.add_argument("--srps_residual_score_floor", type=float, default=0.0)
    p.add_argument("--srps_residual_box_det_px", type=float, default=8.0)
    p.add_argument("--srps_snap_radius", type=float, default=12.0)
    p.add_argument("--srps_gate", type=float, default=12.0)
    p.add_argument("--srps_seed_window", type=int, default=3)
    p.add_argument("--srps_seed_required_hits", type=int, default=2)
    p.add_argument("--srps_max_misses", type=int, default=2)
    p.add_argument("--srps_max_confirmed_age", type=int, default=90)
    p.add_argument("--srps_max_emit_per_frame", type=int, default=3)
    p.add_argument("--srps_seed_score_weight", type=float, default=1.2)
    p.add_argument("--srps_seed_rank_penalty", type=float, default=0.5)
    p.add_argument("--srps_seed_beta", type=float, default=0.8)
    p.add_argument("--srps_follow_beta", type=float, default=0.8)
    p.add_argument("--srps_score_weight", type=float, default=1.0)
    p.add_argument("--srps_disable_global_residual_seed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--srps_do_not_hijack_stable_track", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--srps_teacher_residual_proposals_csv", default="")
    p.add_argument("--srps_teacher_candidate_csv", default="")
    p.add_argument(
        "--srps_teacher_coord_scale",
        type=float,
        default=-1.0,
        help="Scale teacher residual/candidate centers into detector coordinates. <=0 uses --downscale.",
    )
    p.add_argument(
        "--srps_teacher_candidate_coord_scale",
        type=float,
        default=1.0,
        help="Scale teacher candidate boxes into detector coordinates. Existing detector top_tubes use 1.0.",
    )
    p.add_argument("--srps_dump_runtime_residual_proposals", action="store_true")
    p.add_argument("--srps_dump_seed_candidates", action="store_true")
    p.add_argument("--srps_dump_path_candidates", action="store_true")
    p.add_argument("--srps_multi_seed_compete", action="store_true")
    p.add_argument("--srps_confirm_requires_verified_seed", action="store_true")
    p.add_argument("--srps_reconfirm_on_recovery_epoch", action="store_true")
    p.add_argument("--srps_disable_stale_confirmed_paths", action="store_true")
    p.add_argument("--srps_false_path_veto", action="store_true")
    p.add_argument("--srps_verified_candidate_priority", action="store_true")
    p.add_argument("--srps_verified_seed_score", type=float, default=1.5)
    p.add_argument("--srps_verified_replacement_margin", type=float, default=1.0)
    p.add_argument("--srps_residual_density_radius", type=float, default=12.0)
    p.add_argument("--srps_residual_density_threshold", type=int, default=10)
    p.add_argument(
        "--target_local_state_select",
        action="store_true",
        help=(
            "Opt-in selector override for recent target-local recovery states. "
            "This does not create proposals; it only lets an already-generated "
            "target-local state replace a far-off selected lock."
        ),
    )
    p.add_argument("--target_local_state_select_error_px", type=float, default=9.0)
    p.add_argument("--target_local_state_select_improvement_px", type=float, default=6.0)
    p.add_argument("--target_local_state_select_max_pred_error_px", type=float, default=6.0)
    p.add_argument("--target_local_state_select_anchor_px", type=float, default=18.0)
    p.add_argument("--target_local_state_select_min_side", type=float, default=7.0)
    p.add_argument("--target_local_state_select_max_side", type=float, default=18.0)
    p.add_argument("--target_local_state_select_top_n", type=int, default=80)
    p.add_argument("--target_local_state_select_sources", default="target_local_recovery")
    p.add_argument(
        "--target_local_state_select_allow_missed_anchor",
        action="store_true",
        help="Allow a near-anchor one-frame coast state to beat current clutter in target-local selection.",
    )
    p.add_argument("--target_local_state_select_missed_max_misses", type=int, default=1)
    p.add_argument(
        "--replay_handoff_select",
        action="store_true",
        help=(
            "Opt-in selected-output override for source-scoped replay/recenter states. "
            "This is a bounded handoff over existing states; it does not create proposals "
            "or feed the chosen state back into BeamTBD."
        ),
    )
    p.add_argument("--replay_handoff_sources", default="candidate_local_recenter,track_only")
    p.add_argument("--replay_handoff_max_rank", type=int, default=80)
    p.add_argument("--replay_handoff_min_hits", type=int, default=1)
    p.add_argument("--replay_handoff_max_misses", type=int, default=2)
    p.add_argument("--replay_handoff_promote_hits", type=int, default=2)
    p.add_argument("--replay_handoff_window", type=int, default=3)
    p.add_argument("--replay_handoff_min_side", type=float, default=3.0)
    p.add_argument("--replay_handoff_max_side", type=float, default=14.0)
    p.add_argument("--replay_handoff_max_line_context", type=float, default=1.3)
    p.add_argument("--replay_handoff_max_attached_support", type=float, default=16.0)
    p.add_argument("--replay_handoff_min_map_score", type=float, default=0.20)
    p.add_argument("--replay_handoff_max_shift_det", type=float, default=18.0)
    p.add_argument(
        "--replay_handoff_diagnostic_same_frame",
        action="store_true",
        help="Emit eligible handoff states without waiting for the 2-of-N continuity commit.",
    )
    p.add_argument(
        "--replay_handoff_allow_target_local_recovery",
        action="store_true",
        help="Allow target_local_recovery states only when they pass local evidence confirmation.",
    )
    p.add_argument(
        "--candidate_local_recenter_proposals",
        action="store_true",
        help="Opt-in high-res local recentering around top cheap proposals before TBD ranking.",
    )
    p.add_argument("--candidate_local_recenter_seed_top_k", type=int, default=24)
    p.add_argument("--candidate_local_recenter_top_k", type=int, default=48)
    p.add_argument("--candidate_local_recenter_radius_det_px", type=float, default=16.0)
    p.add_argument("--candidate_local_recenter_radii", default="2,3,4")
    p.add_argument(
        "--candidate_local_recenter_seed_families",
        default="cheap",
        help="Comma-separated recenter seed families: cheap,track_state,previous_recenter.",
    )
    p.add_argument(
        "--candidate_local_recenter_seed_family_quota",
        default="",
        help="Comma-separated family:N seed quotas before global fill, e.g. cheap:24,track_state:24.",
    )
    p.add_argument("--candidate_local_recenter_track_state_seed_top_k", type=int, default=24)
    p.add_argument("--candidate_local_recenter_track_state_min_hits", type=int, default=2)
    p.add_argument("--candidate_local_recenter_track_state_max_misses", type=int, default=2)
    p.add_argument("--candidate_local_recenter_track_state_max_age", type=int, default=3)
    p.add_argument("--candidate_local_recenter_track_state_min_verified_score", type=float, default=-5.0)
    p.add_argument("--candidate_local_recenter_track_state_max_velocity_px", type=float, default=4.0)
    p.add_argument(
        "--candidate_local_recenter_track_only_replay",
        action="store_true",
        help=(
            "Opt-in one-frame-delayed recenter parents from post-update track_only states. "
            "This uses weak TBD path hypotheses on the next frame; it does not feed back into the same update."
        ),
    )
    p.add_argument("--candidate_local_recenter_track_only_replay_delay", type=int, default=1)
    p.add_argument("--candidate_local_recenter_track_only_replay_top_k", type=int, default=32)
    p.add_argument("--candidate_local_recenter_track_only_replay_rank_max", type=int, default=80)
    p.add_argument("--candidate_local_recenter_track_only_replay_max_misses", type=int, default=2)
    p.add_argument("--candidate_local_recenter_track_only_replay_min_hits", type=int, default=1)
    p.add_argument("--candidate_local_recenter_track_only_replay_max_age", type=int, default=4)
    p.add_argument("--candidate_local_recenter_track_only_replay_min_side", type=float, default=3.0)
    p.add_argument("--candidate_local_recenter_track_only_replay_max_side", type=float, default=14.0)
    p.add_argument("--candidate_local_recenter_raw_seed_top_k", type=int, default=48)
    p.add_argument("--candidate_local_recenter_raw_seed_rank_min", type=int, default=25)
    p.add_argument("--candidate_local_recenter_raw_seed_rank_max", type=int, default=400)
    p.add_argument("--candidate_local_recenter_raw_seed_grid_px", type=float, default=12.0)
    p.add_argument(
        "--candidate_local_recenter_raw_seed_source_quota",
        default="motion:20,appearance:12,map:12,native:12,large_dark:8",
        help="Comma-separated source-family:N quotas for raw_low_rank seed-only parents.",
    )
    p.add_argument(
        "--candidate_local_recenter_seed_mode",
        choices=("score", "spatial_grid"),
        default="score",
        help="How to choose recenter seed candidates. spatial_grid keeps local coverage for low-score terrain targets.",
    )
    p.add_argument("--candidate_local_recenter_seed_grid_px", type=float, default=24.0)
    p.add_argument(
        "--candidate_local_recenter_response_maps",
        default="compact_dark",
        help="Comma-separated recenter response maps: compact_dark,dog,blackhat.",
    )
    p.add_argument("--candidate_local_recenter_peaks_per_seed", type=int, default=4)
    p.add_argument("--candidate_local_recenter_texture_weight", type=float, default=0.010)
    p.add_argument("--candidate_local_recenter_shift_penalty", type=float, default=0.025)
    p.add_argument("--candidate_local_recenter_score_weight", type=float, default=1.10)
    p.add_argument("--candidate_local_recenter_seed_score_weight", type=float, default=0.035)
    p.add_argument("--candidate_local_recenter_box_det_px", type=float, default=4.0)
    p.add_argument(
        "--candidate_local_recenter_router_scope",
        choices=("all", "surface_context"),
        default="surface_context",
    )
    p.add_argument(
        "--runtime_mode",
        choices=("baseline", "clean_sky", "boundary", "surface", "auto"),
        default="baseline",
        help="Experimental state-conditioned runtime mode. baseline preserves the old behavior.",
    )
    p.add_argument(
        "--candidate_router",
        choices=("off", "log", "apply"),
        default="off",
        help="Candidate-local background router. auto mode logs by default; apply changes candidate selection.",
    )
    p.add_argument("--router_surface_source_penalty", type=float, default=2.5)
    p.add_argument("--router_line_penalty", type=float, default=1.5)
    p.add_argument("--router_surface_bonus", type=float, default=0.0)
    p.add_argument("--surface_branch_allow_acquisition", action="store_true")
    p.add_argument("--surface_branch_min_candidates", type=int, default=4)
    p.add_argument("--surface_branch_min_score", type=float, default=2.0)
    p.add_argument("--surface_branch_track_min_hits", type=int, default=6)
    p.add_argument("--surface_branch_track_min_score", type=float, default=24.0)
    p.add_argument("--surface_branch_track_rate", type=float, default=0.55)
    p.add_argument(
        "--surface_ranker_scope",
        choices=("all", "surface_backed", "surface_context"),
        default="all",
        help="When using a tube verifier/ranker, optionally apply it only to surface-backed tubes.",
    )
    p.add_argument("--surface_ranker_min_rate", type=float, default=0.45)
    p.add_argument("--surface_ranker_model", default="", help="Optional trained surface ranker .joblib bundle.")
    p.add_argument(
        "--surface_ranker_policy",
        choices=("off", "confidence_fallback"),
        default="off",
        help="Use learned surface ranker only as a confidence-gated selector fallback.",
    )
    p.add_argument(
        "--surface_ranker_gate",
        choices=(
            "none",
            "learned_not_map",
            "source_large_dark_or_appearance",
            "high_support",
            "high_texture_support",
            "large_dark_high_support",
            "low_sky_high_support",
            "negative_bg_pair",
            "support_negative_bg_pair",
        ),
        default="none",
        help="Optional learned-candidate gate for state-specific surface fallback experiments.",
    )
    p.add_argument("--surface_ranker_threshold", type=float, default=0.76)
    p.add_argument("--surface_ranker_top_n", type=int, default=80, help="Maximum baseline-ranked states to rescore; <=0 means all.")
    p.add_argument("--scenario_balance", action="store_true")
    p.add_argument("--scenario_pool_factor", type=float, default=3.0)
    p.add_argument("--scenario_sky_top_k", type=int, default=24)
    p.add_argument("--scenario_surface_top_k", type=int, default=34)
    p.add_argument("--scenario_boundary_top_k", type=int, default=20)
    p.add_argument("--scenario_large_top_k", type=int, default=20)
    p.add_argument("--scenario_coast_top_k", type=int, default=18)
    p.add_argument("--temporal_stack_peaks", action="store_true")
    p.add_argument("--temporal_stack_offsets", default="-8,-5,-3,-2,-1")
    p.add_argument("--temporal_stack_radii", default="2,3,4,5,7")
    p.add_argument("--temporal_stack_top_k", type=int, default=120)
    p.add_argument("--temporal_stack_min_frames", type=int, default=2)
    p.add_argument("--temporal_stack_nms_px", type=float, default=5.5)
    p.add_argument("--temporal_stack_halo_bases", type=int, default=32)
    p.add_argument(
        "--temporal_stack_halo_offsets",
        default="0:0,-6:0,6:0,0:-6,0:6,-9:0,9:0,0:-9,0:9,-6:-6,-6:6,6:-6,6:6",
    )
    p.add_argument("--temporal_stack_halo_penalty", type=float, default=0.025)
    p.add_argument("--temporal_stack_score_weight", type=float, default=0.78)
    p.add_argument("--temporal_stack_native_weight", type=float, default=0.55)
    p.add_argument("--temporal_stack_clahe_weight", type=float, default=0.25)
    p.add_argument(
        "--temporal_stack_candidate_local",
        action="store_true",
        help="Score temporal-stack evidence only around cheap candidate centers instead of full-frame peak maps.",
    )
    p.add_argument("--temporal_stack_seed_top_k", type=int, default=36)
    p.add_argument("--temporal_stack_local_halo_limit", type=int, default=5)
    p.add_argument(
        "--temporal_stack_direct_warp",
        action="store_true",
        help="Estimate each temporal-stack history frame directly to the current frame instead of composing frame-to-frame warps.",
    )
    p.add_argument("--window", type=int, default=9)
    p.add_argument("--min_path_hits", type=int, default=3)
    p.add_argument("--max_misses", type=int, default=2)
    p.add_argument("--max_selected_misses", type=int, default=1)
    p.add_argument("--selected_score", type=float, default=6.0)
    p.add_argument("--birth_penalty", type=float, default=1.8)
    p.add_argument("--miss_penalty", type=float, default=1.1)
    p.add_argument("--obs_weight", type=float, default=0.95)
    p.add_argument("--pair_weight", type=float, default=0.85)
    p.add_argument("--speed_weight", type=float, default=0.16)
    p.add_argument("--accel_weight", type=float, default=0.12)
    p.add_argument("--static_penalty_weight", type=float, default=0.0)
    p.add_argument("--static_penalty_px", type=float, default=1.2)
    p.add_argument("--line_weight", type=float, default=0.35)
    p.add_argument("--support_penalty_weight", type=float, default=0.0)
    p.add_argument("--support_penalty_threshold", type=float, default=3.5)
    p.add_argument("--app_low_residual_penalty", type=float, default=0.0)
    p.add_argument("--app_low_residual_px", type=float, default=4.0)
    p.add_argument("--tube_verifier", choices=("off", "heuristic", "likelihood"), default="off")
    p.add_argument("--tube_verifier_weight", type=float, default=1.0)
    p.add_argument("--tube_verifier_floor", type=float, default=-999.0)
    p.add_argument("--density_penalty_weight", type=float, default=0.0)
    p.add_argument("--selection_margin", type=float, default=0.0)
    p.add_argument("--bg_motion_transition", action="store_true")
    p.add_argument("--bg_pair_geometry", action="store_true")
    p.add_argument("--local_pair_norm", action="store_true")
    p.add_argument("--pair_search_penalty", type=float, default=0.0)
    p.add_argument("--sky_bonus_weight", type=float, default=0.0)
    p.add_argument("--sky_rescue", action="store_true")
    p.add_argument("--sky_rescue_score", type=float, default=6.0)
    p.add_argument("--sky_rescue_min_sky", type=float, default=0.25)
    p.add_argument("--sky_rescue_max_line", type=float, default=0.45)
    p.add_argument("--sky_rescue_max_support", type=float, default=1.4)
    p.add_argument("--sky_rescue_min_hits", type=int, default=4)
    p.add_argument("--pair_radius", type=int, default=3)
    p.add_argument("--pair_min_step_px", type=float, default=1.0)
    p.add_argument("--top_k_debug", type=int, default=12)
    p.add_argument("--export_top_tubes", type=int, default=0)
    p.add_argument(
        "--delayed_sequence_select",
        action="store_true",
        help="Use a lightweight delayed Viterbi selector over recent top states for selected_tracks.csv.",
    )
    p.add_argument("--delayed_sequence_top_n", type=int, default=20)
    p.add_argument(
        "--delayed_sequence_score_source",
        choices=("verified", "surface_ranker"),
        default="verified",
        help="Score used inside delayed sequence selection. surface_ranker requires --surface_ranker_model.",
    )
    p.add_argument("--delayed_sequence_min_hits", type=int, default=1)
    p.add_argument("--delayed_sequence_window", type=int, default=15)
    p.add_argument("--delayed_sequence_max_jump_px", type=float, default=10.0)
    p.add_argument("--delayed_sequence_transition_weight", type=float, default=1.5)
    p.add_argument("--delayed_sequence_threshold", type=float, default=0.0)
    p.add_argument(
        "--delayed_sequence_acquire_threshold",
        type=float,
        default=None,
        help="Optional score required to acquire delayed-sequence output. Enables acquire/keep hysteresis.",
    )
    p.add_argument("--delayed_sequence_acquire_hits", type=int, default=1)
    p.add_argument(
        "--delayed_sequence_keep_threshold",
        type=float,
        default=None,
        help="Optional score required to keep delayed-sequence output after acquisition. Defaults to --delayed_sequence_threshold.",
    )
    p.add_argument("--delayed_sequence_lost_patience", type=int, default=0)
    p.add_argument(
        "--delayed_sequence_require_floor",
        action="store_true",
        help="Require each delayed-sequence state to pass the immediate detector floor. Safer but can lose recall.",
    )
    p.add_argument(
        "--delayed_sequence_commit_prefix",
        action="store_true",
        help="Commit the emitted delayed-sequence branch before future pops. Reduces jump risk but can lose recall.",
    )
    p.add_argument(
        "--selected_jsonl",
        default="",
        help="Optional live newline-delimited selected-box stream for onboard consumers.",
    )
    p.add_argument(
        "--telemetry_jsonl",
        default="",
        help="Optional per-processed-frame telemetry stream with selected/null status and latency.",
    )
    p.add_argument("--draw_debug", action="store_true")
    return p.parse_args()


@dataclass
class PathState:
    sid: int
    bbox: tuple[int, int, int, int]
    vx: float = 0.0
    vy: float = 0.0
    last_frame: int = 0
    misses: int = 0
    age: int = 1
    hits: int = 1
    last_candidate: base.Candidate | None = None
    contribs: list[float] = field(default_factory=list)
    hit_flags: list[bool] = field(default_factory=list)
    history: list[tuple[int, tuple[int, int, int, int], float, bool]] = field(default_factory=list)
    candidate_history: list[dict | None] = field(default_factory=list)
    pair_history: list[float] = field(default_factory=list)
    pair_raw_history: list[float] = field(default_factory=list)
    pair_bg_history: list[float] = field(default_factory=list)
    pair_bg_local_history: list[float] = field(default_factory=list)
    align_gain_history: list[float] = field(default_factory=list)
    bg_dist_history: list[float] = field(default_factory=list)
    cv_resid_history: list[float] = field(default_factory=list)
    bg_minus_cv_history: list[float] = field(default_factory=list)
    cand_density_history: list[float] = field(default_factory=list)
    speed_history: list[float] = field(default_factory=list)
    accel_history: list[float] = field(default_factory=list)

    def score(self) -> float:
        return float(sum(self.contribs))

    def hit_count(self) -> int:
        return int(sum(1 for h in self.hit_flags if h))

    def predict_bbox(self, w_img: int, h_img: int) -> tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return base.clip_bbox_float((x + self.vx, y + self.vy, w, h), w_img, h_img)


@dataclass
class TemporalStackFrame:
    gray_full: np.ndarray
    h_to_current_full: np.ndarray


@dataclass
class FrameRouterDecision:
    mode: str
    allow_surface_extras: bool
    max_candidates: int
    confidence: float
    features: dict[str, float]


@dataclass
class TargetLocalRecoverySeed:
    sid: int
    bbox: tuple[int, int, int, int]
    vx: float
    vy: float
    frame_no: int
    hits: int
    verified_score: float

    @classmethod
    def from_state(
        cls,
        frame_no: int,
        st: PathState,
        verified_score: float,
    ) -> "TargetLocalRecoverySeed":
        return cls(
            sid=st.sid,
            bbox=st.bbox,
            vx=st.vx,
            vy=st.vy,
            frame_no=frame_no,
            hits=st.hit_count(),
            verified_score=float(verified_score),
        )

    def age(self, frame_no: int) -> int:
        return int(frame_no - self.frame_no)

    def predict_bbox(self, frame_no: int, w_img: int, h_img: int) -> tuple[int, int, int, int]:
        dt = max(1, self.age(frame_no))
        x, y, w, h = self.bbox
        return base.clip_bbox_float((x + self.vx * dt, y + self.vy * dt, w, h), w_img, h_img)


@dataclass
class AnchorQuarantine:
    cx: float
    cy: float
    frame_no: int
    ttl: int
    radius_px: float
    reason: str

    def active(self, frame_no: int) -> bool:
        return frame_no - self.frame_no <= self.ttl

    def contains(self, bbox: tuple[int, int, int, int], frame_no: int) -> bool:
        if not self.active(frame_no):
            return False
        cx, cy = base.bbox_center(bbox)
        return math.hypot(cx - self.cx, cy - self.cy) <= self.radius_px


@dataclass
class RuntimeLocalAnchor:
    aid: int
    bbox: tuple[int, int, int, int]
    frame_no: int
    vx: float = 0.0
    vy: float = 0.0
    trust_score: float = 1.0
    hit_streak: int = 1
    miss_streak: int = 0
    source: str = ""
    router_state: str = "unrouted"
    last_selected_sid: int | None = None
    poison_flags: set[str] = field(default_factory=set)

    def age(self, frame_no: int) -> int:
        return int(frame_no - self.frame_no)

    def center(self) -> tuple[float, float]:
        return base.bbox_center(self.bbox)

    def update_from_candidate(self, frame_no: int, cand: base.Candidate, trust_delta: float) -> None:
        prev_cx, prev_cy = self.center()
        cx, cy = base.bbox_center(cand.bbox)
        dt = max(1, frame_no - self.frame_no)
        self.vx = (cx - prev_cx) / dt
        self.vy = (cy - prev_cy) / dt
        self.bbox = cand.bbox
        self.frame_no = frame_no
        self.source = cand.source
        self.router_state = getattr(cand, "router_state", "unrouted")
        self.hit_streak += 1
        self.miss_streak = 0
        self.trust_score = min(6.0, self.trust_score + trust_delta)


@dataclass
class PendingLocalAnchor:
    pid: int
    bbox: tuple[int, int, int, int]
    frame_no: int
    source: str
    router_state: str
    hit_streak: int = 1
    score: float = 0.0
    distinct_hit_frames: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.distinct_hit_frames:
            self.distinct_hit_frames.add(self.frame_no)

    def center(self) -> tuple[float, float]:
        return base.bbox_center(self.bbox)


class RuntimeAnchorBank:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.anchors: list[RuntimeLocalAnchor] = []
        self.pending: list[PendingLocalAnchor] = []
        self.quarantines: list[AnchorQuarantine] = []
        self.next_aid = 1
        self.next_pid = 1
        self.last_events: list[dict[str, object]] = []

    def prune(self, frame_no: int) -> None:
        ttl = max(1, int(getattr(self.args, "target_local_anchor_bank_anchor_ttl", 12)))
        self.anchors = [
            anchor
            for anchor in self.anchors
            if anchor.age(frame_no) <= ttl and anchor.trust_score > 0.0 and anchor.miss_streak <= ttl
        ]
        pending_ttl = max(1, int(getattr(self.args, "target_local_anchor_bank_pending_ttl", 3)))
        self.pending = [p for p in self.pending if frame_no - p.frame_no <= pending_ttl]
        max_pending = max(1, int(getattr(self.args, "target_local_anchor_bank_max_pending", 8)))
        self.pending.sort(key=lambda p: (p.hit_streak, p.score), reverse=True)
        self.pending = self.pending[:max_pending]
        self.quarantines = [q for q in self.quarantines if q.active(frame_no)]

    def in_quarantine(self, bbox: tuple[int, int, int, int], frame_no: int) -> bool:
        return any(q.contains(bbox, frame_no) for q in self.quarantines)

    def candidate_allowed(self, cand: base.Candidate, frame_no: int) -> tuple[bool, str]:
        allowed_sources = _parse_source_set(
            getattr(
                self.args,
                "target_local_anchor_bank_allowed_sources",
                "candidate_local_recenter,target_local_recovery",
            )
        )
        if allowed_sources and cand.source not in allowed_sources:
            return False, "source"
        w = float(cand.bbox[2])
        h = float(cand.bbox[3])
        side = max(w, h)
        if side < getattr(self.args, "target_local_anchor_bank_min_side", 4.0):
            return False, "small"
        if side > getattr(self.args, "target_local_anchor_bank_max_side", 12.0):
            return False, "large"
        if float(getattr(cand, "line_context", 0.0)) > getattr(self.args, "target_local_anchor_bank_max_line_context", 0.85):
            return False, "line"
        if float(getattr(cand, "attached_support", 0.0)) > getattr(
            self.args,
            "target_local_anchor_bank_max_attached_support",
            16.0,
        ):
            return False, "attached"
        if float(getattr(cand, "map_score", 0.0)) < getattr(self.args, "target_local_anchor_bank_min_map_score", 0.20):
            return False, "map"
        if self.in_quarantine(cand.bbox, frame_no):
            return False, "quarantine"
        return True, "ok"

    @staticmethod
    def _candidate_event_fields(cand: base.Candidate) -> dict[str, object]:
        cx, cy = base.bbox_center(cand.bbox)
        parent_source = getattr(cand, "recenter_parent_source", getattr(cand, "target_local_anchor_source", ""))
        out: dict[str, object] = {
            "candidate_source": cand.source,
            "parent_source": parent_source,
            "candidate_box": list(cand.bbox),
            "candidate_cx": round(cx, 3),
            "candidate_cy": round(cy, 3),
            "map_score": round(float(getattr(cand, "map_score", 0.0)), 3),
            "score": round(float(getattr(cand, "score", 0.0)), 3),
            "line_context": round(float(getattr(cand, "line_context", 0.0)), 3),
            "attached_support": round(float(getattr(cand, "attached_support", 0.0)), 3),
            "router_state": getattr(cand, "router_state", "unrouted"),
        }
        for attr in (
            "recenter_shift_det",
            "recenter_peak_radius",
            "recenter_peak_score",
            "recenter_second_peak_margin",
            "recenter_seed_rank",
            "recenter_seed_score",
        ):
            if hasattr(cand, attr):
                value = getattr(cand, attr)
                out[attr] = round(float(value), 3) if isinstance(value, (float, int)) else value
        return out

    def add_quarantine(self, bbox: tuple[int, int, int, int], frame_no: int, reason: str) -> None:
        cx, cy = base.bbox_center(bbox)
        self.quarantines.append(
            AnchorQuarantine(
                cx=cx,
                cy=cy,
                frame_no=frame_no,
                ttl=max(1, int(getattr(self.args, "target_local_anchor_bank_quarantine_ttl", 18))),
                radius_px=float(getattr(self.args, "target_local_anchor_bank_quarantine_radius_px", 12.0)),
                reason=reason,
            )
        )
        self.last_events.append({"event": "quarantine", "frame": frame_no, "reason": reason, "x": round(cx, 3), "y": round(cy, 3)})

    def create_anchor(
        self,
        cand: base.Candidate,
        frame_no: int,
        trust: float,
        pending_id: int | None = None,
    ) -> RuntimeLocalAnchor:
        anchor = RuntimeLocalAnchor(
            aid=self.next_aid,
            bbox=cand.bbox,
            frame_no=frame_no,
            trust_score=trust,
            source=cand.source,
            router_state=getattr(cand, "router_state", "unrouted"),
        )
        self.next_aid += 1
        self.anchors.append(anchor)
        self.last_events.append(
            {
                "event": "anchor_created",
                "event_type": "promote",
                "frame": frame_no,
                "aid": anchor.aid,
                "pending_id": pending_id if pending_id is not None else "",
                "source": cand.source,
                "anchor_trust": round(anchor.trust_score, 3),
                **self._candidate_event_fields(cand),
            }
        )
        return anchor

    def update_from_candidates(self, frame_no: int, candidates: list[base.Candidate]) -> None:
        self.last_events = []
        self.prune(frame_no)
        continuity_px = float(getattr(self.args, "target_local_anchor_bank_continuity_px", 8.0))
        rejected: dict[str, int] = {}
        max_promotion_candidates = max(1, int(getattr(self.args, "target_local_anchor_bank_max_promotion_candidates", 24)))
        sorted_candidates = sorted(candidates, key=lambda c: float(getattr(c, "map_score", c.score)), reverse=True)[
            :max_promotion_candidates
        ]
        for cand in sorted_candidates:
            ok, reason = self.candidate_allowed(cand, frame_no)
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                self.last_events.append(
                    {
                        "event": "anchor_candidate_rejected",
                        "event_type": "reject",
                        "frame": frame_no,
                        "rejection_reason": reason,
                        **self._candidate_event_fields(cand),
                    }
                )
                continue
            matched = False
            for anchor in self.anchors:
                if center_distance(anchor.bbox, cand.bbox) <= continuity_px:
                    anchor.update_from_candidate(frame_no, cand, trust_delta=0.6)
                    self.last_events.append(
                        {
                            "event": "anchor_refreshed",
                            "event_type": "anchor_hit",
                            "frame": frame_no,
                            "aid": anchor.aid,
                            "anchor_age": anchor.age(frame_no),
                            "anchor_trust": round(anchor.trust_score, 3),
                            **self._candidate_event_fields(cand),
                        }
                    )
                    matched = True
                    break
            if matched:
                continue
            for pending in self.pending:
                if center_distance(pending.bbox, cand.bbox) <= continuity_px:
                    if pending.frame_no == frame_no:
                        rejected["same_frame_pending"] = rejected.get("same_frame_pending", 0) + 1
                        self.last_events.append(
                            {
                                "event": "anchor_candidate_rejected",
                                "event_type": "reject",
                                "frame": frame_no,
                                "rejection_reason": "same_frame_pending",
                                "pending_id": pending.pid,
                                **self._candidate_event_fields(cand),
                            }
                        )
                        matched = True
                        break
                    distinct_frame = frame_no not in pending.distinct_hit_frames
                    pending.bbox = cand.bbox
                    pending.frame_no = frame_no
                    pending.source = cand.source
                    pending.router_state = getattr(cand, "router_state", "unrouted")
                    if distinct_frame:
                        pending.hit_streak += 1
                        pending.distinct_hit_frames.add(frame_no)
                    pending.score = max(pending.score, float(getattr(cand, "map_score", cand.score)))
                    matched = True
                    self.last_events.append(
                        {
                            "event": "pending_anchor_hit",
                            "event_type": "pending_hit",
                            "frame": frame_no,
                            "pending_id": pending.pid,
                            "hit_streak": pending.hit_streak,
                            "distinct_hit_frames": len(pending.distinct_hit_frames),
                            "distinct_frame": int(distinct_frame),
                            **self._candidate_event_fields(cand),
                        }
                    )
                    promotion_hits = max(2, int(getattr(self.args, "target_local_anchor_bank_promotion_hits", 3)))
                    if pending.hit_streak >= promotion_hits and len(pending.distinct_hit_frames) >= promotion_hits:
                        self.create_anchor(cand, frame_no, trust=max(1.0, min(3.0, pending.score)), pending_id=pending.pid)
                        self.pending.remove(pending)
                    break
            if matched:
                continue
            max_pending = max(1, int(getattr(self.args, "target_local_anchor_bank_max_pending", 8)))
            if len(self.pending) >= max_pending:
                weakest = min(self.pending, key=lambda p: (p.hit_streak, p.score))
                cand_score = float(getattr(cand, "map_score", cand.score))
                if (weakest.hit_streak, weakest.score) >= (1, cand_score):
                    rejected["pending_full"] = rejected.get("pending_full", 0) + 1
                    self.last_events.append(
                        {
                            "event": "anchor_candidate_rejected",
                            "event_type": "reject",
                            "frame": frame_no,
                            "rejection_reason": "pending_full",
                            **self._candidate_event_fields(cand),
                        }
                    )
                    continue
                self.pending.remove(weakest)
            pending_id = self.next_pid
            self.next_pid += 1
            self.pending.append(
                PendingLocalAnchor(
                    pid=pending_id,
                    bbox=cand.bbox,
                    frame_no=frame_no,
                    source=cand.source,
                    router_state=getattr(cand, "router_state", "unrouted"),
                    score=float(getattr(cand, "map_score", cand.score)),
                )
            )
            self.last_events.append(
                {
                    "event": "pending_anchor_started",
                    "event_type": "pending_start",
                    "frame": frame_no,
                    "pending_id": pending_id,
                    **self._candidate_event_fields(cand),
                }
            )
        if rejected:
            self.last_events.append({"event": "anchor_rejected", "frame": frame_no, "reasons": rejected})
        self.anchors.sort(key=lambda a: (a.trust_score, a.hit_streak), reverse=True)
        max_anchors = max(1, int(getattr(self.args, "target_local_anchor_bank_max_anchors", 2)))
        self.anchors = self.anchors[:max_anchors]

    def confirm_selected(self, frame_no: int, selected: "PathState | None") -> None:
        if selected is None:
            for anchor in self.anchors:
                anchor.trust_score = max(0.0, anchor.trust_score - 0.1)
            return
        selected_source = selected.last_candidate.source if selected.last_candidate is not None else ""
        if selected_source not in {"candidate_local_recenter", "target_local_recovery", "target_local_anchor_bank"}:
            return
        for anchor in self.anchors:
            if center_distance(anchor.bbox, selected.bbox) <= float(
                getattr(self.args, "target_local_anchor_bank_continuity_px", 8.0)
            ):
                anchor.last_selected_sid = selected.sid
                anchor.trust_score = min(6.0, anchor.trust_score + 0.25)
            else:
                anchor.trust_score = max(0.0, anchor.trust_score - 0.05)

    def poison_static_or_attached(self, frame_no: int) -> None:
        kept: list[RuntimeLocalAnchor] = []
        for anchor in self.anchors:
            if "attached" in anchor.poison_flags or "static" in anchor.poison_flags:
                self.add_quarantine(anchor.bbox, frame_no, ",".join(sorted(anchor.poison_flags)))
                continue
            kept.append(anchor)
        self.anchors = kept


def target_local_seed_prediction_bbox(
    seed: TargetLocalRecoverySeed,
    frame_no: int,
    w_img: int,
    h_img: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    predictor = getattr(args, "target_local_recovery_predictor", "clamped_velocity")
    if predictor == "state_velocity":
        return seed.predict_bbox(frame_no, w_img, h_img)
    x, y, w, h = seed.bbox
    if predictor == "previous":
        return base.clip_bbox_float((x, y, w, h), w_img, h_img)

    dt = max(1, seed.age(frame_no))
    vx = float(seed.vx)
    vy = float(seed.vy)
    max_step = max(0.0, float(getattr(args, "target_local_recovery_max_velocity_px", 3.0))) * dt
    speed = math.hypot(vx * dt, vy * dt)
    if max_step > 0.0 and speed > max_step:
        scale = max_step / max(1e-6, speed)
        vx *= scale
        vy *= scale
    return base.clip_bbox_float((x + vx * dt, y + vy * dt, w, h), w_img, h_img)


def target_local_seed_prediction_bank(
    seed: TargetLocalRecoverySeed,
    frame_no: int,
    w_img: int,
    h_img: int,
    args: argparse.Namespace,
) -> list[tuple[tuple[int, int, int, int], str]]:
    predictor = getattr(args, "target_local_recovery_predictor", "clamped_velocity")
    if predictor != "path_bank":
        return [(target_local_seed_prediction_bbox(seed, frame_no, w_img, h_img, args), predictor)]

    x, y, w, h = seed.bbox
    dt = max(1, seed.age(frame_no))
    vx = float(seed.vx)
    vy = float(seed.vy)
    max_step = max(0.0, float(getattr(args, "target_local_recovery_max_velocity_px", 3.0))) * dt
    speed = math.hypot(vx * dt, vy * dt)
    if max_step > 0.0 and speed > max_step:
        scale = max_step / max(1e-6, speed)
        vx *= scale
        vy *= scale

    offsets = parse_xy_offsets(getattr(args, "target_local_recovery_path_bank_offsets", "0:0"))
    velocity_scales = parse_float_list(getattr(args, "target_local_recovery_path_bank_velocity_scales", "0,1"))
    bases: list[tuple[float, float, str]] = [(float(x), float(y), "previous")]
    for vel_scale in velocity_scales:
        bases.append((x + vx * dt * vel_scale, y + vy * dt * vel_scale, f"vel{vel_scale:g}"))

    out: list[tuple[tuple[int, int, int, int], str]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for bx, by, method in bases:
        for dx, dy in offsets:
            bbox = base.clip_bbox_float((bx + dx, by + dy, w, h), w_img, h_img)
            suffix = "center" if abs(dx) < 1e-6 and abs(dy) < 1e-6 else f"off{dx:g}_{dy:g}"
            key = (bbox[0], bbox[1], bbox[2], bbox[3], method)
            if key in seen:
                continue
            seen.add(key)
            out.append((bbox, f"{method}_{suffix}"))
    return out or [(target_local_seed_prediction_bbox(seed, frame_no, w_img, h_img, args), "path_bank")]


def robust_sigma(img: np.ndarray) -> float:
    vals = img.reshape(-1).astype(np.float32)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(3.0, 1.4826 * mad)


def patch_mean(img: np.ndarray, cx: float, cy: float, radius: int) -> float:
    h, w = img.shape[:2]
    r = max(1, int(radius))
    x0 = max(0, int(round(cx)) - r)
    y0 = max(0, int(round(cy)) - r)
    x1 = min(w, int(round(cx)) + r + 1)
    y1 = min(h, int(round(cy)) + r + 1)
    patch = img[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else 0.0


def annulus_stats(
    img: np.ndarray,
    cx: float,
    cy: float,
    inner_radius: int,
    outer_radius: int,
) -> tuple[float, float]:
    h, w = img.shape[:2]
    outer = max(inner_radius + 1, int(outer_radius))
    inner = max(1, int(inner_radius))
    x0 = max(0, int(round(cx)) - outer)
    y0 = max(0, int(round(cy)) - outer)
    x1 = min(w, int(round(cx)) + outer + 1)
    y1 = min(h, int(round(cy)) + outer + 1)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0, 3.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    mask = (dist2 > inner * inner) & (dist2 <= outer * outer)
    vals = patch[mask]
    if vals.size < 4:
        vals = patch.reshape(-1)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, max(3.0, 1.4826 * mad)


def warp_bbox(
    bbox: tuple[int, int, int, int],
    h_mat: np.ndarray | None,
    w_img: int,
    h_img: int,
) -> tuple[int, int, int, int]:
    if h_mat is None:
        return bbox
    x, y, w, h = bbox
    cx, cy = base.bbox_center(bbox)
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    try:
        warped = cv2.perspectiveTransform(pt, h_mat).reshape(2)
    except cv2.error:
        return bbox
    if not np.isfinite(warped).all():
        return bbox
    return base.clip_bbox_float((float(warped[0]) - 0.5 * w, float(warped[1]) - 0.5 * h, w, h), w_img, h_img)


def dark_dipole_score(
    signed_diff: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    cur_bbox: tuple[int, int, int, int],
    sigma: float,
    radius: int,
    min_step_px: float,
    local_norm: bool = False,
    search_penalty: float = 0.0,
) -> float:
    pcx, pcy = base.bbox_center(prev_bbox)
    ccx, ccy = base.bbox_center(cur_bbox)
    step = math.hypot(ccx - pcx, ccy - pcy)
    if step < min_step_px:
        return 0.0
    old_mean = patch_mean(signed_diff, pcx, pcy, radius)
    new_mean = patch_mean(signed_diff, ccx, ccy, radius)
    if local_norm:
        old_bg, old_sigma = annulus_stats(signed_diff, pcx, pcy, radius + 1, max(radius + 3, 4 * radius))
        new_bg, new_sigma = annulus_stats(signed_diff, ccx, ccy, radius + 1, max(radius + 3, 4 * radius))
        old_positive = old_mean - old_bg
        new_negative = -(new_mean - new_bg)
        denom = math.sqrt(old_sigma * old_sigma + new_sigma * new_sigma)
    else:
        old_positive = old_mean
        new_negative = -new_mean
        denom = 2.0 * sigma
    z = (old_positive + new_negative) / max(1e-3, denom)
    if local_norm or search_penalty != 0.0:
        overlap = min(1.0, max(0.0, (step - min_step_px) / max(1.0, 1.5 * radius)))
        z = overlap * z - search_penalty
        return max(-4.0, min(4.0, z))
    return max(-3.0, min(3.0, z))


def alignment_gain_score(
    cand: base.Candidate,
    bg_pred_bbox: tuple[int, int, int, int],
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
) -> float:
    ccx, ccy = base.bbox_center(cand.bbox)
    bcx, bcy = base.bbox_center(bg_pred_bbox)
    radius = max(1, int(round(0.55 * max(cand.bbox[2], cand.bbox[3]))))
    cur_app = patch_mean(app_resp, ccx, ccy, radius)
    bg_app = patch_mean(app_resp, bcx, bcy, radius)
    cur_res = patch_mean(residual_blur, ccx, ccy, radius)
    bg_res = patch_mean(residual_blur, bcx, bcy, radius)
    app_scale = max(5.0, 0.5 * (abs(cur_app) + abs(bg_app)) + 3.0)
    res_scale = max(5.0, 0.5 * (abs(cur_res) + abs(bg_res)) + 3.0)
    app_gain = (cur_app - bg_app) / app_scale
    res_gain = (cur_res - bg_res) / res_scale
    return float(max(-3.0, min(3.0, 1.6 * app_gain + 0.9 * res_gain)))


def center_distance(
    a: tuple[int, int, int, int],
    b: tuple[float, float, float, float] | tuple[int, int, int, int],
) -> float:
    ax, ay = base.bbox_center(a)
    bx = float(b[0]) + 0.5 * float(b[2])
    by = float(b[1]) + 0.5 * float(b[3])
    return float(math.hypot(ax - bx, ay - by))


def parse_radii(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(max(1, int(round(float(part)))))
    return vals or [2, 3, 5]


def parse_int_offsets(text: str) -> list[int]:
    vals: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(round(float(part))))
    # The online tracker only has past frames. Ignore accidental future offsets.
    vals = sorted({v for v in vals if v < 0})
    return vals or [-5, -3, -2, -1]


def parse_xy_offsets(text: str) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        x_text, y_text = part.split(":")
        offsets.append((float(x_text), float(y_text)))
    return offsets or [(0.0, 0.0)]


def parse_float_list(text: str) -> list[float]:
    vals: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals or [0.0, 1.0]


def structure_maps(gray: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    gray_f = gray.astype(np.float32)
    k = max(3, 2 * radius + 1)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.blur(gx * gx, (k, k))
    jyy = cv2.blur(gy * gy, (k, k))
    jxy = cv2.blur(gx * gy, (k, k))
    trace = jxx + jyy
    disc = np.maximum(0.0, (jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy)
    lam1 = 0.5 * (trace + np.sqrt(disc))
    lam2 = 0.5 * (trace - np.sqrt(disc))
    anisotropy = (lam1 - lam2) / np.maximum(1e-3, lam1 + lam2)
    texture = np.sqrt(np.maximum(0.0, trace))
    tex_scale = max(5.0, float(np.percentile(texture, 85)))
    texture_z = np.clip(texture / tex_scale, 0.0, 2.0)
    line_context = anisotropy * texture_z
    return line_context.astype(np.float32), texture_z.astype(np.float32)


def likelihood_map_for_radius(
    cur_g: np.ndarray,
    residual_blur: np.ndarray,
    radius: int,
    args: argparse.Namespace,
    structure_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cur_g.astype(np.float32)
    residual = residual_blur.astype(np.float32)
    disk_k = max(3, 2 * radius + 1)
    outer_k = max(disk_k + 2, 2 * int(round(3.0 * radius)) + 1)
    if outer_k % 2 == 0:
        outer_k += 1

    disk_mean = cv2.blur(gray, (disk_k, disk_k))
    bg_mean = cv2.blur(gray, (outer_k, outer_k))
    local_dev = cv2.blur(np.abs(gray - bg_mean), (outer_k, outer_k))
    dark_z = (bg_mean - disk_mean) / (1.4826 * local_dev + 3.0)
    dark_z = np.maximum(0.0, dark_z)

    res_bg = cv2.blur(residual, (outer_k, outer_k))
    res_dev = cv2.blur(np.abs(residual - res_bg), (outer_k, outer_k))
    residual_z = (residual - res_bg) / (1.4826 * res_dev + 2.0)
    residual_z = np.maximum(0.0, residual_z)

    structure_radius = max(radius, 3)
    if structure_cache is not None and structure_radius in structure_cache:
        line_context, texture_z = structure_cache[structure_radius]
    else:
        line_context, texture_z = structure_maps(cur_g, radius=structure_radius)
        if structure_cache is not None:
            structure_cache[structure_radius] = (line_context, texture_z)
    score = (
        args.map_dark_weight * dark_z
        + args.map_residual_weight * residual_z
        - args.map_line_weight * line_context
        - args.map_texture_weight * texture_z
    )
    border = max(radius + 2, int(round(args.border_frac * min(cur_g.shape[:2]))))
    if border:
        score[:border, :] = -999.0
        score[-border:, :] = -999.0
        score[:, :border] = -999.0
        score[:, -border:] = -999.0
    return score.astype(np.float32), line_context, texture_z


def make_map_candidate(
    x: int,
    y: int,
    radius: int,
    map_score: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> base.Candidate | None:
    h_img, w_img = cur_g.shape[:2]
    side = max(3, int(round(2.4 * radius)))
    bx = int(round(x - 0.5 * side))
    by = int(round(y - 0.5 * side))
    bx = max(0, min(w_img - side, bx))
    by = max(0, min(h_img - side, by))
    bbox = (bx, by, side, side)
    mask = np.zeros_like(cur_g, dtype=np.uint8)
    mask[by : by + side, bx : bx + side] = 255
    cand = base.candidate_score("map", bbox, side * side, residual_blur, app_resp, mask, cur_g)
    cand.map_score = float(map_score)
    cand.score = 0.55 * cand.score + args.map_score_weight * float(map_score)
    if cand.score <= 0.1:
        return None
    return cand


def map_peak_candidates(
    cur_g: np.ndarray,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not args.map_peaks:
        return []
    peaks: list[tuple[float, int, int, int]] = []
    structure_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for radius in parse_radii(args.map_radii):
        score_map, line_context, texture_z = likelihood_map_for_radius(
            cur_g,
            residual_blur,
            radius,
            args,
            structure_cache,
        )
        nms = max(args.map_nms_px, 2 * radius + 1)
        if nms % 2 == 0:
            nms += 1
        dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
        if radius <= args.micro_map_max_radius:
            smooth_micro = (line_context <= args.micro_map_max_line) & (texture_z <= args.micro_map_max_texture)
            edge_micro = line_context <= args.micro_map_edge_max_line
            weak_smooth = (score_map >= args.micro_map_score_floor) & smooth_micro
            strong_edge = (score_map >= args.micro_map_edge_score_floor) & edge_micro
            mask = (weak_smooth | strong_edge) & (score_map >= dilated - 1e-6)
        else:
            mask = (score_map >= args.map_score_floor) & (score_map >= dilated - 1e-6)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        vals = score_map[ys, xs]
        order = np.argsort(vals)[::-1][: args.map_top_k]
        for idx in order:
            peaks.append((float(vals[idx]), int(xs[idx]), int(ys[idx]), radius))

    cands: list[base.Candidate] = []
    occupied: list[tuple[int, int, int, int]] = []
    for score, x, y, radius in sorted(peaks, reverse=True):
        if len(cands) >= args.map_top_k:
            break
        cand = make_map_candidate(x, y, radius, score, residual_blur, app_resp, cur_g, args)
        if cand is None:
            continue
        duplicate = False
        ccx, ccy = base.bbox_center(cand.bbox)
        for b in occupied:
            bcx, bcy = base.bbox_center(b)
            if math.hypot(ccx - bcx, ccy - bcy) <= max(3.0, 1.5 * radius):
                duplicate = True
                break
        if duplicate:
            continue
        occupied.append(cand.bbox)
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def native_micro_candidates(
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not args.native_micro_peaks or cur_full is None or abs(downscale - 1.0) < 1e-6:
        return []
    full_g = base.ensure_gray(cur_full)
    h_full, w_full = full_g.shape[:2]
    h_img, w_img = cur_g.shape[:2]
    gray = cv2.GaussianBlur(full_g, (3, 3), 0).astype(np.float32)
    peaks: list[tuple[float, int, int, int]] = []
    structure_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for radius in parse_radii(args.native_micro_radii):
        disk_k = max(3, 2 * radius + 1)
        outer_k = max(disk_k + 2, 2 * int(round(3.0 * radius)) + 1)
        if outer_k % 2 == 0:
            outer_k += 1
        disk_mean = cv2.blur(gray, (disk_k, disk_k))
        bg_mean = cv2.blur(gray, (outer_k, outer_k))
        local_dev = cv2.blur(np.abs(gray - bg_mean), (outer_k, outer_k))
        dark_z = np.maximum(0.0, (bg_mean - disk_mean) / (1.4826 * local_dev + 3.0))
        structure_radius = max(3, radius)
        if structure_radius not in structure_cache:
            structure_cache[structure_radius] = structure_maps(full_g, structure_radius)
        line_context, texture_z = structure_cache[structure_radius]
        score_map = (
            dark_z
            - args.native_micro_line_weight * line_context
            - args.native_micro_texture_weight * texture_z
        ).astype(np.float32)
        border = max(radius + 2, int(round(args.border_frac * min(w_full, h_full))))
        if border:
            score_map[:border, :] = -999.0
            score_map[-border:, :] = -999.0
            score_map[:, :border] = -999.0
            score_map[:, -border:] = -999.0
        nms = max(args.native_micro_nms_px, 2 * radius + 1)
        if nms % 2 == 0:
            nms += 1
        dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
        mask = (score_map >= args.native_micro_score_floor) & (score_map >= dilated - 1e-6)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        vals = score_map[ys, xs]
        order = np.argsort(vals)[::-1][: args.native_micro_top_k]
        for idx in order:
            peaks.append((float(vals[idx]), int(xs[idx]), int(ys[idx]), radius))

    cands: list[base.Candidate] = []
    occupied: list[tuple[int, int, int, int]] = []
    for score, x_full, y_full, radius in sorted(peaks, reverse=True):
        if len(cands) >= args.native_micro_top_k:
            break
        cx = x_full * downscale
        cy = y_full * downscale
        side = max(3, int(round(2.4 * radius * downscale)))
        bx = int(round(cx - 0.5 * side))
        by = int(round(cy - 0.5 * side))
        bx = max(0, min(w_img - side, bx))
        by = max(0, min(h_img - side, by))
        bbox = (bx, by, side, side)
        duplicate = False
        ccx, ccy = base.bbox_center(bbox)
        for kept in occupied:
            kcx, kcy = base.bbox_center(kept)
            if math.hypot(ccx - kcx, ccy - kcy) <= max(3.0, 1.5 * side):
                duplicate = True
                break
        if duplicate:
            continue
        mask = np.zeros_like(cur_g, dtype=np.uint8)
        mask[by : by + side, bx : bx + side] = 255
        cand = base.candidate_score("native_map", bbox, side * side, residual_blur, app_resp, mask, cur_g)
        cand.map_score = float(score)
        cand.score = 0.45 * cand.score + args.native_micro_score_weight * float(score)
        if cand.score <= 0.1:
            continue
        occupied.append(bbox)
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def large_dark_candidates(
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    """Full-frame dark-silhouette proposals for close, clearly visible drones.

    The existing micro and temporal-stack proposal paths are tuned for tiny
    3-10 px events and motion residuals. In the e271 gap, the drone is larger
    and often obvious as a dark local-contrast object, while residual ranking
    can suppress it. This source deliberately adds a small number of large
    dark-object proposals so the tube/ranker can choose them.
    """
    if not args.large_dark_peaks or cur_full is None or abs(downscale - 1.0) < 1e-6:
        return []
    full_g = base.ensure_gray(cur_full)
    h_full, w_full = full_g.shape[:2]
    h_img, w_img = cur_g.shape[:2]
    gray = cv2.GaussianBlur(full_g, (3, 3), 0).astype(np.float32)
    bg = cv2.GaussianBlur(full_g.astype(np.float32), (0, 0), args.large_dark_bg_sigma)
    score_map = (bg - gray).astype(np.float32)
    border = max(8, int(round(args.border_frac * min(w_full, h_full))))
    if border:
        score_map[:border, :] = -999.0
        score_map[-border:, :] = -999.0
        score_map[:, :border] = -999.0
        score_map[:, -border:] = -999.0

    nms = max(3, int(round(args.large_dark_nms_px)))
    if nms % 2 == 0:
        nms += 1
    dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
    mask = (score_map >= args.large_dark_score_floor) & (score_map >= dilated - 1e-6)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    vals = score_map[ys, xs]
    order = np.argsort(vals)[::-1][: args.large_dark_top_k]

    cands: list[base.Candidate] = []
    occupied: list[tuple[int, int, int, int]] = []
    side = max(3, int(round(args.large_dark_box_full * downscale)))
    for idx in order:
        score = float(vals[idx])
        x_full = int(xs[idx])
        y_full = int(ys[idx])
        cx = x_full * downscale
        cy = y_full * downscale
        bx = max(0, min(w_img - side, int(round(cx - 0.5 * side))))
        by = max(0, min(h_img - side, int(round(cy - 0.5 * side))))
        bbox = (bx, by, side, side)
        ccx, ccy = base.bbox_center(bbox)
        if any(math.hypot(ccx - kcx, ccy - kcy) <= max(3.0, side) for kcx, kcy in occupied):
            continue
        mask_img = np.zeros_like(cur_g, dtype=np.uint8)
        mask_img[by : by + side, bx : bx + side] = 255
        cand = base.candidate_score("large_dark", bbox, side * side, residual_blur, app_resp, mask_img, cur_g)
        cand.map_score = score
        cand.score = 0.45 * cand.score + args.large_dark_score_weight * score
        if cand.score <= 0.1:
            continue
        occupied.append((ccx, ccy))
        cands.append(cand)
        if len(cands) >= args.large_dark_top_k:
            break
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def srps_source_candidates_from_base(
    frame_no: int,
    cands: list[base.Candidate],
    args: argparse.Namespace,
) -> list[SRPSSourceCandidate]:
    allowed = _parse_source_set(getattr(args, "srps_source_candidates", ""))
    ranked = sorted(cands, key=lambda c: float(c.score), reverse=True)
    out: list[SRPSSourceCandidate] = []
    for rank, cand in enumerate(ranked, start=1):
        if rank > max(1, int(getattr(args, "srps_source_top_k", 80))):
            break
        if allowed and cand.source not in allowed:
            continue
        cx, cy = base.bbox_center(cand.bbox)
        out.append(
            SRPSSourceCandidate(
                frame=frame_no,
                cx=float(cx),
                cy=float(cy),
                bbox=tuple(float(v) for v in cand.bbox),
                source=cand.source,
                rank=rank,
                score=float(cand.score),
                track_id=str(getattr(cand, "recenter_seed_track_id", "") or getattr(cand, "target_local_anchor_id", "")),
                coord_space="detector",
                payload=cand,
            )
        )
    return out


def srps_residual_candidates_from_map(
    frame_no: int,
    residual_map: np.ndarray,
    args: argparse.Namespace,
) -> list[SRPSResidualCandidate]:
    score_map = residual_map.astype(np.float32)
    if score_map.size == 0:
        return []
    border = int(round(args.border_frac * min(score_map.shape[:2])))
    if border:
        score_map = score_map.copy()
        score_map[:border, :] = -999.0
        score_map[-border:, :] = -999.0
        score_map[:, :border] = -999.0
        score_map[:, -border:] = -999.0
    nms = max(3, int(round(getattr(args, "srps_residual_nms_px", 4.0))))
    if nms % 2 == 0:
        nms += 1
    dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
    floor = float(getattr(args, "srps_residual_score_floor", 0.0))
    mask = (score_map >= floor) & (score_map >= dilated - 1e-6)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    vals = score_map[ys, xs]
    order = np.argsort(vals)[::-1][: max(1, int(getattr(args, "srps_residual_top", 200)))]
    side = max(1.0, float(getattr(args, "srps_residual_box_det_px", 8.0)))
    out: list[SRPSResidualCandidate] = []
    for local_rank, idx in enumerate(order, start=1):
        x = float(xs[idx])
        y = float(ys[idx])
        out.append(
            SRPSResidualCandidate(
                frame=frame_no,
                cx=x,
                cy=y,
                rank=local_rank,
                score=float(vals[idx]),
                bbox=(x - 0.5 * side, y - 0.5 * side, side, side),
                coord_space="detector",
            )
        )
    return out


def srps_residual_candidates_from_full_map(
    frame_no: int,
    residual_map_full: np.ndarray,
    downscale: float,
    args: argparse.Namespace,
) -> list[SRPSResidualCandidate]:
    """Extract SRPS residual peaks in full-res coordinates, then downscale.

    The old SRPS runtime path resized the temporal residual map first and then
    searched for detector-space maxima. On terrain-backed e271 frames that loses
    centering evidence the offline proposal recovery path keeps. This function
    mirrors the teacher shape more closely while remaining causal.
    """

    if residual_map_full.size == 0 or downscale <= 0:
        return []
    score_map = residual_map_full.astype(np.float32)
    border = int(round(args.border_frac * min(score_map.shape[:2]) / max(1e-6, downscale)))
    if border:
        score_map = score_map.copy()
        score_map[:border, :] = -999.0
        score_map[-border:, :] = -999.0
        score_map[:, :border] = -999.0
        score_map[:, -border:] = -999.0

    peaks: list[tuple[float, float, float, int]] = []
    for radius in parse_radii(getattr(args, "temporal_stack_radii", "2,3,4,5,7")):
        peaks.extend(local_maxima_peaks(score_map, radius, max(1, int(getattr(args, "srps_residual_top", 200)))))

    nms_full = max(2.5, float(getattr(args, "srps_residual_nms_px", 4.0)) / max(1e-6, float(downscale)))
    peaks = dedupe_full_peaks(peaks, nms_full, max(1, int(getattr(args, "srps_residual_top", 200))))
    floor = float(getattr(args, "srps_residual_score_floor", 0.0))
    side = max(1.0, float(getattr(args, "srps_residual_box_det_px", 8.0)))
    out: list[SRPSResidualCandidate] = []
    for local_rank, (peak_score, x_full, y_full, _radius) in enumerate(peaks, start=1):
        if peak_score < floor:
            continue
        cx = float(x_full) * float(downscale)
        cy = float(y_full) * float(downscale)
        out.append(
            SRPSResidualCandidate(
                frame=frame_no,
                cx=cx,
                cy=cy,
                rank=local_rank,
                score=float(peak_score),
                bbox=(cx - 0.5 * side, cy - 0.5 * side, side, side),
                coord_space="detector",
                payload={"source": "temporal_dark_fullres", "x_full": float(x_full), "y_full": float(y_full)},
            )
        )
    return out


def load_srps_teacher_residuals(path: str, coord_scale: float) -> dict[int, list[SRPSResidualCandidate]]:
    if not path:
        return {}
    out: dict[int, list[SRPSResidualCandidate]] = {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--srps_teacher_residual_proposals_csv not found: {path}")
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                frame = int(round(float(row.get("frame", ""))))
                rank = int(round(float(row.get("rank", 999999))))
                cx = float(row.get("cx", 0.0)) * coord_scale
                cy = float(row.get("cy", 0.0)) * coord_scale
                score = float(row.get("score", 0.0))
            except Exception:
                continue
            side = max(1.0, 8.0 * coord_scale)
            out.setdefault(frame, []).append(
                SRPSResidualCandidate(
                    frame=frame,
                    cx=cx,
                    cy=cy,
                    rank=rank,
                    score=score,
                    bbox=(cx - 0.5 * side, cy - 0.5 * side, side, side),
                    coord_space="detector",
                    payload=row,
                )
            )
    for rows in out.values():
        rows.sort(key=lambda r: int(r.rank))
    return out


def load_srps_teacher_sources(
    path: str,
    coord_scale: float,
    allowed_sources: set[str] | None = None,
) -> dict[int, list[SRPSSourceCandidate]]:
    if not path:
        return {}
    allowed_sources = allowed_sources or set()
    out: dict[int, list[SRPSSourceCandidate]] = {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--srps_teacher_candidate_csv not found: {path}")
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                frame = int(round(float(row.get("frame", ""))))
                rank = int(round(float(row.get("rank", 999999))))
                x = float(row.get("x", 0.0)) * coord_scale
                y = float(row.get("y", 0.0)) * coord_scale
                w = max(1.0, float(row.get("w", 1.0)) * coord_scale)
                h = max(1.0, float(row.get("h", 1.0)) * coord_scale)
                score = float(row.get("score", row.get("verified_score", 0.0)) or 0.0)
            except Exception:
                continue
            source = str(row.get("cand_source", row.get("source", "")) or "").strip()
            if not source:
                continue
            if allowed_sources and source not in allowed_sources:
                continue
            out.setdefault(frame, []).append(
                SRPSSourceCandidate(
                    frame=frame,
                    cx=x + 0.5 * w,
                    cy=y + 0.5 * h,
                    bbox=(x, y, w, h),
                    source=source,
                    rank=rank,
                    score=score,
                    track_id=str(row.get("track_id", "") or ""),
                    coord_space="detector",
                    payload=row,
                )
            )
    for rows in out.values():
        rows.sort(key=lambda r: (int(r.rank), -float(r.score)))
    return out


def srps_should_run(
    args: argparse.Namespace,
    frame_decision: FrameRouterDecision,
    surface_extras_allowed: bool,
) -> bool:
    if not getattr(args, "stabilized_residual_path_source", False):
        return False
    mode = getattr(args, "srps_recovery_mode", "terrain_no_sky_only")
    if mode == "all":
        return True
    if mode == "surface_or_low_confidence":
        return surface_extras_allowed or frame_decision.mode in {"surface", "boundary", "unknown"}
    return frame_decision.mode == "surface"


def srps_base_state_from_tbd(tbd: BeamTBD, args: argparse.Namespace) -> SRPSBaseState:
    best = tbd.best()
    if best is None:
        return SRPSBaseState(state="A", stable_t=False, low_confidence=True)
    try:
        score = float(tbd.verified_score(best))
    except Exception:
        score = float(best.score())
    stable = (
        best.misses == 0
        and best.hit_count() >= max(1, int(getattr(args, "min_path_hits", 3)))
        and score >= float(getattr(args, "selected_score", 6.0))
    )
    return SRPSBaseState(
        state="T" if stable else "P",
        stable_t=stable,
        low_confidence=not stable,
        router="unknown",
    )


def srps_to_base_candidate(
    srps: SRPSCandidate,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> base.Candidate:
    h_img, w_img = cur_g.shape[:2]
    bbox = base.clip_bbox_float(srps.bbox, w_img, h_img)
    x, y, w, h = bbox
    mask = np.zeros_like(cur_g, dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    cand = base.candidate_score("stabilized_residual_path", bbox, max(1, w * h), residual_blur, app_resp, mask, cur_g)
    instant = max(-5.0, min(15.0, float(srps.score)))
    path_bonus = 0.15 * max(-10.0, min(40.0, float(srps.path_confidence)))
    confirm_bonus = 2.0 if srps.srps_state == "confirmed_path" else 0.0
    miss_penalty = 0.8 * max(0, int(srps.path_miss_count))
    cand.score = 0.20 * float(cand.score) + float(getattr(args, "srps_score_weight", 1.0)) * (
        instant + path_bonus + confirm_bonus - miss_penalty
    )
    if getattr(args, "srps_verified_candidate_priority", False) and srps.verified_path:
        cand.score = max(float(cand.score), float(getattr(args, "selected_score", 6.0)) + 2.0)
    cand.map_score = float(srps.snap_score)
    cand.srps_state = srps.srps_state
    cand.srps_path_confidence = float(srps.path_confidence)
    cand.srps_seed_type = srps.seed_type
    cand.srps_seed_source = srps.seed_source
    cand.srps_seed_rank = int(srps.seed_rank)
    cand.srps_seed_score = float(srps.seed_score)
    cand.srps_seed_track_id = srps.seed_track_id
    cand.srps_snap_rank = int(srps.snap_rank)
    cand.srps_snap_score = float(srps.snap_score)
    cand.srps_snap_distance = float(srps.snap_distance)
    cand.srps_pred_distance = float(srps.pred_distance)
    cand.srps_path_miss_count = int(srps.path_miss_count)
    cand.srps_hits = int(srps.hits)
    cand.srps_confirm_frame = -1 if srps.confirm_frame is None else int(srps.confirm_frame)
    cand.srps_coord_space = srps.coord_space
    cand.srps_verified_seed = int(srps.verified_seed)
    cand.srps_verified_path = int(srps.verified_path)
    cand.srps_verification_score = float(srps.verification_score)
    cand.srps_support_sources = srps.support_sources
    cand.srps_residual_density = int(srps.residual_density)
    return cand


def compact_dark_map_native(gray: np.ndarray, radius: int, texture_weight: float = 0.025) -> np.ndarray:
    img = gray.astype(np.float32)
    r = max(1, int(radius))
    inner_k = 2 * r + 1
    outer_k = 6 * r + 3
    inner_n = float(inner_k * inner_k)
    outer_n = float(outer_k * outer_k)
    ring_n = max(1.0, outer_n - inner_n)

    inner_sum = cv2.boxFilter(img, cv2.CV_32F, (inner_k, inner_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    outer_sum = cv2.boxFilter(img, cv2.CV_32F, (outer_k, outer_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    inner_mean = inner_sum / inner_n
    ring_mean = (outer_sum - inner_sum) / ring_n

    outer_sq = cv2.boxFilter(img * img, cv2.CV_32F, (outer_k, outer_k), normalize=False, borderType=cv2.BORDER_REFLECT101)
    outer_mean = outer_sum / outer_n
    outer_var = np.maximum(0.0, outer_sq / outer_n - outer_mean * outer_mean)
    outer_std = np.sqrt(outer_var + 9.0)
    dark_z = (ring_mean - inner_mean) / outer_std

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    texture = cv2.boxFilter(grad, cv2.CV_32F, (outer_k, outer_k), normalize=True, borderType=cv2.BORDER_REFLECT101)

    score = dark_z - texture_weight * texture
    border = max(outer_k, 10)
    score[:border, :] = -999.0
    score[-border:, :] = -999.0
    score[:, :border] = -999.0
    score[:, -border:] = -999.0
    return score.astype(np.float32)


def dog_dark_map_native(gray: np.ndarray, radius: int, texture_weight: float = 0.025) -> np.ndarray:
    img = gray.astype(np.float32)
    r = max(1, int(radius))
    inner_sigma = max(0.6, 0.55 * r)
    outer_sigma = max(inner_sigma + 0.5, 1.65 * r)
    inner = cv2.GaussianBlur(img, (0, 0), inner_sigma, borderType=cv2.BORDER_REFLECT101)
    outer = cv2.GaussianBlur(img, (0, 0), outer_sigma, borderType=cv2.BORDER_REFLECT101)
    response = outer - inner
    norm_k = 6 * r + 3
    mean = cv2.boxFilter(response, cv2.CV_32F, (norm_k, norm_k), normalize=True, borderType=cv2.BORDER_REFLECT101)
    sq = cv2.boxFilter(response * response, cv2.CV_32F, (norm_k, norm_k), normalize=True, borderType=cv2.BORDER_REFLECT101)
    std = np.sqrt(np.maximum(4.0, sq - mean * mean))
    score = (response - mean) / std
    if texture_weight > 0.0:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.boxFilter(cv2.magnitude(gx, gy), cv2.CV_32F, (norm_k, norm_k), normalize=True, borderType=cv2.BORDER_REFLECT101)
        score = score - texture_weight * texture
    border = max(norm_k, 10)
    score[:border, :] = -999.0
    score[-border:, :] = -999.0
    score[:, :border] = -999.0
    score[:, -border:] = -999.0
    return score.astype(np.float32)


def blackhat_dark_map_native(gray: np.ndarray, radius: int, texture_weight: float = 0.025) -> np.ndarray:
    img_u8 = gray.astype(np.uint8)
    r = max(1, int(radius))
    k = max(3, 2 * r + 1)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    response = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    norm_k = 6 * r + 3
    mean = cv2.boxFilter(response, cv2.CV_32F, (norm_k, norm_k), normalize=True, borderType=cv2.BORDER_REFLECT101)
    sq = cv2.boxFilter(response * response, cv2.CV_32F, (norm_k, norm_k), normalize=True, borderType=cv2.BORDER_REFLECT101)
    std = np.sqrt(np.maximum(4.0, sq - mean * mean))
    score = (response - mean) / std
    if texture_weight > 0.0:
        img = gray.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.boxFilter(cv2.magnitude(gx, gy), cv2.CV_32F, (norm_k, norm_k), normalize=True, borderType=cv2.BORDER_REFLECT101)
        score = score - texture_weight * texture
    border = max(norm_k, 10)
    score[:border, :] = -999.0
    score[-border:, :] = -999.0
    score[:, :border] = -999.0
    score[:, -border:] = -999.0
    return score.astype(np.float32)


def parse_response_maps(raw: str) -> list[str]:
    allowed = {"compact_dark", "dog", "blackhat"}
    out: list[str] = []
    for part in str(raw or "compact_dark").split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in allowed:
            continue
        if name not in out:
            out.append(name)
    return out or ["compact_dark"]


def build_recenter_score_maps(
    gray: np.ndarray,
    radii: list[int],
    response_maps: list[str],
    texture_weight: float,
) -> dict[tuple[str, int], np.ndarray]:
    maps: dict[tuple[str, int], np.ndarray] = {}
    for r in radii:
        for name in response_maps:
            if name == "dog":
                maps[(name, r)] = dog_dark_map_native(gray, r, texture_weight)
            elif name == "blackhat":
                maps[(name, r)] = blackhat_dark_map_native(gray, r, texture_weight)
            else:
                maps[(name, r)] = compact_dark_map_native(gray, r, texture_weight)
    return maps


def scale_h_to_full(h_down: np.ndarray, downscale: float) -> np.ndarray:
    if abs(downscale - 1.0) < 1e-6:
        return h_down.astype(np.float32)
    s = float(downscale)
    scale = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    inv_scale = np.array([[1.0 / s, 0.0, 0.0], [0.0, 1.0 / s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (inv_scale @ h_down.astype(np.float32) @ scale).astype(np.float32)


def estimate_direct_h_to_current_full(
    prev_full_gray: np.ndarray,
    cur_full_gray: np.ndarray,
    downscale: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, float]:
    if abs(downscale - 1.0) < 1e-6:
        prev_match = prev_full_gray
        cur_match = cur_full_gray
    else:
        prev_match = cv2.resize(prev_full_gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        cur_match = cv2.resize(cur_full_gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    g0, g1 = base.lk_tracks(prev_match, cur_match, None, args)
    if g0 is None or g1 is None:
        return None, 0.0
    chosen = base.choose_model(prev_match, cur_match, g0, g1, args)
    if chosen is None:
        return None, 0.0
    return scale_h_to_full(chosen["h"], downscale), float(chosen["inlier_ratio"])


def update_temporal_stack_history(
    history: list[TemporalStackFrame],
    prev_full_gray: np.ndarray,
    h_prev_to_cur_down: np.ndarray,
    downscale: float,
    max_age: int,
) -> list[TemporalStackFrame]:
    h_full = scale_h_to_full(h_prev_to_cur_down, downscale)
    updated = [
        TemporalStackFrame(entry.gray_full, (h_full @ entry.h_to_current_full).astype(np.float32))
        for entry in history
    ]
    updated.append(TemporalStackFrame(prev_full_gray, h_full))
    return updated[-max_age:]


def temporal_stack_residual_map(
    cur_full_gray: np.ndarray,
    history: list[TemporalStackFrame],
    offsets: list[int],
    min_frames: int,
    downscale: float,
    args: argparse.Namespace,
) -> np.ndarray | None:
    h_full, w_full = cur_full_gray.shape[:2]
    warped: list[np.ndarray] = []
    for off in offsets:
        age = abs(off)
        if age <= 0 or age > len(history):
            continue
        entry = history[-age]
        if args.temporal_stack_direct_warp:
            h_to_current, _inlier = estimate_direct_h_to_current_full(
                entry.gray_full,
                cur_full_gray,
                downscale,
                args,
            )
            if h_to_current is None:
                continue
        else:
            h_to_current = entry.h_to_current_full
        warped.append(
            cv2.warpPerspective(
                entry.gray_full,
                h_to_current,
                (w_full, h_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        )
    if len(warped) < min_frames:
        return None
    med = np.median(np.stack(warped, axis=0), axis=0).astype(np.float32)
    residual_dark = med - cur_full_gray.astype(np.float32)
    local_mean = cv2.GaussianBlur(residual_dark, (0, 0), 5.0)
    local_sq = cv2.GaussianBlur(residual_dark * residual_dark, (0, 0), 5.0)
    local_std = np.sqrt(np.maximum(4.0, local_sq - local_mean * local_mean))
    return ((residual_dark - local_mean) / local_std).astype(np.float32)


def local_maxima_peaks(score_map: np.ndarray, radius: int, top_k: int) -> list[tuple[float, float, float, int]]:
    nms = max(3, int(round(2 * radius + 1)))
    if nms % 2 == 0:
        nms += 1
    dilated = cv2.dilate(score_map, np.ones((nms, nms), dtype=np.uint8))
    mask = (score_map >= dilated - 1e-6) & np.isfinite(score_map) & (score_map > -100.0)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    vals = score_map[ys, xs]
    order = np.argsort(vals)[::-1][:top_k]
    return [(float(vals[i]), float(xs[i]), float(ys[i]), radius) for i in order]


def dedupe_full_peaks(
    peaks: list[tuple[float, float, float, int]],
    nms_px: float,
    max_n: int,
) -> list[tuple[float, float, float, int]]:
    out: list[tuple[float, float, float, int]] = []
    for score, x, y, radius in sorted(peaks, key=lambda p: p[0], reverse=True):
        if len(out) >= max_n:
            break
        if any(math.hypot(x - ox, y - oy) <= nms_px for _os, ox, oy, _or in out):
            continue
        out.append((score, x, y, radius))
    return out


def local_native_peaks_near(
    score_maps: dict[int, np.ndarray],
    center_full: tuple[float, float],
    search_radius_full_px: float,
    max_peaks: int,
) -> list[tuple[float, float, float, int]]:
    """Find compact-response peaks near a predicted full-resolution center."""

    cx, cy = center_full
    peaks: list[tuple[float, float, float, int]] = []
    for radius, score_map in score_maps.items():
        h, w = score_map.shape[:2]
        x0 = max(0, int(math.floor(cx - search_radius_full_px)))
        y0 = max(0, int(math.floor(cy - search_radius_full_px)))
        x1 = min(w, int(math.ceil(cx + search_radius_full_px + 1)))
        y1 = min(h, int(math.ceil(cy + search_radius_full_px + 1)))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = score_map[y0:y1, x0:x1]
        nms = max(3, int(round(2 * radius + 1)))
        if nms % 2 == 0:
            nms += 1
        dilated = cv2.dilate(crop, np.ones((nms, nms), dtype=np.uint8))
        mask = (crop >= dilated - 1e-6) & np.isfinite(crop) & (crop > -100.0)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        vals = crop[ys, xs]
        order = np.argsort(vals)[::-1][: max(1, max_peaks)]
        for idx in order:
            peaks.append((float(vals[idx]), float(x0 + xs[idx]), float(y0 + ys[idx]), radius))
    return dedupe_full_peaks(peaks, nms_px=2.0, max_n=max(1, max_peaks))


def refine_peak_subpixel(score_map: np.ndarray, x: float, y: float) -> tuple[float, float, str, float]:
    xi = int(round(x))
    yi = int(round(y))
    h, w = score_map.shape[:2]
    if xi <= 0 or yi <= 0 or xi >= w - 1 or yi >= h - 1:
        return x, y, "integer", 0.0
    crop = score_map[yi - 1 : yi + 2, xi - 1 : xi + 2].astype(np.float32)
    if not np.all(np.isfinite(crop)):
        return x, y, "integer", 0.0
    weights = crop - float(np.min(crop))
    total = float(np.sum(weights))
    if total <= 1e-6:
        return x, y, "integer", 0.0
    yy, xx = np.mgrid[-1:2, -1:2]
    dx = float(np.sum(xx * weights) / total)
    dy = float(np.sum(yy * weights) / total)
    if abs(dx) > 0.75 or abs(dy) > 0.75:
        return x, y, "integer", total
    return x + dx, y + dy, "centroid3x3", total


def local_native_peaks_near_with_meta(
    score_maps: dict[tuple[str, int], np.ndarray],
    center_full: tuple[float, float],
    search_radius_full_px: float,
    max_peaks: int,
) -> list[dict[str, object]]:
    cx, cy = center_full
    peaks: list[dict[str, object]] = []
    for (response_name, radius), score_map in score_maps.items():
        h, w = score_map.shape[:2]
        x0 = max(0, int(math.floor(cx - search_radius_full_px)))
        y0 = max(0, int(math.floor(cy - search_radius_full_px)))
        x1 = min(w, int(math.ceil(cx + search_radius_full_px + 1)))
        y1 = min(h, int(math.ceil(cy + search_radius_full_px + 1)))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = score_map[y0:y1, x0:x1]
        nms = max(3, int(round(2 * radius + 1)))
        if nms % 2 == 0:
            nms += 1
        dilated = cv2.dilate(crop, np.ones((nms, nms), dtype=np.uint8))
        mask = (crop >= dilated - 1e-6) & np.isfinite(crop) & (crop > -100.0)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        vals = crop[ys, xs]
        order = np.argsort(vals)[::-1][: max(1, max_peaks)]
        sorted_vals = [float(vals[i]) for i in order]
        second = sorted_vals[1] if len(sorted_vals) > 1 else None
        for local_rank, idx in enumerate(order, start=1):
            x_raw = float(x0 + xs[idx])
            y_raw = float(y0 + ys[idx])
            x_ref, y_ref, method, condition = refine_peak_subpixel(score_map, x_raw, y_raw)
            score = float(vals[idx])
            margin = score - second if second is not None and local_rank == 1 else 0.0
            peaks.append(
                {
                    "score": score,
                    "x": x_ref,
                    "y": y_ref,
                    "radius": int(radius),
                    "response_map": response_name,
                    "response_family": response_name,
                    "local_rank": local_rank,
                    "second_peak_margin": max(0.0, float(margin)),
                    "subpixel_dx": x_ref - x_raw,
                    "subpixel_dy": y_ref - y_raw,
                    "subpixel_method": method,
                    "subpixel_condition": condition,
                }
            )
    deduped: list[dict[str, object]] = []
    for peak in sorted(peaks, key=lambda p: float(p["score"]), reverse=True):
        if len(deduped) >= max(1, max_peaks):
            break
        if any(math.hypot(float(peak["x"]) - float(prev["x"]), float(peak["y"]) - float(prev["y"])) <= 2.0 for prev in deduped):
            continue
        deduped.append(peak)
    return deduped


def inverse_warp_point(
    x: float,
    y: float,
    inv_h_to_current: np.ndarray,
) -> tuple[float, float] | None:
    pt = np.array([[[x, y]]], dtype=np.float32)
    try:
        warped = cv2.perspectiveTransform(pt, inv_h_to_current).reshape(2)
    except cv2.error:
        return None
    if not np.isfinite(warped).all():
        return None
    return float(warped[0]), float(warped[1])


def temporal_stack_local_score(
    cur_full_gray: np.ndarray,
    history_refs: list[tuple[TemporalStackFrame, np.ndarray]],
    x_full: float,
    y_full: float,
    radius: int,
    min_frames: int,
) -> float | None:
    vals: list[float] = []
    h_full, w_full = cur_full_gray.shape[:2]
    if not (0 <= x_full < w_full and 0 <= y_full < h_full):
        return None
    for entry, inv_h in history_refs:
        prev_pt = inverse_warp_point(x_full, y_full, inv_h)
        if prev_pt is None:
            continue
        px, py = prev_pt
        if not (0 <= px < entry.gray_full.shape[1] and 0 <= py < entry.gray_full.shape[0]):
            continue
        vals.append(patch_mean(entry.gray_full, px, py, radius))
    if len(vals) < min_frames:
        return None
    cur = patch_mean(cur_full_gray, x_full, y_full, radius)
    residual_dark = float(np.median(vals) - cur)
    spread = max(4.0, float(np.std(vals)) + 3.0)
    return residual_dark / spread


def temporal_stack_candidate_local_candidates(
    cur_full: np.ndarray | None,
    downscale: float,
    history: list[TemporalStackFrame],
    seed_cands: list[base.Candidate],
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not args.temporal_stack_peaks or cur_full is None or downscale <= 0:
        return []
    full_g = base.ensure_gray(cur_full)
    inv_downscale = 1.0 / downscale
    offsets = parse_int_offsets(args.temporal_stack_offsets)
    history_refs: list[tuple[TemporalStackFrame, np.ndarray]] = []
    for off in offsets:
        age = abs(off)
        if age <= 0 or age > len(history):
            continue
        entry = history[-age]
        try:
            inv_h = np.linalg.inv(entry.h_to_current_full)
        except np.linalg.LinAlgError:
            continue
        history_refs.append((entry, inv_h.astype(np.float32)))
    if len(history_refs) < args.temporal_stack_min_frames:
        return []
    halo_offsets = parse_xy_offsets(args.temporal_stack_halo_offsets)[: max(1, args.temporal_stack_local_halo_limit)]
    radii = parse_radii(args.temporal_stack_radii)[:3]
    seed_pool = [
        cand
        for cand in seed_cands
        if getattr(cand, "router_state", "unknown")
        in {"surface_backed", "boundary_mixed", "sky_target_near_surface", "unknown"}
    ]
    if not seed_pool:
        seed_pool = seed_cands
    seed_pool = sorted(seed_pool, key=lambda cand: candidate_obs(cand, args), reverse=True)[
        : max(1, args.temporal_stack_seed_top_k)
    ]

    h_img, w_img = cur_g.shape[:2]
    proposals: list[tuple[float, float, float, int]] = []
    for seed_rank, seed in enumerate(seed_pool, start=1):
        scx, scy = base.bbox_center(seed.bbox)
        x_full = scx * inv_downscale
        y_full = scy * inv_downscale
        for dx, dy in halo_offsets:
            hx = x_full + dx
            hy = y_full + dy
            for radius in radii:
                stack_score = temporal_stack_local_score(
                    full_g,
                    history_refs,
                    hx,
                    hy,
                    radius,
                    args.temporal_stack_min_frames,
                )
                if stack_score is None:
                    continue
                native_score = native_dark_score_for_bbox(
                    full_g,
                    (
                        int(round((hx * downscale) - radius * downscale)),
                        int(round((hy * downscale) - radius * downscale)),
                        max(3, int(round((2 * radius + 1) * downscale))),
                        max(3, int(round((2 * radius + 1) * downscale))),
                    ),
                    inv_downscale,
                )
                score = float(stack_score + args.temporal_stack_native_weight * native_score)
                proposals.append((score, hx, hy, radius))

    peaks = dedupe_full_peaks(
        proposals,
        max(2.5, 0.45 * args.temporal_stack_nms_px),
        args.temporal_stack_top_k,
    )
    cands: list[base.Candidate] = []
    for score, x_full, y_full, radius in peaks:
        if score <= 0.1:
            continue
        side_full = max(3, int(round(2 * radius + 1)))
        side = max(3, int(round(side_full * downscale)))
        cx = x_full * downscale
        cy = y_full * downscale
        bx = max(0, min(w_img - side, int(round(cx - 0.5 * side))))
        by = max(0, min(h_img - side, int(round(cy - 0.5 * side))))
        bbox = (bx, by, side, side)
        mask = np.zeros_like(cur_g, dtype=np.uint8)
        mask[by : by + side, bx : bx + side] = 255
        cand = base.candidate_score("temporal_stack", bbox, side * side, residual_blur, app_resp, mask, cur_g)
        cand.map_score = float(score)
        cand.score = 0.35 * cand.score + args.temporal_stack_score_weight * float(score)
        if cand.score <= 0.1:
            continue
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def temporal_stack_candidates(
    cur_full: np.ndarray | None,
    downscale: float,
    history: list[TemporalStackFrame],
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not args.temporal_stack_peaks or cur_full is None:
        return []
    full_g = base.ensure_gray(cur_full)
    temp_map = temporal_stack_residual_map(
        full_g,
        history,
        parse_int_offsets(args.temporal_stack_offsets),
        args.temporal_stack_min_frames,
        downscale,
        args,
    )
    if temp_map is None:
        return []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(full_g)
    base_peaks: list[tuple[float, float, float, int]] = []
    for radius in parse_radii(args.temporal_stack_radii):
        native_map = compact_dark_map_native(full_g, radius)
        clahe_map = compact_dark_map_native(clahe, radius, texture_weight=0.015)
        score_map = (
            temp_map
            + args.temporal_stack_native_weight * native_map
            + args.temporal_stack_clahe_weight * clahe_map
        )
        base_peaks.extend(local_maxima_peaks(score_map, radius, args.temporal_stack_top_k))

    base_peaks = dedupe_full_peaks(base_peaks, args.temporal_stack_nms_px, args.temporal_stack_top_k)
    halo_offsets = parse_xy_offsets(args.temporal_stack_halo_offsets)
    halo_peaks: list[tuple[float, float, float, int]] = []
    for score, x, y, radius in base_peaks[: args.temporal_stack_halo_bases]:
        for dx, dy in halo_offsets:
            shifted_score = score - args.temporal_stack_halo_penalty * math.hypot(dx, dy)
            halo_peaks.append((shifted_score, x + dx, y + dy, radius))
    peaks = dedupe_full_peaks(
        halo_peaks,
        max(2.5, 0.45 * args.temporal_stack_nms_px),
        args.temporal_stack_top_k,
    )

    h_img, w_img = cur_g.shape[:2]
    cands: list[base.Candidate] = []
    for score, x_full, y_full, radius in peaks:
        side_full = max(3, int(round(2 * radius + 1)))
        side = max(3, int(round(side_full * downscale)))
        cx = x_full * downscale
        cy = y_full * downscale
        bx = max(0, min(w_img - side, int(round(cx - 0.5 * side))))
        by = max(0, min(h_img - side, int(round(cy - 0.5 * side))))
        bbox = (bx, by, side, side)
        mask = np.zeros_like(cur_g, dtype=np.uint8)
        mask[by : by + side, bx : bx + side] = 255
        cand = base.candidate_score("temporal_stack", bbox, side * side, residual_blur, app_resp, mask, cur_g)
        cand.map_score = float(score)
        cand.score = 0.35 * cand.score + args.temporal_stack_score_weight * float(score)
        if cand.score <= 0.1:
            continue
        cands.append(cand)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def candidate_duplicate(cand: base.Candidate, kept: list[base.Candidate]) -> bool:
    ccx, ccy = base.bbox_center(cand.bbox)
    for other in kept:
        ocx, ocy = base.bbox_center(other.bbox)
        if math.hypot(ccx - ocx, ccy - ocy) <= 2.5 or base.bbox_iou(cand.bbox, other.bbox) > 0.25:
            return True
    return False


def dedupe_candidates(
    cands: list[base.Candidate],
    max_n: int,
    score_fn: Callable[[base.Candidate], float] | None = None,
) -> list[base.Candidate]:
    deduped: list[base.Candidate] = []
    key = score_fn if score_fn is not None else (lambda c: c.score)
    for cand in sorted(cands, key=key, reverse=True):
        if not candidate_duplicate(cand, deduped):
            deduped.append(cand)
        if len(deduped) >= max_n:
            break
    return deduped


def hybrid_coast_candidates(
    states: list[PathState],
    tbd: BeamTBD,
    frame_h: np.ndarray | None,
    w_img: int,
    h_img: int,
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    """Use current beam states as cheap predicted proposals, not decisions."""
    if not args.hybrid_coast_proposals or not states:
        return []
    offsets = parse_xy_offsets(args.hybrid_coast_offsets)
    cands: list[base.Candidate] = []
    occupied: list[base.Candidate] = []
    ranked_states = sorted(states, key=lambda st: tbd.verified_score(st), reverse=True)
    highres_score_maps: dict[int, np.ndarray] = {}
    highres_gray: np.ndarray | None = None
    if args.hybrid_coast_highres_recenter and cur_full is not None and downscale > 0:
        highres_gray = base.ensure_gray(cur_full)
        highres_score_maps = {
            r: compact_dark_map_native(highres_gray, r, args.hybrid_coast_recenter_texture_weight)
            for r in parse_radii(args.hybrid_coast_recenter_radii)
        }
    for st in ranked_states[: args.hybrid_coast_top_k]:
        if st.misses > args.hybrid_coast_max_misses:
            continue
        if st.hit_count() < args.hybrid_coast_min_hits:
            continue
        if tbd.verified_score(st) < args.hybrid_coast_min_verified_score:
            continue
        base_bbox = warp_bbox(st.bbox, frame_h, w_img, h_img)
        bx, by, bw, bh = base_bbox
        for dx, dy in offsets:
            pred_bbox = base.clip_bbox_float((bx + st.vx + dx, by + st.vy + dy, bw, bh), w_img, h_img)
            pred_x, pred_y, pred_w, pred_h = pred_bbox
            pred_cx = pred_x + 0.5 * pred_w
            pred_cy = pred_y + 0.5 * pred_h
            candidate_boxes: list[tuple[tuple[int, int, int, int], float, str]] = [(pred_bbox, 0.0, "hybrid_coast")]

            if highres_gray is not None and highres_score_maps:
                peaks = local_native_peaks_near(
                    highres_score_maps,
                    (pred_cx / downscale, pred_cy / downscale),
                    args.hybrid_coast_recenter_radius_det_px / downscale,
                    args.hybrid_coast_recenter_peaks,
                )
                for peak_score, x_full, y_full, radius in peaks:
                    cx = x_full * downscale
                    cy = y_full * downscale
                    side = max(3, int(round((2 * radius + 1) * downscale)))
                    recentered = base.clip_bbox_float((cx - 0.5 * side, cy - 0.5 * side, side, side), w_img, h_img)
                    shift_det = math.hypot(cx - pred_cx, cy - pred_cy)
                    recentered_score = peak_score - args.hybrid_coast_recenter_shift_penalty * shift_det
                    candidate_boxes.append((recentered, recentered_score, "hybrid_coast_highres"))

            state_score = max(0.0, min(12.0, tbd.verified_score(st)))
            for bbox, recentered_score, source in candidate_boxes:
                mask = np.zeros_like(cur_g, dtype=np.uint8)
                x, y, w, h = bbox
                mask[y : y + h, x : x + w] = 255
                cand = base.candidate_score(source, bbox, max(1, w * h), residual_blur, app_resp, mask, cur_g)
                evidence_score = float(cand.score)
                if evidence_score < args.hybrid_coast_min_evidence and source == "hybrid_coast":
                    continue
                if source == "hybrid_coast_highres" and evidence_score < args.hybrid_coast_min_evidence and recentered_score <= 0.0:
                    continue
                cand.map_score = recentered_score if source == "hybrid_coast_highres" else evidence_score
                cand.score = (
                    0.70 * evidence_score
                    + args.hybrid_coast_score_weight * state_score
                    + (args.hybrid_coast_recenter_score_weight * recentered_score if source == "hybrid_coast_highres" else 0.0)
                )
                if cand.score <= 0.1 or candidate_duplicate(cand, occupied):
                    continue
                occupied.append(cand)
                cands.append(cand)
                if len(cands) >= args.hybrid_coast_top_k:
                    return sorted(cands, key=lambda c: c.score, reverse=True)
    return sorted(cands, key=lambda c: c.score, reverse=True)[: max(1, args.target_local_recovery_top_k)]


def target_local_recovery_candidates(
    seed: TargetLocalRecoverySeed | None,
    frame_no: int,
    w_img: int,
    h_img: int,
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    """Search a small native-resolution window around the last accepted track.

    This is intentionally narrower than hybrid coast. It does not introduce a
    new acquisition path; it only gives an already/recently accepted track a
    few centered high-res options when terrain proposals drift off target.
    """
    if not args.target_local_recovery_proposals or seed is None:
        return []
    gap = seed.age(frame_no)
    if gap <= 0 or gap > args.target_local_recovery_max_seed_gap:
        return []
    if seed.hits < args.target_local_recovery_min_hits:
        return []
    if seed.verified_score < args.target_local_recovery_min_verified_score:
        return []
    if cur_full is None or downscale <= 0:
        return []

    full_g = base.ensure_gray(cur_full)
    score_maps = {
        r: compact_dark_map_native(full_g, r, args.target_local_recovery_texture_weight)
        for r in parse_radii(args.target_local_recovery_radii)
    }

    side = max(3, int(round(args.target_local_recovery_box_det_px)))
    state_score = max(0.0, min(12.0, seed.verified_score))
    source = (
        "target_local_path_bank"
        if getattr(args, "target_local_recovery_predictor", "clamped_velocity") == "path_bank"
        else "target_local_recovery"
    )
    cands: list[base.Candidate] = []
    occupied: list[base.Candidate] = []
    for predicted, _method in target_local_seed_prediction_bank(seed, frame_no, w_img, h_img, args):
        pred_cx, pred_cy = base.bbox_center(predicted)
        peaks = local_native_peaks_near(
            score_maps,
            (pred_cx / downscale, pred_cy / downscale),
            args.target_local_recovery_search_radius_det_px / downscale,
            args.target_local_recovery_top_k,
        )
        for peak_score, x_full, y_full, _radius in peaks:
            cx = x_full * downscale
            cy = y_full * downscale
            bbox = base.clip_bbox_float((cx - 0.5 * side, cy - 0.5 * side, side, side), w_img, h_img)
            shift_det = math.hypot(cx - pred_cx, cy - pred_cy)
            recentered_score = float(peak_score - args.target_local_recovery_shift_penalty * shift_det)
            x, y, w, h = bbox
            mask = np.zeros_like(cur_g, dtype=np.uint8)
            mask[y : y + h, x : x + w] = 255
            cand = base.candidate_score(source, bbox, max(1, w * h), residual_blur, app_resp, mask, cur_g)
            cand.map_score = recentered_score
            cand.score = (
                0.40 * float(cand.score)
                + args.target_local_recovery_score_weight * recentered_score
                + args.target_local_recovery_state_score_weight * state_score
            )
            if cand.score <= 0.1 or candidate_duplicate(cand, occupied):
                continue
            occupied.append(cand)
            cands.append(cand)
    return sorted(cands, key=lambda c: c.score, reverse=True)


def parse_family_quota(raw: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        try:
            quotas[key] = max(0, int(value.strip()))
        except ValueError:
            continue
    return quotas


def candidate_seed_family(cand: base.Candidate) -> str:
    return str(getattr(cand, "recenter_seed_family", "cheap") or "cheap")


def raw_seed_source_family(source: str) -> str:
    source = str(source or "").strip()
    if source == "native_map":
        return "native"
    return source or "unknown"


def clone_recenter_seed(
    cand: base.Candidate,
    source: str,
    family: str,
) -> base.Candidate:
    seed = base.Candidate(
        source=source,
        bbox=cand.bbox,
        area=cand.area,
        fill=cand.fill,
        aspect=cand.aspect,
        mean_residual=cand.mean_residual,
        mean_appearance=cand.mean_appearance,
        local_contrast=cand.local_contrast,
        texture=cand.texture,
        line_context=cand.line_context,
        isolation=cand.isolation,
        score=cand.score,
    )
    for attr in (
        "map_score",
        "attached_support",
        "native_dark_score",
        "sky_like",
        "local_flow_score",
        "local_flow_delta_px",
        "local_flow_bg_sigma",
        "local_flow_inside_n",
        "local_flow_annulus_n",
        "local_bg_residual",
        "local_residual_ratio",
        "local_comp_score",
        "stab_x",
        "stab_y",
        "router_state",
        "router_confidence",
        "router_close_sky_like",
        "router_close_texture",
        "router_far_sky_like",
        "router_far_texture",
    ):
        if hasattr(cand, attr):
            setattr(seed, attr, getattr(cand, attr))
    seed.recenter_seed_family = family
    return seed


def raw_low_rank_recenter_seed_candidates(
    cheap_cands: list[base.Candidate],
    args: argparse.Namespace,
) -> list[base.Candidate]:
    families = _parse_source_set(getattr(args, "candidate_local_recenter_seed_families", "cheap"))
    if "raw_low_rank" not in families or not cheap_cands:
        return []

    rank_min = max(1, int(getattr(args, "candidate_local_recenter_raw_seed_rank_min", 25)))
    rank_max = max(rank_min, int(getattr(args, "candidate_local_recenter_raw_seed_rank_max", 400)))
    top_k = max(0, int(getattr(args, "candidate_local_recenter_raw_seed_top_k", 48)))
    if top_k <= 0:
        return []
    grid_px = max(1.0, float(getattr(args, "candidate_local_recenter_raw_seed_grid_px", 12.0)))
    source_quota = parse_family_quota(getattr(args, "candidate_local_recenter_raw_seed_source_quota", ""))
    if not source_quota:
        source_quota = {"motion": 20, "appearance": 12, "map": 12, "native": 12, "large_dark": 8}
    source_counts: dict[str, int] = {key: 0 for key in source_quota}
    seen_cells: set[tuple[str, int, int]] = set()
    out: list[base.Candidate] = []

    ranked = sorted(cheap_cands, key=lambda cand: candidate_obs(cand, args), reverse=True)
    for raw_rank, cand in enumerate(ranked, start=1):
        if raw_rank < rank_min:
            continue
        if raw_rank > rank_max:
            break
        source_family = raw_seed_source_family(cand.source)
        quota = source_quota.get(source_family, 0)
        if quota <= 0 or source_counts.get(source_family, 0) >= quota:
            continue
        cx, cy = base.bbox_center(cand.bbox)
        cell = (source_family, int(cx // grid_px), int(cy // grid_px))
        if cell in seen_cells:
            continue

        seed = clone_recenter_seed(cand, "raw_low_rank_seed", "raw_low_rank")
        seed.recenter_seed_raw_source = cand.source
        seed.recenter_seed_raw_source_family = source_family
        seed.recenter_seed_raw_rank = raw_rank
        seed.recenter_seed_raw_score = float(cand.score)
        seed.recenter_seed_raw_obs = float(candidate_obs(cand, args))
        out.append(seed)
        seen_cells.add(cell)
        source_counts[source_family] = source_counts.get(source_family, 0) + 1
        if len(out) >= top_k:
            break

    return out


def path_state_recenter_seed_candidates(
    states: list[PathState],
    tbd: BeamTBD,
    frame_no: int,
    w_img: int,
    h_img: int,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    families = _parse_source_set(getattr(args, "candidate_local_recenter_seed_families", "cheap"))
    if not ({"track_state", "previous_recenter"} & families):
        return []

    max_age = max(0, int(getattr(args, "candidate_local_recenter_track_state_max_age", 3)))
    min_hits = max(0, int(getattr(args, "candidate_local_recenter_track_state_min_hits", 2)))
    max_misses = max(0, int(getattr(args, "candidate_local_recenter_track_state_max_misses", 2)))
    min_score = float(getattr(args, "candidate_local_recenter_track_state_min_verified_score", -5.0))
    max_velocity = max(0.0, float(getattr(args, "candidate_local_recenter_track_state_max_velocity_px", 4.0)))
    out: list[base.Candidate] = []

    for st in states:
        age = max(0, frame_no - st.last_frame)
        if age > max_age or st.misses > max_misses or st.hit_count() < min_hits:
            continue
        score = float(tbd.verified_score(st))
        if score < min_score:
            continue

        last_source = st.last_candidate.source if st.last_candidate is not None else "track_only"
        family = "previous_recenter" if last_source == "candidate_local_recenter" else "track_state"
        if family not in families:
            continue

        x, y, bw, bh = st.bbox
        dt = max(1, age)
        vx = float(st.vx)
        vy = float(st.vy)
        step = math.hypot(vx * dt, vy * dt)
        if max_velocity > 0.0 and step > max_velocity * dt:
            scale = (max_velocity * dt) / max(1e-6, step)
            vx *= scale
            vy *= scale
        bbox = base.clip_bbox_float((x + vx * dt, y + vy * dt, bw, bh), w_img, h_img)
        last = st.last_candidate
        cand = base.Candidate(
            source=f"{family}_seed",
            bbox=bbox,
            area=max(1, int(bbox[2] * bbox[3])),
            fill=float(getattr(last, "fill", 1.0)) if last is not None else 1.0,
            aspect=float(getattr(last, "aspect", 1.0)) if last is not None else 1.0,
            mean_residual=float(getattr(last, "mean_residual", 0.0)) if last is not None else 0.0,
            mean_appearance=float(getattr(last, "mean_appearance", 0.0)) if last is not None else 0.0,
            local_contrast=float(getattr(last, "local_contrast", 0.0)) if last is not None else 0.0,
            texture=float(getattr(last, "texture", 0.0)) if last is not None else 0.0,
            line_context=float(getattr(last, "line_context", 0.0)) if last is not None else 0.0,
            isolation=float(getattr(last, "isolation", 0.0)) if last is not None else 0.0,
            score=score,
        )
        for attr in (
            "map_score",
            "attached_support",
            "native_dark_score",
            "sky_like",
            "router_state",
            "router_confidence",
            "router_close_sky_like",
            "router_close_texture",
            "router_far_sky_like",
            "router_far_texture",
        ):
            if last is not None and hasattr(last, attr):
                setattr(cand, attr, getattr(last, attr))
        cand.recenter_seed_family = family
        cand.recenter_seed_track_id = st.sid
        cand.recenter_seed_hits = st.hit_count()
        cand.recenter_seed_misses = st.misses
        cand.recenter_seed_age = age
        cand.recenter_seed_verified_score = score
        cand.recenter_seed_last_source = last_source
        cand.recenter_seed_vx = st.vx
        cand.recenter_seed_vy = st.vy
        out.append(cand)

    out.sort(key=lambda cand: float(getattr(cand, "score", 0.0)), reverse=True)
    return out[: max(0, int(getattr(args, "candidate_local_recenter_track_state_seed_top_k", 24)))]


def track_only_replay_seed_candidates(
    states: list[PathState],
    tbd: BeamTBD,
    frame_no: int,
    target_frame_no: int,
    w_img: int,
    h_img: int,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    if not getattr(args, "candidate_local_recenter_track_only_replay", False):
        return []

    top_k = max(0, int(getattr(args, "candidate_local_recenter_track_only_replay_top_k", 32)))
    rank_max = max(1, int(getattr(args, "candidate_local_recenter_track_only_replay_rank_max", 80)))
    if top_k <= 0:
        return []
    max_misses = max(0, int(getattr(args, "candidate_local_recenter_track_only_replay_max_misses", 2)))
    min_hits = max(0, int(getattr(args, "candidate_local_recenter_track_only_replay_min_hits", 1)))
    max_age = max(0, int(getattr(args, "candidate_local_recenter_track_only_replay_max_age", 4)))
    min_side = max(0.0, float(getattr(args, "candidate_local_recenter_track_only_replay_min_side", 3.0)))
    max_side = max(min_side, float(getattr(args, "candidate_local_recenter_track_only_replay_max_side", 14.0)))
    max_velocity = max(0.0, float(getattr(args, "candidate_local_recenter_track_state_max_velocity_px", 4.0)))
    dt = max(1, int(target_frame_no - frame_no))
    if dt > max_age:
        return []

    def has_replay_ancestry(st: PathState) -> bool:
        last = st.last_candidate
        if last is not None and getattr(last, "recenter_seed_family", "") == "track_only_replay":
            return True
        for hist in st.candidate_history[-4:]:
            if hist and hist.get("recenter_seed_family") == "track_only_replay":
                return True
        return False

    ranked = sorted(((float(tbd.verified_score(st)), st) for st in states), key=lambda item: item[0], reverse=True)
    out: list[base.Candidate] = []
    for rank, (score, st) in enumerate(ranked[:rank_max], start=1):
        cand_is_current = st.misses == 0 and st.last_candidate is not None
        if cand_is_current:
            continue
        if has_replay_ancestry(st):
            continue
        if st.misses > max_misses or st.hit_count() < min_hits:
            continue
        _x, _y, bw, bh = st.bbox
        side = max(float(bw), float(bh))
        if side < min_side or side > max_side:
            continue

        vx = float(st.vx)
        vy = float(st.vy)
        step = math.hypot(vx * dt, vy * dt)
        if max_velocity > 0.0 and step > max_velocity * dt:
            scale = (max_velocity * dt) / max(1e-6, step)
            vx *= scale
            vy *= scale
        bbox = base.clip_bbox_float((st.bbox[0] + vx * dt, st.bbox[1] + vy * dt, bw, bh), w_img, h_img)
        last = st.last_candidate
        cand = base.Candidate(
            source="track_only_replay_seed",
            bbox=bbox,
            area=max(1, int(bbox[2] * bbox[3])),
            fill=float(getattr(last, "fill", 1.0)) if last is not None else 1.0,
            aspect=float(getattr(last, "aspect", 1.0)) if last is not None else float(bbox[2]) / max(1.0, float(bbox[3])),
            mean_residual=float(getattr(last, "mean_residual", 0.0)) if last is not None else 0.0,
            mean_appearance=float(getattr(last, "mean_appearance", 0.0)) if last is not None else 0.0,
            local_contrast=float(getattr(last, "local_contrast", 0.0)) if last is not None else 0.0,
            texture=float(getattr(last, "texture", 0.0)) if last is not None else 0.0,
            line_context=float(getattr(last, "line_context", 0.0)) if last is not None else 0.0,
            isolation=float(getattr(last, "isolation", 1.0)) if last is not None else 1.0,
            score=score,
        )
        for attr in (
            "map_score",
            "attached_support",
            "native_dark_score",
            "sky_like",
            "router_state",
            "router_confidence",
            "router_close_sky_like",
            "router_close_texture",
            "router_far_sky_like",
            "router_far_texture",
        ):
            if last is not None and hasattr(last, attr):
                setattr(cand, attr, getattr(last, attr))
        cand.recenter_seed_family = "track_only_replay"
        cand.recenter_seed_track_id = st.sid
        cand.recenter_seed_hits = st.hit_count()
        cand.recenter_seed_misses = st.misses
        cand.recenter_seed_age = dt
        cand.recenter_seed_verified_score = score
        cand.recenter_seed_last_source = "track_only"
        cand.recenter_seed_vx = st.vx
        cand.recenter_seed_vy = st.vy
        cand.recenter_seed_raw_rank = rank
        out.append(cand)
        if len(out) >= top_k:
            break

    return out


def select_candidate_local_recenter_seeds(
    seed_cands: list[base.Candidate],
    args: argparse.Namespace,
) -> list[base.Candidate]:
    limit = max(1, int(args.candidate_local_recenter_seed_top_k))
    allowed_families = _parse_source_set(getattr(args, "candidate_local_recenter_seed_families", "cheap"))
    if getattr(args, "candidate_local_recenter_track_only_replay", False):
        allowed_families.add("track_only_replay")
    if allowed_families:
        seed_cands = [cand for cand in seed_cands if candidate_seed_family(cand) in allowed_families]
    scored = sorted(seed_cands, key=lambda cand: candidate_obs(cand, args), reverse=True)
    quotas = parse_family_quota(getattr(args, "candidate_local_recenter_seed_family_quota", ""))
    if quotas:
        kept: list[base.Candidate] = []
        seen: set[int] = set()
        for family, quota in quotas.items():
            if quota <= 0:
                continue
            family_rows = [cand for cand in scored if candidate_seed_family(cand) == family]
            for cand in family_rows[:quota]:
                if len(kept) >= limit:
                    break
                kept.append(cand)
                seen.add(id(cand))
        for cand in scored:
            if len(kept) >= limit:
                break
            if id(cand) not in seen:
                kept.append(cand)
                seen.add(id(cand))
        return kept[:limit]

    mode = getattr(args, "candidate_local_recenter_seed_mode", "score")
    if mode != "spatial_grid":
        return scored[:limit]

    grid_px = max(4.0, float(getattr(args, "candidate_local_recenter_seed_grid_px", 24.0)))
    best_by_cell: dict[tuple[int, int], tuple[float, base.Candidate]] = {}
    for cand in scored:
        cx, cy = base.bbox_center(cand.bbox)
        cell = (int(cx // grid_px), int(cy // grid_px))
        obs = candidate_obs(cand, args)
        previous = best_by_cell.get(cell)
        if previous is None or obs > previous[0]:
            best_by_cell[cell] = (obs, cand)

    kept: list[base.Candidate] = [
        cand for _obs, cand in sorted(best_by_cell.values(), key=lambda item: item[0], reverse=True)
    ][:limit]
    seen = {id(cand) for cand in kept}
    for cand in scored:
        if len(kept) >= limit:
            break
        if id(cand) not in seen:
            kept.append(cand)
            seen.add(id(cand))
    return kept


def candidate_local_recenter_candidates(
    seed_cands: list[base.Candidate],
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> list[base.Candidate]:
    """Recenter top cheap proposals with a native compact-dark local search."""
    if not args.candidate_local_recenter_proposals or cur_full is None or downscale <= 0 or not seed_cands:
        return []
    seed_pool = list(seed_cands)
    if args.candidate_local_recenter_router_scope == "surface_context":
        scoped = [
            cand
            for cand in seed_pool
            if getattr(cand, "router_state", "unknown")
            in {"surface_backed", "boundary_mixed", "sky_target_near_surface", "unknown", "unrouted"}
        ]
        if scoped:
            seed_pool = scoped
    seed_pool = select_candidate_local_recenter_seeds(seed_pool, args)

    full_g = base.ensure_gray(cur_full)
    score_maps = build_recenter_score_maps(
        full_g,
        parse_radii(args.candidate_local_recenter_radii),
        parse_response_maps(getattr(args, "candidate_local_recenter_response_maps", "compact_dark")),
        args.candidate_local_recenter_texture_weight,
    )
    side = max(3, int(round(args.candidate_local_recenter_box_det_px)))
    h_img, w_img = cur_g.shape[:2]
    cands: list[base.Candidate] = []
    occupied: list[base.Candidate] = []
    for seed_rank, seed in enumerate(seed_pool, start=1):
        seed_cx, seed_cy = base.bbox_center(seed.bbox)
        peaks = local_native_peaks_near_with_meta(
            score_maps,
            (seed_cx / downscale, seed_cy / downscale),
            args.candidate_local_recenter_radius_det_px / downscale,
            max(1, min(getattr(args, "candidate_local_recenter_peaks_per_seed", 4), args.candidate_local_recenter_top_k)),
        )
        seed_score = max(0.0, min(30.0, candidate_obs(seed, args)))
        for peak in peaks:
            peak_score = float(peak["score"])
            x_full = float(peak["x"])
            y_full = float(peak["y"])
            radius = int(peak["radius"])
            cx = x_full * downscale
            cy = y_full * downscale
            shift_det = math.hypot(cx - seed_cx, cy - seed_cy)
            recentered_score = float(peak_score - args.candidate_local_recenter_shift_penalty * shift_det)
            bbox = base.clip_bbox_float((cx - 0.5 * side, cy - 0.5 * side, side, side), w_img, h_img)
            x, y, w, h = bbox
            mask = np.zeros_like(cur_g, dtype=np.uint8)
            mask[y : y + h, x : x + w] = 255
            cand = base.candidate_score("candidate_local_recenter", bbox, max(1, w * h), residual_blur, app_resp, mask, cur_g)
            cand.map_score = recentered_score
            cand.score = (
                0.35 * float(cand.score)
                + args.candidate_local_recenter_score_weight * recentered_score
                + args.candidate_local_recenter_seed_score_weight * seed_score
            )
            cand.router_state = getattr(seed, "router_state", cand.router_state)
            cand.router_confidence = getattr(seed, "router_confidence", cand.router_confidence)
            cand.sky_like = getattr(seed, "sky_like", cand.sky_like)
            cand.router_close_sky_like = getattr(seed, "router_close_sky_like", cand.router_close_sky_like)
            cand.router_close_texture = getattr(seed, "router_close_texture", cand.router_close_texture)
            cand.router_far_sky_like = getattr(seed, "router_far_sky_like", cand.router_far_sky_like)
            cand.router_far_texture = getattr(seed, "router_far_texture", cand.router_far_texture)
            cand.recenter_parent_source = seed.source
            cand.recenter_parent_router_state = getattr(seed, "router_state", "unrouted")
            cand.recenter_parent_score = float(getattr(seed, "score", 0.0))
            cand.recenter_parent_bbox = tuple(seed.bbox)
            cand.recenter_seed_family = candidate_seed_family(seed)
            cand.recenter_seed_track_id = getattr(seed, "recenter_seed_track_id", "")
            cand.recenter_seed_hits = getattr(seed, "recenter_seed_hits", "")
            cand.recenter_seed_misses = getattr(seed, "recenter_seed_misses", "")
            cand.recenter_seed_age = getattr(seed, "recenter_seed_age", "")
            cand.recenter_seed_verified_score = getattr(seed, "recenter_seed_verified_score", "")
            cand.recenter_seed_last_source = getattr(seed, "recenter_seed_last_source", "")
            cand.recenter_seed_raw_source = getattr(seed, "recenter_seed_raw_source", "")
            cand.recenter_seed_raw_source_family = getattr(seed, "recenter_seed_raw_source_family", "")
            cand.recenter_seed_raw_rank = getattr(seed, "recenter_seed_raw_rank", "")
            cand.recenter_seed_raw_score = getattr(seed, "recenter_seed_raw_score", "")
            cand.recenter_seed_raw_obs = getattr(seed, "recenter_seed_raw_obs", "")
            cand.recenter_shift_det = shift_det
            cand.recenter_peak_radius = radius
            cand.recenter_peak_score = float(peak_score)
            cand.recenter_second_peak_margin = float(peak.get("second_peak_margin", 0.0))
            cand.recenter_seed_rank = int(seed_rank)
            cand.recenter_seed_score = seed_score
            cand.recenter_response_family = str(peak.get("response_family", "compact_dark"))
            cand.recenter_response_map = str(peak.get("response_map", "compact_dark"))
            cand.recenter_response_radius = radius
            cand.recenter_response_score = float(peak_score)
            cand.recenter_response_local_rank = int(peak.get("local_rank", 0))
            cand.recenter_subpixel_dx = float(peak.get("subpixel_dx", 0.0)) * downscale
            cand.recenter_subpixel_dy = float(peak.get("subpixel_dy", 0.0)) * downscale
            cand.recenter_subpixel_method = str(peak.get("subpixel_method", "integer"))
            cand.recenter_subpixel_condition = float(peak.get("subpixel_condition", 0.0))
            if cand.score <= 0.1 or candidate_duplicate(cand, occupied):
                continue
            occupied.append(cand)
            cands.append(cand)
    return sorted(cands, key=lambda c: c.score, reverse=True)[: args.candidate_local_recenter_top_k]


def target_local_anchor_bank_predictions(
    anchor: RuntimeLocalAnchor,
    frame_no: int,
    w_img: int,
    h_img: int,
    args: argparse.Namespace,
) -> list[tuple[tuple[int, int, int, int], str]]:
    dt = max(1, anchor.age(frame_no))
    x, y, w, h = anchor.bbox
    vx = float(anchor.vx)
    vy = float(anchor.vy)
    max_step = max(0.0, float(getattr(args, "target_local_recovery_max_velocity_px", 3.0))) * dt
    speed = math.hypot(vx * dt, vy * dt)
    if max_step > 0.0 and speed > max_step:
        scale = max_step / max(1e-6, speed)
        vx *= scale
        vy *= scale
    base_preds = [
        ((x, y, w, h), "previous"),
        ((x + vx * dt, y + vy * dt, w, h), "clamped_velocity"),
    ]
    offsets = parse_xy_offsets(getattr(args, "target_local_anchor_bank_offsets", "0:0"))
    out: list[tuple[tuple[int, int, int, int], str]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for raw_bbox, method in base_preds:
        bx, by, bw, bh = raw_bbox
        for dx, dy in offsets:
            bbox = base.clip_bbox_float((bx + dx, by + dy, bw, bh), w_img, h_img)
            suffix = "center" if abs(dx) < 1e-6 and abs(dy) < 1e-6 else f"off{dx:g}_{dy:g}"
            key = (bbox[0], bbox[1], bbox[2], bbox[3], method)
            if key in seen:
                continue
            seen.add(key)
            out.append((bbox, f"{method}_{suffix}"))
    return out


def target_local_anchor_bank_candidates(
    anchor_bank: RuntimeAnchorBank,
    frame_no: int,
    w_img: int,
    h_img: int,
    cur_full: np.ndarray | None,
    downscale: float,
    residual_blur: np.ndarray,
    app_resp: np.ndarray,
    cur_g: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[base.Candidate], dict[str, object]]:
    if not args.target_local_anchor_bank_proposals or cur_full is None or downscale <= 0:
        return [], {"enabled": bool(args.target_local_anchor_bank_proposals), "used": False}
    start = time.perf_counter()
    budget_ms = max(0.0, float(getattr(args, "target_local_anchor_bank_max_ms", 4.0)))
    full_g = base.ensure_gray(cur_full)
    score_maps = {
        r: compact_dark_map_native(full_g, r, args.target_local_anchor_bank_texture_weight)
        for r in parse_radii(args.target_local_anchor_bank_radii)
    }
    max_predictions = max(1, int(getattr(args, "target_local_anchor_bank_max_predictions", 10)))
    peaks_per_prediction = max(1, int(getattr(args, "target_local_anchor_bank_peaks_per_prediction", 2)))
    top_k = max(1, int(getattr(args, "target_local_anchor_bank_top_k", 8)))
    side_floor = max(3, int(round(getattr(args, "target_local_anchor_bank_min_side", 4.0))))
    cands: list[base.Candidate] = []
    occupied: list[base.Candidate] = []
    prediction_count = 0
    budget_hit = False
    for anchor in anchor_bank.anchors[: max(1, int(getattr(args, "target_local_anchor_bank_max_anchors", 2)))]:
        predictions = target_local_anchor_bank_predictions(anchor, frame_no, w_img, h_img, args)
        for predicted, method in predictions:
            if prediction_count >= max_predictions:
                break
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if budget_ms > 0.0 and elapsed_ms > budget_ms:
                budget_hit = True
                break
            prediction_count += 1
            pred_cx, pred_cy = base.bbox_center(predicted)
            peaks = local_native_peaks_near(
                score_maps,
                (pred_cx / downscale, pred_cy / downscale),
                args.target_local_recovery_search_radius_det_px / downscale,
                peaks_per_prediction,
            )
            for peak_score, x_full, y_full, _radius in peaks:
                cx = x_full * downscale
                cy = y_full * downscale
                side = max(side_floor, int(round(anchor.bbox[2])))
                bbox = base.clip_bbox_float((cx - 0.5 * side, cy - 0.5 * side, side, side), w_img, h_img)
                if anchor_bank.in_quarantine(bbox, frame_no):
                    continue
                shift_det = math.hypot(cx - pred_cx, cy - pred_cy)
                recentered_score = float(peak_score - args.target_local_anchor_bank_shift_penalty * shift_det)
                if recentered_score < args.target_local_anchor_bank_min_map_score:
                    continue
                x, y, w, h = bbox
                mask = np.zeros_like(cur_g, dtype=np.uint8)
                mask[y : y + h, x : x + w] = 255
                cand = base.candidate_score("target_local_anchor_bank", bbox, max(1, w * h), residual_blur, app_resp, mask, cur_g)
                cand.map_score = recentered_score
                cand.score = (
                    0.35 * float(cand.score)
                    + args.target_local_anchor_bank_score_weight * recentered_score
                    + args.target_local_anchor_bank_anchor_trust_weight * anchor.trust_score
                )
                cand.anchor_id = anchor.aid
                cand.anchor_source = anchor.source
                cand.anchor_age = anchor.age(frame_no)
                cand.anchor_trust = anchor.trust_score
                cand.anchor_prediction = method
                cand.anchor_shift_px = shift_det
                cand.target_local_anchor_id = anchor.aid
                cand.target_local_anchor_source = anchor.source
                cand.target_local_anchor_age = anchor.age(frame_no)
                cand.target_local_anchor_trust = anchor.trust_score
                cand.target_local_anchor_method = method
                cand.target_local_anchor_shift_px = shift_det
                if cand.score <= 0.1 or candidate_duplicate(cand, occupied):
                    continue
                occupied.append(cand)
                cands.append(cand)
                if len(cands) >= top_k:
                    break
            if len(cands) >= top_k or budget_hit:
                break
        if len(cands) >= top_k or budget_hit or prediction_count >= max_predictions:
            break
    cands = sorted(cands, key=lambda c: c.score, reverse=True)[:top_k]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return cands, {
        "enabled": True,
        "used": bool(cands),
        "anchors": len(anchor_bank.anchors),
        "predictions": prediction_count,
        "candidates": len(cands),
        "budget_hit": budget_hit,
        "elapsed_ms": round(elapsed_ms, 3),
    }


def candidate_scenario(cand: base.Candidate, use_router: bool = False) -> str:
    if cand.source in {"hybrid_coast", "target_local_recovery", "target_local_anchor_bank", "target_local_path_bank"}:
        return "coast"
    if cand.source == "large_dark" or max(cand.bbox[2], cand.bbox[3]) >= 10:
        return "large"
    if use_router:
        state = getattr(cand, "router_state", "unknown")
        if state == "clean_sky":
            return "sky"
        if state == "surface_backed":
            return "surface"
        if state in {"boundary_mixed", "sky_target_near_surface", "line_attached"}:
            return "boundary"
    sky = float(getattr(cand, "sky_like", 0.0))
    texture = float(getattr(cand, "texture", 0.0))
    line = float(getattr(cand, "line_context", 0.0))
    if sky >= 0.25 and texture < 45.0 and line < 0.55:
        return "sky"
    if sky < 0.10 or texture >= 45.0:
        return "surface"
    return "boundary"


def scenario_balanced_candidates(
    cands: list[base.Candidate],
    args: argparse.Namespace,
    use_router: bool = False,
    max_n: int | None = None,
) -> list[base.Candidate]:
    if not args.scenario_balance:
        return dedupe_candidates(cands, max_n or args.top_k_candidates)
    limit = max(1, max_n or args.top_k_candidates)
    quotas = {
        "sky": max(0, args.scenario_sky_top_k),
        "surface": max(0, args.scenario_surface_top_k),
        "boundary": max(0, args.scenario_boundary_top_k),
        "large": max(0, args.scenario_large_top_k),
        "coast": max(0, args.scenario_coast_top_k),
    }
    quota_sum = sum(quotas.values())
    if quota_sum > limit:
        scaled: dict[str, int] = {}
        remainder: list[tuple[float, str]] = []
        for bucket, quota in quotas.items():
            raw = quota * limit / quota_sum if quota > 0 else 0.0
            scaled[bucket] = int(math.floor(raw))
            remainder.append((raw - scaled[bucket], bucket))
        remaining = limit - sum(scaled.values())
        for _, bucket in sorted(remainder, reverse=True):
            if remaining <= 0:
                break
            if quotas[bucket] > 0:
                scaled[bucket] += 1
                remaining -= 1
        quotas = scaled
    kept: list[base.Candidate] = []
    by_bucket: dict[str, list[base.Candidate]] = {k: [] for k in quotas}
    sort_key = (
        (lambda c: candidate_obs(c, args))
        if candidate_obs_sort_needed(args, use_router)
        else (lambda c: c.score)
    )
    for cand in sorted(cands, key=sort_key, reverse=True):
        by_bucket.setdefault(candidate_scenario(cand, use_router), []).append(cand)
    for bucket, quota in quotas.items():
        for cand in by_bucket.get(bucket, [])[:quota]:
            if len(kept) >= limit:
                return kept
            if not candidate_duplicate(cand, kept):
                kept.append(cand)
    for cand in sorted(cands, key=sort_key, reverse=True):
        if len(kept) >= limit:
            break
        if not candidate_duplicate(cand, kept):
            kept.append(cand)
    return kept


def candidate_obs_sort_needed(args: argparse.Namespace, use_router: bool) -> bool:
    return bool(
        use_router
        or args.native_roi_score
        or args.line_weight > 0
        or args.support_penalty_weight > 0
        or args.app_low_residual_penalty > 0
    )


def candidate_obs(c: base.Candidate, args: argparse.Namespace) -> float:
    # Existing score is the main high-recall objectness cue. Add small explicit
    # penalties for line/texture contexts that repeatedly produce false boxes.
    line = getattr(c, "line_context", 0.0)
    obs = args.obs_weight * c.score - args.line_weight * line
    if router_applies(args):
        router_state = getattr(c, "router_state", "unknown")
        if candidate_surface_source(c) and router_state != "surface_backed":
            obs -= args.router_surface_source_penalty
        if router_state == "line_attached":
            obs -= args.router_line_penalty * max(0.5, getattr(c, "router_confidence", 0.5))
        if router_state == "surface_backed":
            obs += args.router_surface_bonus
    if args.support_penalty_weight > 0:
        excess_support = max(0.0, getattr(c, "attached_support", 0.0) - args.support_penalty_threshold)
        obs -= args.support_penalty_weight * excess_support
    if args.native_roi_score:
        native_excess = max(0.0, getattr(c, "native_dark_score", 0.0) - args.native_roi_neutral)
        obs += args.native_roi_weight * native_excess
    if (
        args.app_low_residual_penalty > 0
        and c.source == "appearance"
        and getattr(c, "map_score", 0.0) <= 0.0
        and c.mean_residual < args.app_low_residual_px
    ):
        deficit = (args.app_low_residual_px - c.mean_residual) / max(1e-3, args.app_low_residual_px)
        obs -= args.app_low_residual_penalty * deficit
    return obs


def attached_support_score(
    cur_g: np.ndarray,
    gx_abs: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """Score dark/edge structure continuing below a candidate, e.g. pole tops."""
    x, y, w, h = bbox
    h_img, w_img = cur_g.shape[:2]
    cx = int(round(x + 0.5 * w))
    half_col = max(2, int(round(0.5 * max(3, w + 2))))
    y0 = min(h_img, y + h + 1)
    y1 = min(h_img, y0 + max(24, 5 * max(w, h)))
    if y1 <= y0 + 4:
        return 0.0

    x0 = max(0, cx - half_col)
    x1 = min(w_img, cx + half_col + 1)
    below = cur_g[y0:y1, x0:x1].astype(np.float32)
    if below.size == 0:
        return 0.0

    side_pad = max(4, 2 * half_col)
    sx0 = max(0, x0 - side_pad)
    sx1 = min(w_img, x1 + side_pad)
    context = cur_g[y0:y1, sx0:sx1].astype(np.float32)
    mask = np.ones(context.shape, dtype=bool)
    mask[:, x0 - sx0 : x1 - sx0] = False
    side = context[mask]
    dark_column = max(0.0, float(np.median(side) - np.median(below))) if side.size else 0.0

    edge_column = float(np.mean(gx_abs[y0:y1, x0:x1]))
    return float(dark_column + 0.05 * edge_column)


def assign_attached_support(cands: list[base.Candidate], cur_g: np.ndarray, only_missing: bool = False) -> None:
    if only_missing:
        cands = [cand for cand in cands if getattr(cand, "router_state", "unrouted") == "unrouted"]
    if not cands:
        return
    gx_abs = np.abs(cv2.Sobel(cur_g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    for cand in cands:
        cand.attached_support = attached_support_score(cur_g, gx_abs, cand.bbox)


def native_dark_score_for_bbox(
    full_g: np.ndarray,
    bbox: tuple[int, int, int, int],
    inv_downscale: float,
) -> float:
    x, y, w, h = bbox
    h_full, w_full = full_g.shape[:2]
    cx = int(round((x + 0.5 * w) * inv_downscale))
    cy = int(round((y + 0.5 * h) * inv_downscale))
    radius = max(1, int(round(0.55 * max(w, h) * inv_downscale)))
    outer = max(radius + 2, int(round(3.0 * radius)))
    x0 = max(0, cx - outer)
    y0 = max(0, cy - outer)
    x1 = min(w_full, cx + outer + 1)
    y1 = min(h_full, cy + outer + 1)
    patch = full_g[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    inner_mask = dist2 <= radius * radius
    ring_mask = (dist2 > radius * radius) & (dist2 <= outer * outer)
    if not np.any(inner_mask) or not np.any(ring_mask):
        return 0.0
    inner = patch[inner_mask]
    ring = patch[ring_mask]
    mad = float(np.median(np.abs(ring - np.median(ring))))
    return float(max(0.0, (np.median(ring) - np.mean(inner)) / (1.4826 * mad + 3.0)))


def assign_native_roi_scores(
    cands: list[base.Candidate],
    cur_full: np.ndarray | None,
    downscale: float,
) -> None:
    if cur_full is None or abs(downscale - 1.0) < 1e-6 or downscale <= 0:
        return
    full_g = base.ensure_gray(cur_full)
    inv_downscale = 1.0 / downscale
    for cand in cands:
        cand.native_dark_score = native_dark_score_for_bbox(full_g, cand.bbox, inv_downscale)


def sky_like_score(
    cur_g: np.ndarray,
    grad_mag: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    x, y, w, h = bbox
    h_img, w_img = cur_g.shape[:2]
    cx = int(round(x + 0.5 * w))
    cy = int(round(y + 0.5 * h))
    inner = max(2, int(round(0.8 * max(w, h))))
    outer = max(inner + 4, int(round(4.0 * max(w, h))))
    x0 = max(0, cx - outer)
    x1 = min(w_img, cx + outer + 1)
    y0 = max(0, cy - outer)
    y1 = min(h_img, cy + outer + 1)
    patch = cur_g[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    ring = (dist2 >= inner * inner) & (dist2 <= outer * outer)
    if not np.any(ring):
        ring = np.ones(patch.shape, dtype=bool)
    vals = patch[ring]
    mean_bg = float(np.mean(vals))
    std_bg = float(np.std(vals))
    grad_bg = float(np.mean(grad_mag[y0:y1, x0:x1][ring]))

    bright = np.clip((mean_bg - 95.0) / 75.0, 0.0, 1.0)
    smooth = np.clip((38.0 - std_bg) / 32.0, 0.0, 1.0)
    low_grad = np.clip((38.0 - grad_bg) / 38.0, 0.0, 1.0)
    return float(bright * smooth * low_grad)


def assign_sky_context(cands: list[base.Candidate], cur_g: np.ndarray) -> None:
    gray_f = cur_g.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    for cand in cands:
        cand.sky_like = sky_like_score(cur_g, grad_mag, cand.bbox)


def router_is_active(args: argparse.Namespace) -> bool:
    return args.candidate_router != "off" or args.runtime_mode != "baseline"


def router_applies(args: argparse.Namespace) -> bool:
    return args.candidate_router == "apply"


def runtime_budget_applies(args: argparse.Namespace) -> bool:
    return args.candidate_router == "apply" or args.runtime_mode in {"clean_sky", "boundary", "surface"}


def candidate_surface_source(cand: base.Candidate) -> bool:
    return cand.source in {"temporal_stack", "temporal_stack_local"}


def surface_ranker_transition_features_needed(
    st: PathState,
    cand: base.Candidate,
    args: argparse.Namespace,
) -> bool:
    """Return whether a transition can become surface-ranker eligible.

    Surface-ranker feature histories are comparatively expensive because they
    require additional target-vs-background pair diagnostics. When the ranker
    is explicitly scoped to surface-backed tubes, avoid computing those
    diagnostics for paths that cannot pass the same scope check.
    """
    if args.surface_ranker_scope == "all":
        return True
    states = {"surface_backed"}
    if args.surface_ranker_scope == "surface_context":
        states |= {"boundary_mixed", "sky_target_near_surface"}
    window = max(1, int(args.window))
    recent = st.candidate_history[-max(0, window - 1) :]
    scoped_hits = 0
    hits = 0
    for item in recent:
        if item is None:
            continue
        hits += 1
        if str(item.get("router_state", "unrouted")) in states:
            scoped_hits += 1
    hits += 1
    if str(getattr(cand, "router_state", "unrouted")) in states:
        scoped_hits += 1
    return (scoped_hits / max(1, hits)) >= args.surface_ranker_min_rate


def existing_surface_track(states: list[PathState], args: argparse.Namespace) -> bool:
    for st in states[: max(1, args.beam_width // 3)]:
        if st.misses > args.max_selected_misses:
            continue
        if st.hit_count() < args.surface_branch_track_min_hits:
            continue
        if st.score() < args.surface_branch_track_min_score:
            continue
        features = tube_features(st)
        if features.get("router_surface_backed_rate", 0.0) >= args.surface_branch_track_rate:
            return True
    return False


def surface_branch_needed(
    cands: list[base.Candidate],
    states: list[PathState],
    frame_decision: FrameRouterDecision,
    args: argparse.Namespace,
) -> bool:
    if not router_applies(args):
        return True
    if args.runtime_mode == "clean_sky" or args.runtime_mode == "boundary":
        return False
    if args.runtime_mode == "surface":
        return True
    has_surface_track = existing_surface_track(states, args)
    if frame_decision.mode != "surface" and not has_surface_track:
        return False
    if has_surface_track:
        return True
    if args.runtime_mode == "auto" and not args.surface_branch_allow_acquisition:
        return False
    if not args.surface_branch_allow_acquisition and not has_surface_track:
        return False
    strong_surface = [
        cand
        for cand in cands
        if getattr(cand, "router_state", "unknown") == "surface_backed"
        and cand.score >= args.surface_branch_min_score
    ]
    if len(strong_surface) >= args.surface_branch_min_candidates:
        return True
    return existing_surface_track(states, args)


def frame_router_decision(cur_g: np.ndarray, args: argparse.Namespace) -> FrameRouterDecision:
    gray = cur_g.astype(np.float32)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(np.mean(grad))
    edge_density = float(np.mean(grad > 42.0))
    bright_frac = float(np.mean(gray > 220.0))
    texture = 0.55 * std + 0.45 * grad_mean
    bright = np.clip((mean - 95.0) / 75.0, 0.0, 1.0)
    smooth = np.clip((38.0 - std) / 32.0, 0.0, 1.0)
    low_grad = np.clip((38.0 - grad_mean) / 38.0, 0.0, 1.0)
    sky_like = float(bright * smooth * low_grad)
    features = {
        "mean": mean,
        "std": std,
        "grad": grad_mean,
        "edge_density": edge_density,
        "texture": texture,
        "bright_frac": bright_frac,
        "sky_like": sky_like,
    }

    if args.runtime_mode in {"clean_sky", "boundary", "surface"}:
        mode = args.runtime_mode
        conf = 1.0
    elif sky_like >= 0.22 and texture < 36.0 and edge_density < 0.20:
        mode = "clean_sky"
        conf = min(1.0, 0.55 + sky_like)
    elif texture >= 56.0 or edge_density >= 0.34:
        mode = "surface"
        conf = min(1.0, 0.45 + texture / 120.0)
    elif 0.08 <= sky_like < 0.24 or 0.20 <= edge_density < 0.34:
        mode = "boundary"
        conf = 0.62
    else:
        mode = "unknown"
        conf = 0.35

    if mode == "clean_sky":
        max_candidates = min(args.top_k_candidates, max(24, args.scenario_sky_top_k + args.scenario_large_top_k))
    elif mode == "surface":
        max_candidates = args.top_k_candidates
    elif mode == "boundary":
        max_candidates = min(args.top_k_candidates, max(40, args.scenario_boundary_top_k + args.scenario_sky_top_k))
    else:
        max_candidates = args.top_k_candidates
    return FrameRouterDecision(
        mode=mode,
        allow_surface_extras=(mode == "surface"),
        max_candidates=max_candidates,
        confidence=float(conf),
        features=features,
    )


def _router_ring_stats(
    cur_g: np.ndarray,
    grad: np.ndarray,
    bbox: tuple[int, int, int, int],
    inner_mult: float,
    outer_mult: float,
) -> dict[str, float]:
    x, y, w, h = bbox
    h_img, w_img = cur_g.shape[:2]
    cx = int(round(x + 0.5 * w))
    cy = int(round(y + 0.5 * h))
    radius = max(4.0, float(max(w, h)))
    inner = inner_mult * radius
    outer = outer_mult * radius
    x0 = max(0, int(math.floor(cx - outer)))
    x1 = min(w_img, int(math.ceil(cx + outer + 1)))
    y0 = max(0, int(math.floor(cy - outer)))
    y1 = min(h_img, int(math.ceil(cy + outer + 1)))
    if x1 <= x0 or y1 <= y0:
        return {"mean": 0.0, "std": 0.0, "grad": 0.0, "texture": 0.0, "sky_like": 0.0}
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    mask = (dist2 >= inner * inner) & (dist2 <= outer * outer)
    if not np.any(mask):
        return {"mean": 0.0, "std": 0.0, "grad": 0.0, "texture": 0.0, "sky_like": 0.0}
    patch = cur_g[y0:y1, x0:x1].astype(np.float32)
    gpatch = grad[y0:y1, x0:x1]
    vals = patch[mask]
    gvals = gpatch[mask]
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    grad_mean = float(np.mean(gvals))
    texture = 0.55 * std + 0.45 * grad_mean
    bright = np.clip((mean - 95.0) / 75.0, 0.0, 1.0)
    smooth = np.clip((38.0 - std) / 32.0, 0.0, 1.0)
    low_grad = np.clip((38.0 - grad_mean) / 38.0, 0.0, 1.0)
    sky_like = float(bright * smooth * low_grad)
    return {"mean": mean, "std": std, "grad": grad_mean, "texture": texture, "sky_like": sky_like}


def _integral_rect_sum(integ: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])


def _router_box_ring_stats(
    gray_i: np.ndarray,
    gray2_i: np.ndarray,
    grad_i: np.ndarray,
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    inner_mult: float,
    outer_mult: float,
) -> dict[str, float]:
    h_img, w_img = shape
    x, y, w, h = bbox
    cx = int(round(x + 0.5 * w))
    cy = int(round(y + 0.5 * h))
    radius = max(4.0, float(max(w, h)))
    inner = int(round(inner_mult * radius))
    outer = int(round(outer_mult * radius))
    ox0 = max(0, cx - outer)
    ox1 = min(w_img, cx + outer + 1)
    oy0 = max(0, cy - outer)
    oy1 = min(h_img, cy + outer + 1)
    ix0 = max(0, cx - inner)
    ix1 = min(w_img, cx + inner + 1)
    iy0 = max(0, cy - inner)
    iy1 = min(h_img, cy + inner + 1)
    outer_n = max(0, ox1 - ox0) * max(0, oy1 - oy0)
    inner_n = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    n = outer_n - inner_n
    if n <= 0:
        return {"mean": 0.0, "std": 0.0, "grad": 0.0, "texture": 0.0, "sky_like": 0.0}

    sum_g = _integral_rect_sum(gray_i, ox0, oy0, ox1, oy1) - _integral_rect_sum(gray_i, ix0, iy0, ix1, iy1)
    sum_g2 = _integral_rect_sum(gray2_i, ox0, oy0, ox1, oy1) - _integral_rect_sum(gray2_i, ix0, iy0, ix1, iy1)
    sum_grad = _integral_rect_sum(grad_i, ox0, oy0, ox1, oy1) - _integral_rect_sum(grad_i, ix0, iy0, ix1, iy1)
    mean = sum_g / n
    var = max(0.0, sum_g2 / n - mean * mean)
    std = math.sqrt(var)
    grad_mean = sum_grad / n
    texture = 0.55 * std + 0.45 * grad_mean
    bright = np.clip((mean - 95.0) / 75.0, 0.0, 1.0)
    smooth = np.clip((38.0 - std) / 32.0, 0.0, 1.0)
    low_grad = np.clip((38.0 - grad_mean) / 38.0, 0.0, 1.0)
    sky_like = float(bright * smooth * low_grad)
    return {"mean": mean, "std": std, "grad": grad_mean, "texture": texture, "sky_like": sky_like}


def classify_candidate_router_state(
    cand: base.Candidate,
    close: dict[str, float],
    far: dict[str, float],
) -> tuple[str, float]:
    close_sky = close["sky_like"] >= 0.22 and close["texture"] < 32.0
    far_sky = far["sky_like"] >= 0.18 and far["texture"] < 38.0
    close_surface = close["sky_like"] < 0.08 and close["texture"] >= 38.0
    far_surface = far["sky_like"] < 0.12 and far["texture"] >= 42.0
    line = float(getattr(cand, "line_context", 0.0))
    support = float(getattr(cand, "attached_support", 0.0))

    if support >= 5.5 or (line >= 0.85 and support >= 2.5):
        return "line_attached", min(1.0, 0.55 + 0.08 * support + 0.25 * line)
    if close_sky and far_sky:
        return "clean_sky", min(1.0, 0.45 + 0.5 * close["sky_like"] + 0.35 * far["sky_like"])
    if close_sky and far_surface:
        return "sky_target_near_surface", min(1.0, 0.55 + 0.4 * close["sky_like"])
    if close_surface:
        return "surface_backed", min(1.0, 0.45 + close["texture"] / 120.0)
    if far_surface or abs(close["texture"] - far["texture"]) >= 18.0:
        return "boundary_mixed", 0.62
    return "unknown", 0.35


def assign_candidate_router_states(
    cands: list[base.Candidate],
    cur_g: np.ndarray,
    only_missing: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if only_missing:
        unrouted: list[base.Candidate] = []
        for cand in cands:
            state = getattr(cand, "router_state", "unrouted")
            if state == "unrouted":
                unrouted.append(cand)
            else:
                counts[str(state)] = counts.get(str(state), 0) + 1
        cands = unrouted
    if not cands:
        return counts
    gray = cur_g.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    gray_i = cv2.integral(gray)
    gray2_i = cv2.integral(gray * gray)
    grad_i = cv2.integral(grad)
    for cand in cands:
        close = _router_box_ring_stats(gray_i, gray2_i, grad_i, cur_g.shape[:2], cand.bbox, 1.4, 3.2)
        far = _router_box_ring_stats(gray_i, gray2_i, grad_i, cur_g.shape[:2], cand.bbox, 3.8, 7.0)
        state, confidence = classify_candidate_router_state(cand, close, far)
        cand.router_state = state
        cand.router_confidence = confidence
        cand.sky_like = close["sky_like"]
        cand.router_close_sky_like = close["sky_like"]
        cand.router_close_texture = close["texture"]
        cand.router_far_sky_like = far["sky_like"]
        cand.router_far_texture = far["texture"]
        counts[state] = counts.get(state, 0) + 1
    return counts


def candidate_router_state_counts(cands: list[base.Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cand in cands:
        state = getattr(cand, "router_state", "unrouted")
        counts[state] = counts.get(state, 0) + 1
    return counts


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def tube_features(st: PathState) -> dict[str, float]:
    hist = st.candidate_history[-9:]
    cands = [c for c in hist if c is not None]
    n = max(1, len(hist))
    hits = len(cands)
    source_vals = [c.get("source", "") for c in cands]
    router_vals = [str(c.get("router_state", "unrouted")) for c in cands]
    map_scores = [float(c.get("map_score", 0.0)) for c in cands]
    line_vals = [float(c.get("line_context", 0.0)) for c in cands]
    support_vals = [float(c.get("attached_support", 0.0)) for c in cands]
    native_vals = [float(c.get("native_dark_score", 0.0)) for c in cands]
    sky_vals = [float(c.get("sky_like", 0.0)) for c in cands]
    texture_vals = [float(c.get("texture", 0.0)) for c in cands]
    residual_vals = [float(c.get("mean_residual", 0.0)) for c in cands]
    appearance_vals = [float(c.get("mean_appearance", 0.0)) for c in cands]
    score_vals = [float(c.get("score", 0.0)) for c in cands]
    area_vals = [float(c.get("area", 0.0)) for c in cands]
    tiny_vals = [
        1.0
        for c in cands
        if c.get("bbox")
        and float(c["bbox"][2]) <= 4.0
        and float(c["bbox"][3]) <= 4.0
    ]
    pair_vals = [float(v) for v in st.pair_history[-9:]]
    pair_raw_vals = [float(v) for v in st.pair_raw_history[-9:]]
    pair_bg_vals = [float(v) for v in st.pair_bg_history[-9:]]
    pair_bg_local_vals = [float(v) for v in st.pair_bg_local_history[-9:]]
    align_vals = [float(v) for v in st.align_gain_history[-9:]]
    bg_dist_vals = [float(v) for v in st.bg_dist_history[-9:]]
    cv_resid_vals = [float(v) for v in st.cv_resid_history[-9:]]
    bg_minus_cv_vals = [float(v) for v in st.bg_minus_cv_history[-9:]]
    density_vals = [float(v) for v in st.cand_density_history[-9:]]
    speed_vals = [float(v) for v in st.speed_history[-9:]]
    accel_vals = [float(v) for v in st.accel_history[-9:]]
    app_only = [
        1.0
        for c in cands
        if c.get("source") == "appearance" and float(c.get("map_score", 0.0)) <= 0.0
    ]
    return {
        "tube_hits": float(hits),
        "tube_len": float(n),
        "hit_rate": hits / n,
        "miss_rate": 1.0 - hits / n,
        "mean_score": _mean(score_vals),
        "score_std": float(np.std(score_vals)) if len(score_vals) > 1 else 0.0,
        "mean_map_score": _mean(map_scores),
        "map_hit_rate": sum(1 for v in map_scores if v > 0.0) / max(1, hits),
        "appearance_only_rate": len(app_only) / max(1, hits),
        "surface_source_rate": sum(1 for v in source_vals if v in {"temporal_stack", "temporal_stack_local"})
        / max(1, hits),
        "router_surface_backed_rate": sum(1 for v in router_vals if v == "surface_backed") / max(1, hits),
        "router_clean_sky_rate": sum(1 for v in router_vals if v == "clean_sky") / max(1, hits),
        "router_boundary_rate": sum(1 for v in router_vals if v in {"boundary_mixed", "sky_target_near_surface"}) / max(1, hits),
        "router_line_attached_rate": sum(1 for v in router_vals if v == "line_attached") / max(1, hits),
        "router_unknown_rate": sum(1 for v in router_vals if v in {"unknown", "unrouted"}) / max(1, hits),
        "tiny_rate": len(tiny_vals) / max(1, hits),
        "mean_line_context": _mean(line_vals),
        "max_line_context": max(line_vals) if line_vals else 0.0,
        "mean_attached_support": _mean(support_vals),
        "max_attached_support": max(support_vals) if support_vals else 0.0,
        "mean_native_dark_score": _mean(native_vals),
        "max_native_dark_score": max(native_vals) if native_vals else 0.0,
        "mean_sky_like": _mean(sky_vals),
        "max_sky_like": max(sky_vals) if sky_vals else 0.0,
        "sky_hit_rate": sum(1 for v in sky_vals if v >= 0.2) / max(1, hits),
        "mean_texture": _mean(texture_vals),
        "mean_residual": _mean(residual_vals),
        "mean_appearance": _mean(appearance_vals),
        "mean_area": _mean(area_vals),
        "mean_pair_score": _mean(pair_vals),
        "positive_pair_rate": sum(1 for v in pair_vals if v > 0.2) / max(1, len(pair_vals)),
        "mean_pair_raw": _mean(pair_raw_vals),
        "positive_pair_raw_rate": sum(1 for v in pair_raw_vals if v > 0.2) / max(1, len(pair_raw_vals)),
        "mean_pair_bg": _mean(pair_bg_vals),
        "positive_pair_bg_rate": sum(1 for v in pair_bg_vals if v > 0.2) / max(1, len(pair_bg_vals)),
        "mean_pair_bg_local": _mean(pair_bg_local_vals),
        "positive_pair_bg_local_rate": sum(1 for v in pair_bg_local_vals if v > 0.2) / max(1, len(pair_bg_local_vals)),
        "mean_align_gain": _mean(align_vals),
        "positive_align_rate": sum(1 for v in align_vals if v > 0.15) / max(1, len(align_vals)),
        "mean_bg_dist": _mean(bg_dist_vals),
        "mean_cv_resid": _mean(cv_resid_vals),
        "mean_bg_minus_cv": _mean(bg_minus_cv_vals),
        "mean_cand_density": _mean(density_vals),
        "log_cand_density": math.log1p(_mean(density_vals)),
        "mean_speed": _mean(speed_vals),
        "max_speed": max(speed_vals) if speed_vals else 0.0,
        "mean_accel": _mean(accel_vals),
        "max_accel": max(accel_vals) if accel_vals else 0.0,
    }


def heuristic_tube_verifier_score(features: dict[str, float]) -> float:
    # A lightweight hand-fit verifier until there are enough labeled tubes to
    # train logistic/GBM weights. Positive terms reward tiny map-backed tubes;
    # negative terms target the reviewed hard negatives: app-only clutter,
    # attached poles/stems, line context, and unstable high-texture blobs.
    score = 0.0
    score += 1.6 * features["map_hit_rate"]
    score += 0.45 * min(3.0, features["mean_map_score"])
    score += 0.85 * features["tiny_rate"]
    score += 0.55 * features["positive_pair_rate"]
    score += 0.25 * max(-2.0, min(2.0, features["mean_pair_score"]))
    score += 0.35 * features["hit_rate"]
    score += 0.18 * min(3.0, features["mean_speed"])
    score += 0.35 * max(0.0, features.get("mean_native_dark_score", 0.0) - 1.0)
    score += 0.20 * max(0.0, features.get("max_native_dark_score", 0.0) - 1.4)
    if (
        features["positive_pair_rate"] >= 0.45
        and features["mean_pair_score"] >= 0.8
        and features["mean_residual"] >= 20.0
        and features["mean_appearance"] >= 25.0
    ):
        score += 2.0

    score -= 2.4 * features["appearance_only_rate"]
    score -= 0.9 * max(0.0, features["mean_line_context"] - 0.35)
    score -= 0.65 * max(0.0, features["mean_attached_support"] - 2.4)
    score -= 0.45 * max(0.0, features["max_attached_support"] - 5.5)
    score -= 0.35 * max(0.0, (features["mean_texture"] - 85.0) / 35.0)
    if features["appearance_only_rate"] > 0.7 and features["mean_speed"] < 0.45:
        score -= 1.4
    if features["appearance_only_rate"] > 0.7 and features["mean_residual"] < 2.0:
        score -= 1.0
    if (
        features["map_hit_rate"] > 0.8
        and features["appearance_only_rate"] < 0.1
        and features["positive_pair_rate"] < 0.1
        and features["mean_residual"] < 2.5
        and features["mean_appearance"] < 20.0
    ):
        score -= 14.0
    return float(score)


def _logsumexp(vals: list[float]) -> float:
    if not vals:
        return 0.0
    m = max(vals)
    return float(m + math.log(sum(math.exp(v - m) for v in vals)))


def likelihood_tube_verifier_score(features: dict[str, float]) -> float:
    """Approximate tube log-likelihood ratio against explicit clutter modes.

    This is still hand-weighted, but it changes the structure from additive
    one-off penalties to "target tube must beat the best clutter explanation".
    The score scale is kept near the existing heuristic verifier so old
    selected_score values remain interpretable.
    """
    pair = max(-2.5, min(3.0, features["mean_pair_score"]))
    align = max(-2.5, min(3.0, features.get("mean_align_gain", 0.0)))
    low_pair = 1.0 - features["positive_pair_rate"]
    low_align = 1.0 - features.get("positive_align_rate", 0.0)
    low_residual = max(0.0, (5.0 - features["mean_residual"]) / 5.0)
    low_appearance = max(0.0, (20.0 - features["mean_appearance"]) / 20.0)
    low_speed = max(0.0, (0.75 - features["mean_speed"]) / 0.75)
    high_density = min(3.0, features.get("log_cand_density", 0.0))
    residual_strength = min(3.0, features["mean_residual"] / 18.0)
    appearance_strength = min(3.0, features["mean_appearance"] / 30.0)

    target = (
        0.95 * features["hit_rate"]
        + 0.95 * features["map_hit_rate"]
        + 0.35 * min(3.0, features["mean_map_score"])
        + 1.15 * features["positive_pair_rate"]
        + 0.55 * pair
        + 1.25 * features.get("positive_align_rate", 0.0)
        + 0.75 * align
        + 0.35 * residual_strength
        + 0.30 * appearance_strength
        + 0.42 * max(0.0, features.get("mean_native_dark_score", 0.0) - 1.0)
        + 0.30 * features["tiny_rate"]
    )

    static_hotspot = (
        -0.15
        + 1.40 * features["map_hit_rate"]
        + 1.10 * low_pair
        + 1.20 * low_align
        + 1.20 * low_residual
        + 0.85 * low_appearance
        + 0.65 * low_speed
    )
    line_attached = (
        -0.25
        + 2.00 * max(0.0, features["mean_line_context"] - 0.28)
        + 0.55 * max(0.0, features["max_line_context"] - 0.55)
        + 0.45 * max(0.0, features["mean_attached_support"] - 1.8)
        + 0.30 * max(0.0, features["max_attached_support"] - 4.5)
    )
    parallax_edge = (
        -0.55
        + 0.55 * max(0.0, (features["mean_texture"] - 45.0) / 35.0)
        + 0.55 * max(0.0, features["mean_line_context"] - 0.25)
        + 0.35 * residual_strength
        + 0.55 * low_pair
        + 0.35 * high_density
    )
    appearance_only_blob = (
        -0.10
        + 2.00 * features["appearance_only_rate"]
        + 0.75 * low_pair
        + 0.85 * low_align
        + 0.70 * low_residual
        + 0.20 * appearance_strength
    )
    boundary_artifact = (
        -0.45
        + 0.80 * features.get("max_sky_like", 0.0)
        + 0.65 * max(0.0, features["mean_line_context"] - 0.18)
        + 0.50 * low_pair
        + 0.40 * low_align
    )
    noise = (
        -0.35
        + 1.30 * features["miss_rate"]
        + 0.55 * max(0.0, 3.0 - features["tube_hits"])
        + 0.35 * high_density
        + 0.35 * max(0.0, features["score_std"] - 8.0) / 8.0
    )
    null_score = _logsumexp([
        static_hotspot,
        line_attached,
        parallax_edge,
        appearance_only_blob,
        boundary_artifact,
        noise,
    ])

    search_penalty = 0.18 * high_density + 0.12 * features["miss_rate"]
    score = 2.6 + target - 0.78 * null_score - search_penalty

    if (
        features["map_hit_rate"] > 0.8
        and features["positive_pair_rate"] < 0.1
        and features.get("positive_align_rate", 0.0) < 0.1
        and features["mean_residual"] < 2.5
        and features["mean_appearance"] < 20.0
    ):
        score -= 8.0
    if (
        features["positive_pair_rate"] >= 0.45
        and features["mean_pair_score"] >= 0.8
        and features.get("positive_align_rate", 0.0) >= 0.25
        and features["mean_residual"] >= 16.0
    ):
        score += 1.4
    return float(score)


def tube_verifier_score(features: dict[str, float], mode: str) -> float:
    if mode == "likelihood":
        return likelihood_tube_verifier_score(features)
    return heuristic_tube_verifier_score(features)


def surface_ranker_scope_rate(features: dict[str, float], args: argparse.Namespace) -> float:
    if args.surface_ranker_scope == "surface_backed":
        return features.get("router_surface_backed_rate", 0.0)
    if args.surface_ranker_scope == "surface_context":
        return features.get("router_surface_backed_rate", 0.0) + features.get("router_boundary_rate", 0.0)
    return 1.0


def surface_ranker_scope_allows(features: dict[str, float], args: argparse.Namespace) -> bool:
    if args.surface_ranker_scope == "all":
        return True
    return surface_ranker_scope_rate(features, args) >= args.surface_ranker_min_rate


def scoped_tube_verifier_score(features: dict[str, float], args: argparse.Namespace) -> float:
    if args.tube_verifier == "off":
        return 0.0
    if router_applies(args) and not surface_ranker_scope_allows(features, args):
        return 0.0
    return tube_verifier_score(features, args.tube_verifier)


def surface_ranker_applies(features: dict[str, float], args: argparse.Namespace) -> bool:
    if args.surface_ranker_policy == "off":
        return False
    return surface_ranker_scope_allows(features, args)


def state_feature_row(
    st: PathState,
    args: argparse.Namespace,
    rank: int,
    baseline_score: float,
    tube_score: float,
    competitor_margin: float | None = None,
) -> dict[str, object]:
    features = tube_features(st)
    eligible = st.misses <= args.max_selected_misses and st.hit_count() >= args.min_path_hits
    passes_floor = (
        st.score() >= args.selected_score
        if args.tube_verifier == "off"
        else tube_score >= args.tube_verifier_floor and baseline_score >= args.selected_score
    )
    row: dict[str, object] = {
        "rank": rank,
        "track_id": st.sid,
        "x": st.bbox[0],
        "y": st.bbox[1],
        "w": st.bbox[2],
        "h": st.bbox[3],
        "score": st.score(),
        "verified_score": baseline_score,
        "tube_verifier_score": tube_score,
        "eligible": int(eligible),
        "passes_floor": int(passes_floor),
        "selected": 0,
        "hits": st.hit_count(),
        "misses": st.misses,
        "vx": st.vx,
        "vy": st.vy,
        "competitor_margin": "" if competitor_margin is None else competitor_margin,
    }
    cand_is_current = st.misses == 0 and st.last_candidate is not None
    if cand_is_current:
        cand_json = st.last_candidate.to_json()
        for key, value in cand_json.items():
            if isinstance(value, (int, float, str)):
                row[f"cand_{key}"] = value
            elif isinstance(value, (list, tuple)):
                row[f"cand_{key}"] = json.dumps(value)
    for key, value in features.items():
        row[f"tube_{key}"] = float(value)
    return row


def ranker_row_float(row: dict[str, object], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value in (None, ""):
        return default
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return out if math.isfinite(out) else default


def surface_ranker_gate_allows(row: dict[str, object], gate: str) -> bool:
    if gate == "none":
        return True
    source = str(row.get("cand_source", ""))
    support = max(
        ranker_row_float(row, "cand_attached_support"),
        ranker_row_float(row, "tube_mean_attached_support"),
    )
    texture = max(
        ranker_row_float(row, "cand_texture"),
        ranker_row_float(row, "tube_mean_texture"),
    )
    sky = max(
        ranker_row_float(row, "cand_sky_like"),
        ranker_row_float(row, "tube_mean_sky_like"),
    )
    pair_bg = ranker_row_float(row, "tube_mean_pair_bg")
    if gate == "learned_not_map":
        return source != "map"
    if gate == "source_large_dark_or_appearance":
        return source in {"large_dark", "appearance"}
    if gate == "high_support":
        return support >= 8.0
    if gate == "high_texture_support":
        return texture >= 55.0 and support >= 6.0
    if gate == "large_dark_high_support":
        return source == "large_dark" and support >= 6.0
    if gate == "low_sky_high_support":
        return sky <= 0.02 and support >= 6.0
    if gate == "negative_bg_pair":
        return pair_bg <= 0.0
    if gate == "support_negative_bg_pair":
        return support >= 6.0 and pair_bg <= 0.0
    return False


class LearnedSurfaceRanker:
    def __init__(self, bundle_path: str):
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - depends on optional runtime install
            raise SystemExit("surface ranker requires joblib/scikit-learn in the active environment") from exc
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.numeric_features = list(bundle.get("numeric_features", []))
        self.source_features = list(bundle.get("source_features", []))

    @staticmethod
    def _float(value: object) -> float:
        if value in (None, ""):
            return float("nan")
        try:
            out = float(value)  # type: ignore[arg-type]
        except Exception:
            return float("nan")
        return out if math.isfinite(out) else float("nan")

    def vectorize(self, rows: list[dict[str, object]]) -> np.ndarray:
        data: list[list[float]] = []
        for row in rows:
            vals = [self._float(row.get(name)) for name in self.numeric_features]
            src = str(row.get("cand_source", ""))
            vals.extend(1.0 if name == f"src_{src}" else 0.0 for name in self.source_features)
            data.append(vals)
        return np.asarray(data, dtype=np.float64)

    def scores(self, rows: list[dict[str, object]]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        x = self.vectorize(rows)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        return self.model.decision_function(x)


class BeamTBD:
    def __init__(self, args: argparse.Namespace, px_per_frame: float):
        self.args = args
        self.px_per_frame = px_per_frame
        self.states: list[PathState] = []
        self.next_sid = 1
        self.surface_ranker = (
            LearnedSurfaceRanker(args.surface_ranker_model)
            if args.surface_ranker_policy != "off" and args.surface_ranker_model
            else None
        )
        self.last_surface_ranker_rows = 0
        self.last_surface_ranker_score: float | None = None
        self.last_surface_ranker_sid: int | None = None
        self.last_surface_ranker_used = False

    def verified_score(self, st: PathState) -> float:
        if self.args.tube_verifier == "off":
            return st.score()
        features = tube_features(st)
        tube_score = scoped_tube_verifier_score(features, self.args)
        sky_bonus = self.args.sky_bonus_weight * (
            features.get("max_sky_like", 0.0) + 0.5 * features.get("sky_hit_rate", 0.0)
        )
        density_penalty = self.args.density_penalty_weight * features.get("log_cand_density", 0.0)
        return st.score() + self.args.tube_verifier_weight * tube_score + sky_bonus - density_penalty

    def _ranked_eligible(self, eligible: list[PathState]) -> list[tuple[float, float, PathState]]:
        scored: list[tuple[float, float, PathState]] = []
        for st in eligible:
            features = tube_features(st)
            tube_score = scoped_tube_verifier_score(features, self.args)
            if self.args.tube_verifier != "off" and tube_score < self.args.tube_verifier_floor:
                continue
            scored.append((self.verified_score(st), tube_score, st))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _surface_ranker_fallback_best(self, scored: list[tuple[float, float, PathState]]) -> PathState | None:
        self.last_surface_ranker_rows = 0
        self.last_surface_ranker_score = None
        self.last_surface_ranker_sid = None
        if self.surface_ranker is None or self.args.surface_ranker_policy != "confidence_fallback":
            return None
        rows: list[dict[str, object]] = []
        states: list[PathState] = []
        limit = self.args.surface_ranker_top_n
        rescored = scored if limit <= 0 else scored[:limit]
        for rank, (baseline_score, tube_score, st) in enumerate(rescored, start=1):
            features = tube_features(st)
            if not surface_ranker_applies(features, self.args):
                continue
            next_score = scored[rank][0] if rank < len(scored) else None
            margin = (baseline_score - next_score) if next_score is not None else None
            row = state_feature_row(st, self.args, rank, baseline_score, tube_score, margin)
            rows.append(row)
            states.append(st)
        if not rows:
            return None
        self.last_surface_ranker_rows = len(rows)
        scores = self.surface_ranker.scores(rows)
        if len(scores) == 0:
            return None
        best_i = int(np.argmax(scores))
        self.last_surface_ranker_score = float(scores[best_i])
        self.last_surface_ranker_sid = states[best_i].sid
        if (
            float(scores[best_i]) >= self.args.surface_ranker_threshold
            and surface_ranker_gate_allows(rows[best_i], self.args.surface_ranker_gate)
        ):
            return states[best_i]
        return None

    def _sky_rescue_best(self) -> PathState | None:
        if not self.args.sky_rescue:
            return None
        rescued: list[tuple[float, PathState]] = []
        for st in self.states:
            if st.misses > self.args.max_selected_misses:
                continue
            if st.hit_count() < self.args.sky_rescue_min_hits:
                continue
            features = tube_features(st)
            if features["max_sky_like"] < self.args.sky_rescue_min_sky:
                continue
            if features["mean_line_context"] > self.args.sky_rescue_max_line:
                continue
            if features["mean_attached_support"] > self.args.sky_rescue_max_support:
                continue
            if features["appearance_only_rate"] > 0.35:
                continue
            score = self.verified_score(st)
            if score >= self.args.sky_rescue_score:
                rescued.append((score, st))
        if not rescued:
            return None
        return max(rescued, key=lambda x: x[0])[1]

    def _trim(self, contribs: list[float], hits: list[bool]) -> tuple[list[float], list[bool]]:
        n = max(1, self.args.window)
        return contribs[-n:], hits[-n:]

    def _append_score(self, prev_contribs: list[float], contrib: float) -> float:
        keep_prev = max(0, self.args.window - 1)
        if keep_prev == 0:
            return float(contrib)
        return float(sum(prev_contribs[-keep_prev:]) + contrib)

    def _transition_cost(
        self,
        st: PathState,
        cand: base.Candidate,
        frame_no: int,
        prev_ref_bbox: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float, float]:
        dt = max(1, frame_no - st.last_frame)
        pcx, pcy = base.bbox_center(prev_ref_bbox)
        ccx, ccy = base.bbox_center(cand.bbox)
        vx = (ccx - pcx) / dt
        vy = (ccy - pcy) / dt
        speed = math.hypot(vx, vy)
        allowed = self.px_per_frame * dt
        excess = max(0.0, speed - allowed)
        accel = math.hypot(vx - st.vx, vy - st.vy)
        cost = self.args.speed_weight * excess * excess + self.args.accel_weight * accel * accel
        if self.args.static_penalty_weight > 0:
            static_deficit = max(0.0, self.args.static_penalty_px - speed)
            cost += self.args.static_penalty_weight * static_deficit * static_deficit
        return cost, vx, vy, speed, accel

    def update(
        self,
        frame_no: int,
        cands: list[base.Candidate],
        signed_diff: np.ndarray,
        signed_sigma: float,
        w_img: int,
        h_img: int,
        frame_h: np.ndarray | None,
        residual_blur: np.ndarray,
        app_resp: np.ndarray,
    ) -> list[PathState]:
        cands = sorted(cands, key=lambda c: c.score, reverse=True)[: self.args.top_k_candidates]
        new_states: list[PathState] = []
        cand_density = float(len(cands))
        state_refs: list[
            tuple[
                PathState,
                tuple[int, int, int, int],
                tuple[int, int, int, int],
                tuple[int, int, int, int],
                tuple[float, float, int, int],
            ]
        ] = []
        need_extra_pair_features_global = (
            self.args.tube_verifier == "likelihood"
            or self.args.export_top_tubes > 0
            or (self.surface_ranker is not None and self.args.surface_ranker_scope == "all")
        )
        need_extra_pair_features_scoped = (
            self.surface_ranker is not None and self.args.surface_ranker_scope != "all"
        )
        for st in self.states:
            dt = max(1, frame_no - st.last_frame)
            if dt > self.args.max_misses + 1:
                continue
            transition_ref_bbox = (
                warp_bbox(st.bbox, frame_h, w_img, h_img)
                if self.args.bg_motion_transition
                else st.bbox
            )
            pair_ref_bbox = (
                warp_bbox(st.bbox, frame_h, w_img, h_img)
                if self.args.bg_motion_transition or self.args.bg_pair_geometry
                else st.bbox
            )
            bg_ref_bbox = warp_bbox(st.bbox, frame_h, w_img, h_img)
            x0, y0, bw, bh = st.bbox
            dt_f = max(1, frame_no - st.last_frame)
            cv_pred = (x0 + st.vx * dt_f, y0 + st.vy * dt_f, bw, bh)
            state_refs.append((st, transition_ref_bbox, pair_ref_bbox, bg_ref_bbox, cv_pred))

        for cand in cands:
            obs = candidate_obs(cand, self.args)
            birth_contrib = obs - self.args.birth_penalty
            birth_contribs, birth_hits = self._trim([birth_contrib], [True])
            birth = PathState(
                self.next_sid,
                cand.bbox,
                last_frame=frame_no,
                misses=0,
                age=1,
                hits=1,
                last_candidate=cand,
                contribs=birth_contribs,
                hit_flags=birth_hits,
                history=[(frame_no, cand.bbox, birth_contrib, True)],
                candidate_history=[cand.to_json()],
                pair_history=[0.0],
                pair_raw_history=[0.0],
                pair_bg_history=[0.0],
                pair_bg_local_history=[0.0],
                align_gain_history=[0.0],
                bg_dist_history=[0.0],
                cv_resid_history=[0.0],
                bg_minus_cv_history=[0.0],
                cand_density_history=[cand_density],
                speed_history=[0.0],
                accel_history=[0.0],
            )
            self.next_sid += 1
            best_state = birth
            best_score = birth.score()

            for st, transition_ref_bbox, pair_ref_bbox, bg_ref_bbox, cv_pred in state_refs:
                bg_dist = center_distance(cand.bbox, bg_ref_bbox)
                cv_resid = center_distance(cand.bbox, cv_pred)
                bg_minus_cv = bg_dist - cv_resid
                cost, vx, vy, speed, accel = self._transition_cost(st, cand, frame_no, transition_ref_bbox)
                if cost > 30.0:
                    continue
                pair = dark_dipole_score(
                    signed_diff,
                    pair_ref_bbox,
                    cand.bbox,
                    signed_sigma,
                    self.args.pair_radius,
                    self.args.pair_min_step_px,
                    self.args.local_pair_norm,
                    self.args.pair_search_penalty,
                )
                extra_pair_features = need_extra_pair_features_global or (
                    need_extra_pair_features_scoped
                    and surface_ranker_transition_features_needed(st, cand, self.args)
                )
                if extra_pair_features:
                    pair_raw = dark_dipole_score(
                        signed_diff,
                        st.bbox,
                        cand.bbox,
                        signed_sigma,
                        self.args.pair_radius,
                        self.args.pair_min_step_px,
                    )
                    pair_bg = dark_dipole_score(
                        signed_diff,
                        bg_ref_bbox,
                        cand.bbox,
                        signed_sigma,
                        self.args.pair_radius,
                        self.args.pair_min_step_px,
                    )
                    pair_bg_local = dark_dipole_score(
                        signed_diff,
                        bg_ref_bbox,
                        cand.bbox,
                        signed_sigma,
                        self.args.pair_radius,
                        self.args.pair_min_step_px,
                        True,
                    )
                    align_gain = alignment_gain_score(cand, bg_ref_bbox, residual_blur, app_resp)
                else:
                    pair_raw = pair
                    pair_bg = 0.0
                    pair_bg_local = 0.0
                    align_gain = 0.0
                contrib = obs + self.args.pair_weight * pair - cost
                score = self._append_score(st.contribs, contrib)
                if score <= best_score:
                    continue
                contribs, hits = self._trim(st.contribs + [contrib], st.hit_flags + [True])
                ns = PathState(
                    st.sid,
                    cand.bbox,
                    vx=0.65 * st.vx + 0.35 * vx,
                    vy=0.65 * st.vy + 0.35 * vy,
                    last_frame=frame_no,
                    misses=0,
                    age=st.age + 1,
                    hits=st.hits + 1,
                    last_candidate=cand,
                    contribs=contribs,
                    hit_flags=hits,
                    history=st.history[-self.args.window :] + [(frame_no, cand.bbox, contrib, True)],
                    candidate_history=(st.candidate_history + [cand.to_json()])[-self.args.window :],
                    pair_history=(st.pair_history + [pair])[-self.args.window :],
                    pair_raw_history=(st.pair_raw_history + [pair_raw])[-self.args.window :],
                    pair_bg_history=(st.pair_bg_history + [pair_bg])[-self.args.window :],
                    pair_bg_local_history=(st.pair_bg_local_history + [pair_bg_local])[-self.args.window :],
                    align_gain_history=(st.align_gain_history + [align_gain])[-self.args.window :],
                    bg_dist_history=(st.bg_dist_history + [bg_dist])[-self.args.window :],
                    cv_resid_history=(st.cv_resid_history + [cv_resid])[-self.args.window :],
                    bg_minus_cv_history=(st.bg_minus_cv_history + [bg_minus_cv])[-self.args.window :],
                    cand_density_history=(st.cand_density_history + [cand_density])[-self.args.window :],
                    speed_history=(st.speed_history + [speed])[-self.args.window :],
                    accel_history=(st.accel_history + [accel])[-self.args.window :],
                )
                best_score = score
                best_state = ns
            new_states.append(best_state)

        for st in self.states[: self.args.beam_width]:
            if st.misses >= self.args.max_misses:
                continue
            if self.args.bg_motion_transition:
                bg_pred = warp_bbox(st.bbox, frame_h, w_img, h_img)
                bx, by, bw, bh = bg_pred
                pred = base.clip_bbox_float((bx + st.vx, by + st.vy, bw, bh), w_img, h_img)
            else:
                pred = st.predict_bbox(w_img, h_img)
            contribs, hits = self._trim(st.contribs + [-self.args.miss_penalty], st.hit_flags + [False])
            missed = PathState(
                st.sid,
                pred,
                vx=st.vx,
                vy=st.vy,
                last_frame=frame_no,
                misses=st.misses + 1,
                age=st.age + 1,
                hits=st.hits,
                last_candidate=st.last_candidate,
                contribs=contribs,
                hit_flags=hits,
                history=st.history[-self.args.window :] + [(frame_no, pred, -self.args.miss_penalty, False)],
                candidate_history=(st.candidate_history + [None])[-self.args.window :],
                pair_history=(st.pair_history + [0.0])[-self.args.window :],
                pair_raw_history=(st.pair_raw_history + [0.0])[-self.args.window :],
                pair_bg_history=(st.pair_bg_history + [0.0])[-self.args.window :],
                pair_bg_local_history=(st.pair_bg_local_history + [0.0])[-self.args.window :],
                align_gain_history=(st.align_gain_history + [0.0])[-self.args.window :],
                bg_dist_history=(st.bg_dist_history + [0.0])[-self.args.window :],
                cv_resid_history=(st.cv_resid_history + [0.0])[-self.args.window :],
                bg_minus_cv_history=(st.bg_minus_cv_history + [0.0])[-self.args.window :],
                cand_density_history=(st.cand_density_history + [cand_density])[-self.args.window :],
                speed_history=(st.speed_history + [math.hypot(st.vx, st.vy)])[-self.args.window :],
                accel_history=(st.accel_history + [0.0])[-self.args.window :],
            )
            new_states.append(missed)

        # Merge duplicate-ish states by keeping the stronger state near each center.
        deduped: list[PathState] = []
        for st in sorted(new_states, key=lambda s: s.score(), reverse=True):
            scx, scy = base.bbox_center(st.bbox)
            duplicate = False
            for kept in deduped:
                kcx, kcy = base.bbox_center(kept.bbox)
                if math.hypot(scx - kcx, scy - kcy) < 3.0 and base.bbox_iou(st.bbox, kept.bbox) > 0.1:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(st)
            if len(deduped) >= self.args.beam_width:
                break
        self.states = deduped
        return self.states

    def raw_best_for_mask(self) -> PathState | None:
        eligible = [
            st
            for st in self.states
            if st.misses <= self.args.max_selected_misses and st.hit_count() >= self.args.min_path_hits
        ]
        if not eligible:
            return None
        best = max(eligible, key=lambda st: st.score())
        return best if best.score() >= self.args.selected_score else None

    def best(self) -> PathState | None:
        self.last_surface_ranker_used = False
        eligible = [
            st
            for st in self.states
            if st.misses <= self.args.max_selected_misses and st.hit_count() >= self.args.min_path_hits
        ]
        if not eligible:
            return None

        if (
            self.surface_ranker is None
            and self.args.tube_verifier == "off"
            and self.args.selection_margin <= 0.0
            and not self.args.sky_rescue
        ):
            best = max(eligible, key=lambda st: st.score())
            return best if best.score() >= self.args.selected_score else None

        scored = self._ranked_eligible(eligible)
        if not scored:
            return self._sky_rescue_best()

        learned_best = self._surface_ranker_fallback_best(scored)
        if learned_best is not None:
            learned_score = self.verified_score(learned_best)
            if learned_score >= self.args.selected_score:
                self.last_surface_ranker_used = True
                return learned_best

        best_score, _tube_score, best = scored[0]
        if self.args.selection_margin > 0.0 and len(scored) > 1:
            if best_score - scored[1][0] < self.args.selection_margin:
                return self._sky_rescue_best()
        if best_score >= self.args.selected_score:
            return best
        return self._sky_rescue_best()


def _parse_source_set(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def target_local_state_select_override(
    selected: PathState | None,
    states: list[PathState],
    seed: TargetLocalRecoverySeed | None,
    frame_no: int,
    w_img: int,
    h_img: int,
    tbd: BeamTBD,
    args: argparse.Namespace,
) -> tuple[PathState | None, dict[str, object]]:
    """Choose a current target-local recovery state only when it fixes a bad lock.

    The recovery proposal path can generate a centered candidate near the last
    trusted target-local track, while normal TBD ranking can still prefer a
    stronger terrain/edge lock. This override is deliberately narrow: it only
    runs from a recent trusted seed, only considers current-frame recovery
    states, and requires a clear prediction-error improvement over the selected
    state.
    """
    info: dict[str, object] = {"enabled": bool(getattr(args, "target_local_state_select", False)), "used": False}
    if not getattr(args, "target_local_state_select", False):
        return selected, info
    if seed is None:
        info["reason"] = "no_seed"
        return selected, info
    seed_age = seed.age(frame_no)
    info["seed_age"] = seed_age
    if seed_age <= 0 or seed_age > args.target_local_recovery_max_seed_gap:
        info["reason"] = "seed_gap"
        return selected, info
    if seed.hits < args.target_local_recovery_min_hits:
        info["reason"] = "seed_hits"
        return selected, info
    if seed.verified_score < args.target_local_recovery_min_verified_score:
        info["reason"] = "seed_score"
        return selected, info

    predicted = target_local_seed_prediction_bbox(seed, frame_no, w_img, h_img, args)
    pred_cx, pred_cy = base.bbox_center(predicted)
    selected_error = None
    if selected is not None:
        sel_cx, sel_cy = base.bbox_center(selected.bbox)
        seed_cx, seed_cy = base.bbox_center(seed.bbox)
        selected_error = math.hypot(sel_cx - pred_cx, sel_cy - pred_cy)
        selected_anchor_error = math.hypot(sel_cx - seed_cx, sel_cy - seed_cy)
        info["selected_error_px"] = round(selected_error, 3)
        info["selected_anchor_error_px"] = round(selected_anchor_error, 3)
        if selected_error < args.target_local_state_select_error_px:
            info["reason"] = "selected_near_prediction"
            return selected, info
        if selected_anchor_error <= args.target_local_state_select_anchor_px:
            info["reason"] = "selected_near_anchor"
            return selected, info

    allowed_sources = _parse_source_set(args.target_local_state_select_sources)
    ranked = sorted(states, key=lambda st: tbd.verified_score(st), reverse=True)
    if args.target_local_state_select_top_n > 0:
        ranked = ranked[: args.target_local_state_select_top_n]

    best: tuple[float, float, PathState] | None = None
    for rank, st in enumerate(ranked, start=1):
        source = st.last_candidate.source if st.last_candidate is not None else "missed_anchor"
        missed_anchor = st.misses > 0 or st.last_candidate is None
        if missed_anchor:
            if not getattr(args, "target_local_state_select_allow_missed_anchor", False):
                continue
            if st.misses > args.target_local_state_select_missed_max_misses:
                continue
        elif allowed_sources and source not in allowed_sources:
            continue
        _x, _y, bw, bh = st.bbox
        min_side = min(bw, bh)
        max_side = max(bw, bh)
        if min_side < args.target_local_state_select_min_side:
            continue
        if args.target_local_state_select_max_side > 0 and max_side > args.target_local_state_select_max_side:
            continue

        cx, cy = base.bbox_center(st.bbox)
        pred_error = math.hypot(cx - pred_cx, cy - pred_cy)
        seed_cx, seed_cy = base.bbox_center(seed.bbox)
        anchor_error = math.hypot(cx - seed_cx, cy - seed_cy)
        if (
            pred_error > args.target_local_state_select_max_pred_error_px
            and anchor_error > args.target_local_state_select_anchor_px
        ):
            continue
        if (
            selected_error is not None
            and pred_error + args.target_local_state_select_improvement_px > selected_error
        ):
            continue
        score = tbd.verified_score(st)
        candidate_key = (pred_error, -score, st)
        if best is None or candidate_key[:2] < best[:2]:
            best = candidate_key
            info.update(
                {
                    "candidate_rank": rank,
                    "candidate_source": source,
                    "candidate_error_px": round(pred_error, 3),
                    "candidate_anchor_error_px": round(anchor_error, 3),
                    "candidate_score": round(float(score), 3),
                }
            )

    if best is None:
        info["reason"] = "no_candidate"
        return selected, info

    chosen = best[2]
    info["used"] = True
    info["reason"] = "target_local_state_select"
    info["selected_track_id"] = chosen.sid
    return chosen, info


def srps_verified_state_select_override(
    selected: PathState | None,
    states: list[PathState],
    tbd: BeamTBD,
    args: argparse.Namespace,
) -> tuple[PathState | None, dict[str, object]]:
    info: dict[str, object] = {"enabled": bool(getattr(args, "srps_verified_candidate_priority", False)), "used": False}
    if not getattr(args, "srps_verified_candidate_priority", False):
        return selected, info
    current: list[PathState] = []
    for st in states:
        if st.misses != 0 or st.last_candidate is None:
            continue
        cand = st.last_candidate
        if int(getattr(cand, "srps_verified_path", 0) or 0) != 1:
            continue
        if str(getattr(cand, "srps_state", "")) not in {"confirmed_path", "coast"}:
            continue
        current.append(st)
    info["candidates"] = len(current)
    if not current:
        info["reason"] = "no_current_verified_srps"
        return selected, info

    def srps_key(st: PathState) -> tuple[float, float, float, int]:
        cand = st.last_candidate
        return (
            float(getattr(cand, "srps_verification_score", 0.0) or 0.0),
            float(getattr(cand, "srps_path_confidence", 0.0) or 0.0),
            float(tbd.verified_score(st)),
            st.hit_count(),
        )

    best = max(current, key=srps_key)
    if selected is not None and selected.sid == best.sid and selected.bbox == best.bbox:
        info["reason"] = "already_selected"
        return selected, info
    selected_has_verified_srps = (
        selected is not None
        and selected.misses == 0
        and selected.last_candidate is not None
        and int(getattr(selected.last_candidate, "srps_verified_path", 0) or 0) == 1
    )
    if selected_has_verified_srps and srps_key(selected) >= srps_key(best):
        info["reason"] = "selected_verified_srps_stronger"
        return selected, info
    cand = best.last_candidate
    info.update(
        {
            "used": True,
            "reason": "srps_verified_priority",
            "selected_track_id": None if selected is None else selected.sid,
            "srps_track_id": best.sid,
            "srps_state": str(getattr(cand, "srps_state", "")),
            "srps_seed_source": str(getattr(cand, "srps_seed_source", "")),
            "srps_verification_score": round(float(getattr(cand, "srps_verification_score", 0.0) or 0.0), 6),
        }
    )
    return best, info


def replay_handoff_source(st: PathState) -> str:
    if st.misses > 0 or st.last_candidate is None:
        return "track_only"
    return st.last_candidate.source or "track_only"


def replay_handoff_candidate_json(st: PathState) -> dict:
    if st.misses == 0 and st.last_candidate is not None:
        return st.last_candidate.to_json()
    for cand in reversed(st.candidate_history):
        if isinstance(cand, dict):
            return cand
    return {}


def replay_handoff_metric(st: PathState, features: dict[str, float], key: str) -> float:
    cand = replay_handoff_candidate_json(st)
    if key == "line_context":
        return max(float(cand.get("line_context", 0.0) or 0.0), float(features.get("mean_line_context", 0.0)))
    if key == "attached_support":
        return max(
            float(cand.get("attached_support", 0.0) or 0.0),
            float(features.get("mean_attached_support", 0.0)),
        )
    if key == "map_score":
        return max(
            float(cand.get("map_score", 0.0) or 0.0),
            float(cand.get("recenter_peak_score", 0.0) or 0.0),
            float(cand.get("recenter_response_score", 0.0) or 0.0),
            float(features.get("mean_map_score", 0.0)),
        )
    if key == "recenter_shift":
        return float(cand.get("recenter_shift_det", 0.0) or 0.0)
    return 0.0


def replay_handoff_key(st: PathState) -> str:
    cand = replay_handoff_candidate_json(st)
    seed = str(cand.get("recenter_seed_track_id", "")).strip()
    if seed:
        return f"seed:{seed}"
    return f"track:{st.sid}"


def replay_handoff_target_local_confirmed(st: PathState, features: dict[str, float]) -> bool:
    return (
        replay_handoff_source(st) == "target_local_recovery"
        and st.hit_count() >= 2
        and float(features.get("positive_pair_rate", 0.0)) >= 0.25
    )


class ReplayHandoffSelector:
    """Bounded source-scoped selected-output handoff over existing TBD states."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.recent: deque[tuple[int, str]] = deque(maxlen=max(1, int(args.replay_handoff_window)))
        self.last_info: dict[str, object] = {"enabled": bool(args.replay_handoff_select), "used": False}

    def state_reject_reasons(self, st: PathState, rank: int, features: dict[str, float]) -> list[str]:
        args = self.args
        reasons: list[str] = []
        source = replay_handoff_source(st)
        allowed_sources = _parse_source_set(args.replay_handoff_sources)
        if rank > args.replay_handoff_max_rank:
            reasons.append("rank")
        if source not in allowed_sources:
            if not (
                getattr(args, "replay_handoff_allow_target_local_recovery", False)
                and replay_handoff_target_local_confirmed(st, features)
            ):
                reasons.append("source")
        if source == "target_local_recovery" and not replay_handoff_target_local_confirmed(st, features):
            reasons.append("target_local_recovery_unconfirmed")
        if st.hit_count() < args.replay_handoff_min_hits:
            reasons.append("hits")
        if st.misses > args.replay_handoff_max_misses:
            reasons.append("misses")
        _x, _y, bw, bh = st.bbox
        side = max(float(bw), float(bh))
        if side < args.replay_handoff_min_side or side > args.replay_handoff_max_side:
            reasons.append("side")
        line = replay_handoff_metric(st, features, "line_context")
        support = replay_handoff_metric(st, features, "attached_support")
        if line > args.replay_handoff_max_line_context:
            reasons.append("line_context")
        if support > args.replay_handoff_max_attached_support:
            reasons.append("attached_support")
        if source == "candidate_local_recenter":
            if replay_handoff_metric(st, features, "map_score") < args.replay_handoff_min_map_score:
                reasons.append("map_score")
            if replay_handoff_metric(st, features, "recenter_shift") > args.replay_handoff_max_shift_det:
                reasons.append("recenter_shift")
        return reasons

    def state_score(self, st: PathState, rank: int, features: dict[str, float]) -> float:
        line = replay_handoff_metric(st, features, "line_context")
        support = replay_handoff_metric(st, features, "attached_support")
        shift = replay_handoff_metric(st, features, "recenter_shift")
        map_score = replay_handoff_metric(st, features, "map_score")
        return float(
            map_score
            + 0.05 * min(10, st.hit_count())
            + (0.20 if replay_handoff_key(st).startswith("seed:") else 0.0)
            - 0.012 * max(0, rank - 1)
            - 0.80 * line
            - 0.04 * support
            - 0.03 * shift
        )

    def choose(
        self,
        frame_no: int,
        selected: PathState | None,
        states: list[PathState],
        tbd: BeamTBD,
    ) -> tuple[PathState | None, dict[str, object]]:
        info: dict[str, object] = {"enabled": bool(self.args.replay_handoff_select), "used": False}
        self.last_info = info
        if not self.args.replay_handoff_select:
            return selected, info

        ranked = sorted(states, key=lambda st: tbd.verified_score(st), reverse=True)
        eligible: list[tuple[float, int, PathState, dict[str, float]]] = []
        reject_counts: Counter[str] = Counter()
        for rank, st in enumerate(ranked, start=1):
            features = tube_features(st)
            reasons = self.state_reject_reasons(st, rank, features)
            if reasons:
                reject_counts.update(reasons)
                continue
            eligible.append((self.state_score(st, rank, features), rank, st, features))

        info["eligible_count"] = len(eligible)
        info["reject_counts"] = dict(sorted(reject_counts.items()))
        if not eligible:
            info["reason"] = "no_eligible"
            return selected, info

        score, rank, chosen, features = max(eligible, key=lambda item: item[0])
        key = replay_handoff_key(chosen)
        self.recent.append((frame_no, key))
        recent_window = max(1, int(self.args.replay_handoff_window))
        recent_hits = sum(1 for f, k in self.recent if frame_no - f < recent_window and k == key)
        info.update(
            {
                "reason": "candidate",
                "candidate_rank": rank,
                "candidate_track_id": chosen.sid,
                "candidate_source": replay_handoff_source(chosen),
                "candidate_key": key,
                "candidate_score": round(float(score), 3),
                "candidate_hits": chosen.hit_count(),
                "candidate_misses": chosen.misses,
                "line_context": round(replay_handoff_metric(chosen, features, "line_context"), 3),
                "attached_support": round(replay_handoff_metric(chosen, features, "attached_support"), 3),
                "map_score": round(replay_handoff_metric(chosen, features, "map_score"), 3),
                "recenter_shift": round(replay_handoff_metric(chosen, features, "recenter_shift"), 3),
                "continuity_hits": recent_hits,
            }
        )
        if not self.args.replay_handoff_diagnostic_same_frame and recent_hits < self.args.replay_handoff_promote_hits:
            info["reason"] = "continuity_wait"
            return selected, info

        info["used"] = True
        info["reason"] = "replay_handoff_select"
        return chosen, info


class DelayedSequenceSelector:
    """Small delayed Viterbi selector over recent detector states.

    This mirrors the Raspberry Pi export replay selector, but runs inside the
    detector over a bounded top-N/window so it can be used as a live delayed
    output mode without sklearn or full-video dynamic programming.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.core = StreamingViterbiSelector(
            max_jump_px=args.delayed_sequence_max_jump_px,
            transition_weight=args.delayed_sequence_transition_weight,
        )
        self.active = False
        self.pending_hits = 0
        self.pending_frame: int | None = None
        self.pending_bbox: tuple[int, int, int, int] | None = None
        self.last_frame: int | None = None
        self.last_bbox: tuple[int, int, int, int] | None = None
        self.lost = 0

    def _ranked_states(self, states: list[PathState], tbd: BeamTBD) -> list[tuple[float, PathState]]:
        verified_ranked = sorted(
            [(float(tbd.verified_score(st)), st) for st in states if state_is_sequence_candidate(st, tbd, self.args)],
            key=lambda item: item[0],
            reverse=True,
        )[: max(1, self.args.delayed_sequence_top_n)]
        if self.args.delayed_sequence_score_source != "surface_ranker" or tbd.surface_ranker is None:
            return verified_ranked

        rows: list[dict[str, object]] = []
        ranked_states: list[PathState] = []
        for rank, (baseline_score, st) in enumerate(verified_ranked, start=1):
            features = tube_features(st)
            if not surface_ranker_applies(features, self.args):
                continue
            tube_score = scoped_tube_verifier_score(features, self.args)
            next_score = verified_ranked[rank][0] if rank < len(verified_ranked) else None
            margin = (baseline_score - next_score) if next_score is not None else None
            row = state_feature_row(st, self.args, rank, baseline_score, tube_score, margin)
            if not surface_ranker_gate_allows(row, self.args.surface_ranker_gate):
                continue
            rows.append(row)
            ranked_states.append(st)
        if not rows:
            return verified_ranked
        scores = tbd.surface_ranker.scores(rows)
        if len(scores) == 0:
            return verified_ranked
        return sorted(
            [(float(score), st) for score, st in zip(scores, ranked_states)],
            key=lambda item: item[0],
            reverse=True,
        )[: max(1, self.args.delayed_sequence_top_n)]

    def add_frame(self, frame_no: int, states: list[PathState], tbd: BeamTBD) -> None:
        ranked = self._ranked_states(states, tbd)
        self.core.add_layer(
            frame_no,
            [
                SequenceItem(
                    frame=frame_no,
                    bbox=(float(st.bbox[0]), float(st.bbox[1]), float(st.bbox[2]), float(st.bbox[3])),
                    score=float(score),
                    payload=st,
                )
                for score, st in ranked
            ],
        )

    def ready(self) -> bool:
        return self.core.ready(self.args.delayed_sequence_window)

    def pop_ready(self) -> tuple[int, PathState | None]:
        if not self.core.layers:
            raise RuntimeError("no delayed sequence layers")
        frame_no, selected, selected_score, path_indices = self._select_path()
        self.core.pop_first()
        if selected is not None and path_indices and self.args.delayed_sequence_commit_prefix:
            self._commit_remaining_path(path_indices[1:])
        selected = self._apply_hysteresis(frame_no, selected, selected_score)
        return frame_no, selected

    def flush(self) -> list[tuple[int, PathState | None]]:
        out: list[tuple[int, PathState | None]] = []
        while self.core.layers:
            out.append(self.pop_ready())
        return out

    def _hysteresis_enabled(self) -> bool:
        return self.args.delayed_sequence_acquire_threshold is not None

    def _apply_hysteresis(self, frame_no: int, selected: PathState | None, score: float | None) -> PathState | None:
        if not self._hysteresis_enabled():
            return selected
        if selected is None or score is None:
            self.pending_hits = 0
            self.pending_frame = None
            self.pending_bbox = None
            if self.active:
                self.lost += 1
                if self.lost > max(0, self.args.delayed_sequence_lost_patience):
                    self.active = False
                    self.last_frame = None
                    self.last_bbox = None
            return None

        acquire_threshold = float(self.args.delayed_sequence_acquire_threshold)
        keep_threshold = (
            float(self.args.delayed_sequence_keep_threshold)
            if self.args.delayed_sequence_keep_threshold is not None
            else float(self.args.delayed_sequence_threshold)
        )
        if not self.active:
            if score < acquire_threshold:
                self.pending_hits = 0
                self.pending_frame = None
                self.pending_bbox = None
                return None
            gap = max(1, frame_no - self.pending_frame) if self.pending_frame is not None else 1
            jump_ok = (
                self.pending_bbox is None
                or center_distance(self.pending_bbox, selected.bbox)
                <= self.args.delayed_sequence_max_jump_px * gap
            )
            self.pending_hits = self.pending_hits + 1 if jump_ok else 1
            self.pending_frame = frame_no
            self.pending_bbox = selected.bbox
            if self.pending_hits < max(1, int(self.args.delayed_sequence_acquire_hits)):
                return None
            self.active = True
            self.lost = 0
            self.last_frame = frame_no
            self.last_bbox = selected.bbox
            return selected

        gap = max(1, frame_no - self.last_frame) if self.last_frame is not None else 1
        jump_ok = (
            self.last_bbox is None
            or center_distance(self.last_bbox, selected.bbox)
            <= self.args.delayed_sequence_max_jump_px * gap
        )
        if score >= keep_threshold and jump_ok:
            self.lost = 0
            self.last_frame = frame_no
            self.last_bbox = selected.bbox
            return selected

        self.lost += 1
        if self.lost > max(0, self.args.delayed_sequence_lost_patience):
            self.active = False
            self.pending_hits = 0
            self.pending_frame = None
            self.pending_bbox = None
            self.last_frame = None
            self.last_bbox = None
        return None

    def _select_path(self) -> tuple[int, PathState | None, float | None, list[int]]:
        frame_no, item, score, path_indices = self.core.first_selection()
        first_frame = int(frame_no) if frame_no is not None else 0
        if item is None or score is None:
            return first_frame, None, None, []
        st = item.payload
        if not isinstance(st, PathState):
            return first_frame, None, None, []
        if score < self.args.delayed_sequence_threshold:
            return first_frame, None, float(score), path_indices
        return first_frame, st, float(score), path_indices

    def _commit_remaining_path(self, path_indices: list[int]) -> None:
        self.core.commit_prefix(path_indices)


def draw_overlay(
    frame: np.ndarray,
    frame_no: int,
    model_name: str,
    inlier_ratio: float,
    threshold: float,
    cands: list[base.Candidate],
    states: list[PathState],
    selected: PathState | None,
    args: argparse.Namespace,
) -> np.ndarray:
    ov = frame.copy()
    if args.draw_debug:
        for cand in cands[: args.top_k_debug]:
            x, y, w, h = cand.bbox
            cv2.rectangle(ov, (x, y), (x + w, y + h), (80, 170, 80), 1)
        for st in states[:20]:
            if st.misses:
                continue
            x, y, w, h = st.bbox
            cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 180, 255), 1)
    if selected is not None:
        x, y, w, h = selected.bbox
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            ov,
            f"TBD T{selected.sid} {selected.score():.2f}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    text = f"f{frame_no:05d} {model_name} in={inlier_ratio:.2f} thr={threshold:.1f} cand={len(cands)} states={len(states)}"
    cv2.putText(ov, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ov, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return ov


def tube_state_payload(
    frame_no: int,
    rank: int,
    st: PathState,
    tbd: BeamTBD,
    args: argparse.Namespace,
    selected: PathState | None,
    competitor_margin: float | None = None,
) -> tuple[dict, dict]:
    features = tube_features(st)
    tube_score = scoped_tube_verifier_score(features, args)
    verified = tbd.verified_score(st)
    raw = st.score()
    eligible = st.misses <= args.max_selected_misses and st.hit_count() >= args.min_path_hits
    passes_floor = (
        raw >= args.selected_score
        if args.tube_verifier == "off"
        else tube_score >= args.tube_verifier_floor and verified >= args.selected_score
    )
    is_selected = (
        selected is not None
        and st.sid == selected.sid
        and st.bbox == selected.bbox
        and st.last_frame == selected.last_frame
    )
    cand_is_current = st.misses == 0 and st.last_candidate is not None
    cand_json = st.last_candidate.to_json() if cand_is_current else None
    payload = {
        "rank": rank,
        "track_id": st.sid,
        "bbox": list(st.bbox),
        "score": round(raw, 3),
        "verified_score": round(verified, 3),
        "tube_verifier_score": round(tube_score, 3),
        "eligible": eligible,
        "passes_floor": passes_floor,
        "selected": is_selected,
        "hits": st.hit_count(),
        "misses": st.misses,
        "vx": round(st.vx, 3),
        "vy": round(st.vy, 3),
        "competitor_margin": round(float(competitor_margin), 3) if competitor_margin is not None else None,
        "candidate_is_current": cand_is_current,
        "candidate_frame": frame_no if cand_is_current else None,
        "candidate": cand_json,
        "tube_features": {k: round(float(v), 3) for k, v in features.items()},
    }
    row = {
        "frame": frame_no,
        "rank": rank,
        "track_id": st.sid,
        "x": st.bbox[0],
        "y": st.bbox[1],
        "w": st.bbox[2],
        "h": st.bbox[3],
        "score": round(raw, 6),
        "verified_score": round(verified, 6),
        "tube_verifier_score": round(tube_score, 6),
        "eligible": int(eligible),
        "passes_floor": int(passes_floor),
        "selected": int(is_selected),
        "hits": st.hit_count(),
        "misses": st.misses,
        "vx": round(st.vx, 6),
        "vy": round(st.vy, 6),
        "competitor_margin": round(float(competitor_margin), 6) if competitor_margin is not None else "",
        "cand_is_current": int(cand_is_current),
        "cand_frame": frame_no if cand_is_current else "",
    }
    if cand_json is not None:
        for key, value in cand_json.items():
            if isinstance(value, (int, float, str)):
                row[f"cand_{key}"] = value
            elif isinstance(value, (list, tuple)):
                row[f"cand_{key}"] = json.dumps(value)
    for key, value in features.items():
        row[f"tube_{key}"] = round(float(value), 6)
    return payload, row


def state_is_sequence_candidate(st: PathState, tbd: BeamTBD, args: argparse.Namespace) -> bool:
    if st.misses > args.max_selected_misses or st.hit_count() < args.delayed_sequence_min_hits:
        return False
    if not getattr(args, "delayed_sequence_require_floor", False):
        return True
    raw = st.score()
    if args.tube_verifier == "off":
        return raw >= args.selected_score
    features = tube_features(st)
    tube_score = scoped_tube_verifier_score(features, args)
    return tube_score >= args.tube_verifier_floor and tbd.verified_score(st) >= args.selected_score


def selected_state_json(
    frame_no: int,
    selected: PathState,
    tbd: BeamTBD,
    args: argparse.Namespace,
) -> dict:
    selected_tube_features = tube_features(selected)
    selected_tube_score = scoped_tube_verifier_score(selected_tube_features, args)
    selected_cand_is_current = selected.misses == 0 and selected.last_candidate is not None
    return {
        "track_id": selected.sid,
        "bbox": list(selected.bbox),
        "source": "tbd",
        "score": round(selected.score(), 3),
        "verified_score": round(tbd.verified_score(selected), 3),
        "tube_verifier_score": round(selected_tube_score, 3),
        "surface_ranker_used": tbd.last_surface_ranker_used,
        "surface_ranker_score": None
        if tbd.last_surface_ranker_score is None
        else round(tbd.last_surface_ranker_score, 6),
        "surface_ranker_rows": tbd.last_surface_ranker_rows,
        "tube_features": {k: round(float(v), 3) for k, v in selected_tube_features.items()},
        "hits": selected.hit_count(),
        "misses": selected.misses,
        "vx": round(selected.vx, 3),
        "vy": round(selected.vy, 3),
        "candidate_is_current": selected_cand_is_current,
        "candidate_frame": frame_no if selected_cand_is_current else None,
        "candidate": selected.last_candidate.to_json() if selected_cand_is_current else None,
    }


def append_selected_output(
    frame_no: int,
    selected: PathState | None,
    tbd: BeamTBD,
    args: argparse.Namespace,
    selected_rows: list[list],
    selected_feature_rows: list[dict],
    selected_jsonl_handle=None,
    emitted_at_frame: int | None = None,
    selected_source: str = "tbd",
) -> int:
    if selected is None:
        return 0
    if not args.stream_only:
        selected_rows.append([frame_no, selected.sid, *selected.bbox, selected.score(), selected.misses])
        _payload, row = tube_state_payload(frame_no, 1, selected, tbd, args, selected)
        row["selected_source"] = selected_source
        selected_feature_rows.append(row)
    if selected_jsonl_handle is not None:
        record = {
            "frame": frame_no,
            "emitted_at_frame": emitted_at_frame if emitted_at_frame is not None else frame_no,
            "source": selected_source,
            "track_id": selected.sid,
            "bbox": list(selected.bbox),
            "score": round(float(selected.score()), 6),
            "verified_score": round(float(tbd.verified_score(selected)), 6),
            "misses": selected.misses,
            "hits": selected.hit_count(),
        }
        selected_jsonl_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        selected_jsonl_handle.flush()
    return 1


def write_telemetry_output(
    handle,
    emitted_at_frame: int,
    selected_frame: int,
    selected: PathState | None,
    tbd: BeamTBD,
    source: str,
    status: str,
    process_ms: float,
    wall_ms: float,
) -> None:
    if handle is None:
        return
    record = {
        "emitted_at_frame": emitted_at_frame,
        "selected_frame": selected_frame,
        "source": source,
        "status": status,
        "process_ms": round(float(process_ms), 3),
        "wall_ms": round(float(wall_ms), 3),
        "selected": selected is not None,
    }
    if selected is not None:
        record.update(
            {
                "track_id": selected.sid,
                "bbox": list(selected.bbox),
                "score": round(float(selected.score()), 6),
                "verified_score": round(float(tbd.verified_score(selected)), 6),
                "misses": selected.misses,
                "hits": selected.hit_count(),
            }
        )
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    handle.flush()


def video_capture_source(source: str) -> str | int:
    raw = str(source).strip()
    if raw.startswith("camera:"):
        device = raw.split(":", 1)[1].strip()
        if not device.isdigit():
            raise ValueError(f"camera source must be camera:<integer>, got {source!r}")
        return int(device)
    if raw.isdigit():
        return int(raw)
    return source


class RunningWindowStats:
    def __init__(self, window: int = 4096):
        self.count = 0
        self.total = 0.0
        self.max_value = 0.0
        self.values: deque[float] = deque(maxlen=max(1, int(window)))

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        v = float(value)
        self.total += v
        self.max_value = max(self.max_value, v)
        self.values.append(v)

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, q: float) -> float:
        if not self.values:
            return 0.0
        return float(np.percentile(np.asarray(self.values, dtype=np.float32), q))

    def max(self) -> float:
        return self.max_value if self.count else 0.0


def run(args: argparse.Namespace) -> None:
    if args.surface_ranker_policy != "off" and not args.surface_ranker_model:
        raise SystemExit("--surface_ranker_model is required when --surface_ranker_policy is not off")
    if args.delayed_sequence_score_source == "surface_ranker" and (
        args.surface_ranker_policy == "off" or not args.surface_ranker_model
    ):
        raise SystemExit(
            "--delayed_sequence_score_source surface_ranker requires --surface_ranker_policy and --surface_ranker_model"
        )
    if (
        args.surface_ranker_policy != "off"
        and args.surface_ranker_scope in {"surface_backed", "surface_context"}
        and not router_is_active(args)
    ):
        raise SystemExit("--surface_ranker_scope surface_backed/surface_context requires an active candidate router/runtime mode")

    cap = cv2.VideoCapture(video_capture_source(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, prev = cap.read()
    if not ok:
        raise SystemExit("no frames")
    prev_full = prev.copy()
    if args.downscale != 1.0:
        prev = cv2.resize(prev, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
    prev_g = base.ensure_gray(prev)
    h_img, w_img = prev_g.shape[:2]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = fps_src if fps_src > 1 else 30.0
        video_writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps_out, (w_img, h_img))

    px_per_frame = base.kinematic_px_per_frame(w_img, fps_src, args)
    tbd = BeamTBD(args, px_per_frame)
    delayed_selector = DelayedSequenceSelector(args) if args.delayed_sequence_select else None
    replay_handoff_selector = ReplayHandoffSelector(args) if args.replay_handoff_select else None
    selected_jsonl_handle = None
    if args.selected_jsonl:
        selected_jsonl_path = Path(args.selected_jsonl)
        selected_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        selected_jsonl_handle = selected_jsonl_path.open("w")
    telemetry_jsonl_handle = None
    if args.telemetry_jsonl:
        telemetry_jsonl_path = Path(args.telemetry_jsonl)
        telemetry_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_jsonl_handle = telemetry_jsonl_path.open("w")
    prev_mask: np.ndarray | None = None
    report: list[dict] = []
    process_stats = RunningWindowStats(args.stats_window)
    wall_stats = RunningWindowStats(args.stats_window)
    inlier_stats = RunningWindowStats(args.stats_window)
    candidate_stats = RunningWindowStats(args.stats_window)
    timing_stats: dict[str, RunningWindowStats] = {}
    processed_count = 0
    selected_frames = 0
    noisy_frames = 0
    multi_candidate_frames = 0
    selected_rows: list[list] = []
    selected_output_count = 0
    selected_feature_rows: list[dict] = []
    top_tube_rows: list[dict] = []
    model_counts: dict[str, int] = {}
    temporal_history: list[TemporalStackFrame] = []
    temporal_max_age = max(abs(v) for v in parse_int_offsets(args.temporal_stack_offsets))
    fno = 1
    frame_mode_counts: dict[str, int] = {}
    candidate_router_counts_total: dict[str, int] = {}
    motion_model_mask_boxes: list[tuple[int, int, int, int]] = []
    target_local_recovery_seed: TargetLocalRecoverySeed | None = None
    target_local_anchor_bank = RuntimeAnchorBank(args)
    teacher_scale = args.srps_teacher_coord_scale if args.srps_teacher_coord_scale > 0 else args.downscale
    srps_teacher_residuals = load_srps_teacher_residuals(args.srps_teacher_residual_proposals_csv, teacher_scale)
    srps_teacher_sources = load_srps_teacher_sources(
        args.srps_teacher_candidate_csv,
        args.srps_teacher_candidate_coord_scale,
        _parse_source_set(args.srps_source_candidates),
    )
    srps_source = StabilizedResidualPathSource(
        SRPSConfig(
            residual_top=args.srps_residual_top,
            snap_radius=args.srps_snap_radius,
            follow_gate=args.srps_gate,
            seed_gate=args.srps_gate,
            seed_window=args.srps_seed_window,
            seed_required_hits=args.srps_seed_required_hits,
            max_misses_after_confirm=args.srps_max_misses,
            max_confirmed_age=args.srps_max_confirmed_age if args.srps_disable_stale_confirmed_paths else 10**9,
            max_emit_per_frame=args.srps_max_emit_per_frame,
            seed_score_weight=args.srps_seed_score_weight,
            seed_rank_penalty=args.srps_seed_rank_penalty,
            seed_beta=args.srps_seed_beta,
            follow_beta=args.srps_follow_beta,
            default_box_size=args.srps_residual_box_det_px,
            multi_seed_compete=args.srps_multi_seed_compete,
            confirm_requires_verified_seed=args.srps_confirm_requires_verified_seed,
            reconfirm_on_recovery_epoch=args.srps_reconfirm_on_recovery_epoch,
            false_path_veto=args.srps_false_path_veto,
            verified_seed_score=args.srps_verified_seed_score,
            verified_replacement_margin=args.srps_verified_replacement_margin,
            residual_density_radius=args.srps_residual_density_radius,
            residual_density_threshold=args.srps_residual_density_threshold,
            disable_global_residual_seed=args.srps_disable_global_residual_seed,
            do_not_hijack_stable_track=args.srps_do_not_hijack_stable_track,
            coord_space="detector",
        )
    )
    srps_was_active = False
    srps_runtime_residual_rows: list[dict] = []
    srps_seed_rows: list[dict] = []
    srps_path_rows: list[dict] = []
    target_local_state_select_count = 0
    srps_verified_select_count = 0
    replay_handoff_select_count = 0
    pending_track_only_replay_seeds: list[tuple[int, list[base.Candidate]]] = []

    while True:
        if args.max_frames is not None and fno >= args.max_frames:
            break
        wall_start = time.perf_counter()
        ok, cur = cap.read()
        t_read = time.perf_counter()
        if not ok:
            break
        frame_start = t_read
        cur_full = cur
        if args.downscale != 1.0:
            cur = cv2.resize(cur, None, fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
        cur_g = base.ensure_gray(cur)
        t_preprocess = time.perf_counter()
        frame_decision = (
            frame_router_decision(cur_g, args)
            if router_is_active(args)
            else FrameRouterDecision("baseline", True, args.top_k_candidates, 0.0, {})
        )
        t_frame_router = time.perf_counter()

        feature_mask = base.make_feature_mask(
            prev_g.shape[:2],
            motion_model_mask_boxes if args.mask_selected_for_motion_model else [],
        )
        g0, g1 = base.lk_tracks(prev_g, cur_g, feature_mask, args)
        if g0 is None:
            if not args.motion_model_fallback_identity:
                prev_g = cur_g
                prev_full = cur_full.copy()
                temporal_history = []
                motion_model_mask_boxes = []
                fno += 1
                continue
            g0 = np.empty((0, 2), dtype=np.float32)
            g1 = np.empty((0, 2), dtype=np.float32)
            chosen = {
                "name": "identity_fallback",
                "h": np.eye(3, dtype=np.float32),
                "inlier_ratio": 0.0,
                "median_feature_error": 0.0,
            }
        else:
            chosen = base.choose_model(prev_g, cur_g, g0, g1, args)
        if chosen is None:
            if not args.motion_model_fallback_identity:
                prev_g = cur_g
                prev_full = cur_full.copy()
                temporal_history = []
                motion_model_mask_boxes = []
                fno += 1
                continue
            chosen = {
                "name": "identity_fallback",
                "h": np.eye(3, dtype=np.float32),
                "inlier_ratio": 0.0,
                "median_feature_error": 0.0,
            }
        model_counts[chosen["name"]] = model_counts.get(chosen["name"], 0) + 1
        t_motion_model = time.perf_counter()
        temporal_history = update_temporal_stack_history(
            temporal_history,
            base.ensure_gray(prev_full),
            chosen["h"],
            args.downscale,
            temporal_max_age,
        )

        warped = base.warp_prev(prev_g, chosen["h"], w_img, h_img)
        signed_diff = cur_g.astype(np.float32) - warped.astype(np.float32)
        signed_sigma = robust_sigma(signed_diff)
        residual = cv2.absdiff(warped, cur_g)
        residual_blur = cv2.GaussianBlur(residual, (3, 3), 0)
        threshold = base.robust_threshold(residual_blur, args.threshold_sigma, args.threshold_percentile, args.min_threshold)
        mask = (residual_blur >= threshold).astype(np.uint8) * 255
        app_resp, app_mask, app_threshold = base.appearance_response(cur_g, args)

        border = int(round(args.border_frac * min(w_img, h_img)))
        if border:
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0

        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

        if prev_mask is not None:
            dilate_k = 2 * args.temporal_radius + 1
            prev_dilated = cv2.dilate(prev_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
            confirmed_mask = cv2.bitwise_and(mask, prev_dilated)
            motion_mask = cv2.bitwise_or(confirmed_mask, cv2.erode(mask, k3))
        else:
            motion_mask = mask

        t_residual = time.perf_counter()
        motion_cands = base.extract_candidates("motion", motion_mask, residual_blur, app_resp, cur_g, args)
        app_cands: list[base.Candidate] = []
        if args.appearance != "off":
            app_cands = base.extract_candidates("appearance", app_mask, residual_blur, app_resp, cur_g, args)
        map_cands = map_peak_candidates(cur_g, residual_blur, app_resp, args)
        native_cands = native_micro_candidates(
            cur_full,
            args.downscale,
            residual_blur,
            app_resp,
            cur_g,
            args,
        )
        large_dark_cands = large_dark_candidates(
            cur_full,
            args.downscale,
            residual_blur,
            app_resp,
            cur_g,
            args,
        )
        t_cheap_proposals = time.perf_counter()
        cheap_cands = motion_cands + app_cands + map_cands + native_cands + large_dark_cands
        stack_cands: list[base.Candidate] = []
        hybrid_coast_cands: list[base.Candidate] = []
        target_local_recovery_cands: list[base.Candidate] = []
        target_local_anchor_bank_cands: list[base.Candidate] = []
        srps_cands: list[base.Candidate] = []
        target_local_anchor_bank_info: dict[str, object] = {
            "enabled": bool(args.target_local_anchor_bank_proposals),
            "used": False,
        }
        target_local_anchor_bank_ms = 0.0
        candidate_local_recenter_cands: list[base.Candidate] = []
        track_only_replay_seed_cands: list[base.Candidate] = []
        routed_cheap: list[base.Candidate] = []
        cheap_limit = (
            max(args.top_k_candidates, int(round(args.top_k_candidates * args.scenario_pool_factor)))
            if args.scenario_balance
            else args.top_k_candidates
        )
        surface_extras_allowed = not runtime_budget_applies(args) or args.runtime_mode == "surface"
        surface_branch_reason = (
            "explicit_surface_or_legacy"
            if surface_extras_allowed
            else "disabled_by_frame_mode"
        )

        if router_applies(args):
            if args.native_roi_score:
                assign_native_roi_scores(cheap_cands, cur_full, args.downscale)
            routed_cheap = dedupe_candidates(
                cheap_cands,
                cheap_limit,
                (lambda cand: candidate_obs(cand, args)) if candidate_obs_sort_needed(args, True) else None,
            )
            assign_attached_support(routed_cheap, cur_g)
            assign_candidate_router_states(routed_cheap, cur_g)
            surface_extras_allowed = surface_branch_needed(routed_cheap, tbd.states, frame_decision, args)
            surface_branch_reason = "candidate_local_or_track" if surface_extras_allowed else "not_candidate_local_surface"

        if surface_extras_allowed:
            if args.temporal_stack_candidate_local:
                seed_cands = routed_cheap if routed_cheap else dedupe_candidates(cheap_cands, cheap_limit, None)
                stack_cands = temporal_stack_candidate_local_candidates(
                    cur_full,
                    args.downscale,
                    temporal_history,
                    seed_cands,
                    residual_blur,
                    app_resp,
                    cur_g,
                    args,
                )
            else:
                stack_cands = temporal_stack_candidates(
                    cur_full,
                    args.downscale,
                    temporal_history,
                    residual_blur,
                    app_resp,
                    cur_g,
                    args,
                )
        if args.hybrid_coast_proposals and (surface_extras_allowed or runtime_budget_applies(args)):
            hybrid_coast_cands = hybrid_coast_candidates(
                tbd.states,
                tbd,
                chosen["h"],
                w_img,
                h_img,
                cur_full,
                args.downscale,
                residual_blur,
                app_resp,
                cur_g,
                args,
            )
        if args.candidate_local_recenter_track_only_replay:
            still_pending: list[tuple[int, list[base.Candidate]]] = []
            max_age = max(0, int(getattr(args, "candidate_local_recenter_track_only_replay_max_age", 4)))
            for target_frame, seeds in pending_track_only_replay_seeds:
                if target_frame == fno:
                    track_only_replay_seed_cands.extend(seeds)
                elif target_frame > fno:
                    still_pending.append((target_frame, seeds))
                elif fno - target_frame <= max_age:
                    track_only_replay_seed_cands.extend(seeds)
            pending_track_only_replay_seeds = still_pending
        if args.candidate_local_recenter_proposals:
            recenter_seed_cands = routed_cheap if routed_cheap else dedupe_candidates(cheap_cands, cheap_limit, None)
            raw_seed_cands = raw_low_rank_recenter_seed_candidates(cheap_cands, args)
            state_seed_cands = path_state_recenter_seed_candidates(
                tbd.states,
                tbd,
                fno,
                w_img,
                h_img,
                args,
            )
            if raw_seed_cands:
                recenter_seed_cands = list(recenter_seed_cands) + raw_seed_cands
            if state_seed_cands:
                recenter_seed_cands = list(recenter_seed_cands) + state_seed_cands
            if track_only_replay_seed_cands:
                recenter_seed_cands = list(recenter_seed_cands) + track_only_replay_seed_cands
            candidate_local_recenter_cands = candidate_local_recenter_candidates(
                recenter_seed_cands,
                cur_full,
                args.downscale,
                residual_blur,
                app_resp,
                cur_g,
                args,
            )
        if args.target_local_recovery_proposals:
            target_local_recovery_cands = target_local_recovery_candidates(
                target_local_recovery_seed,
                fno,
                w_img,
                h_img,
                cur_full,
                args.downscale,
                residual_blur,
                app_resp,
                cur_g,
                args,
            )
        if args.target_local_anchor_bank_proposals:
            anchor_start = time.perf_counter()
            target_local_anchor_bank.update_from_candidates(
                fno,
                candidate_local_recenter_cands + target_local_recovery_cands,
            )
            target_local_anchor_bank_cands, target_local_anchor_bank_info = target_local_anchor_bank_candidates(
                target_local_anchor_bank,
                fno,
                w_img,
                h_img,
                cur_full,
                args.downscale,
                residual_blur,
                app_resp,
                cur_g,
                args,
            )
            target_local_anchor_bank_ms = (time.perf_counter() - anchor_start) * 1000.0
        srps_active = srps_should_run(args, frame_decision, surface_extras_allowed)
        if srps_active:
            srps_seed_pool = cheap_cands + candidate_local_recenter_cands + target_local_recovery_cands
            if srps_teacher_residuals:
                srps_residuals = srps_teacher_residuals.get(fno, [])
            else:
                srps_residual_map = residual_blur
                if args.srps_residual_source in {"temporal_combo", "temporal_dark_fullres"} and cur_full is not None:
                    temp_map_full = temporal_stack_residual_map(
                        base.ensure_gray(cur_full),
                        temporal_history,
                        parse_int_offsets(args.temporal_stack_offsets),
                        max(2, int(args.temporal_stack_min_frames)),
                        args.downscale,
                        args,
                    )
                    if args.srps_residual_source == "temporal_dark_fullres" and temp_map_full is not None:
                        srps_residuals = srps_residual_candidates_from_full_map(
                            fno,
                            temp_map_full,
                            args.downscale,
                            args,
                        )
                    elif temp_map_full is not None:
                        srps_residual_map = cv2.resize(
                            temp_map_full,
                            (w_img, h_img),
                            interpolation=cv2.INTER_AREA if args.downscale < 1.0 else cv2.INTER_LINEAR,
                        )
                        srps_residuals = srps_residual_candidates_from_map(fno, srps_residual_map, args)
                    else:
                        srps_residuals = srps_residual_candidates_from_map(fno, srps_residual_map, args)
                else:
                    srps_residuals = srps_residual_candidates_from_map(fno, srps_residual_map, args)
            srps_sources = (
                srps_teacher_sources.get(fno, [])
                if srps_teacher_sources
                else srps_source_candidates_from_base(fno, srps_seed_pool, args)
            )
            if args.srps_dump_runtime_residual_proposals:
                for prop in srps_residuals[: args.srps_residual_top]:
                    srps_runtime_residual_rows.append(
                        {
                            "frame": fno,
                            "rank": prop.rank,
                            "cx": round(float(prop.cx), 3),
                            "cy": round(float(prop.cy), 3),
                            "score": round(float(prop.score), 6),
                            "source": str(
                                prop.payload.get("source", args.srps_residual_source)
                                if isinstance(prop.payload, dict)
                                else args.srps_residual_source
                            ),
                            "teacher_injected": int(bool(srps_teacher_residuals)),
                        }
                    )
            if args.srps_dump_seed_candidates:
                for src_cand in srps_sources[: args.srps_source_top_k]:
                    srps_seed_rows.append(
                        {
                            "frame": fno,
                            "source": src_cand.source,
                            "rank": src_cand.rank,
                            "cx": round(float(src_cand.cx), 3),
                            "cy": round(float(src_cand.cy), 3),
                            "score": round(float(src_cand.score), 6),
                        }
                    )
            srps_base_state = srps_base_state_from_tbd(tbd, args)
            srps_emitted = srps_source.update(
                fno,
                srps_sources,
                srps_residuals,
                SRPSBaseState(
                    state=srps_base_state.state,
                    stable_t=srps_base_state.stable_t,
                    low_confidence=srps_base_state.low_confidence,
                    router=srps_base_state.router,
                    recovery_epoch=not srps_was_active,
                ),
            )
            if args.srps_dump_path_candidates:
                srps_path_rows.extend(cand.to_row() for cand in srps_emitted)
            srps_cands = [
                srps_to_base_candidate(srps, residual_blur, app_resp, cur_g, args)
                for srps in srps_emitted[: max(1, int(args.srps_max_emit_per_frame))]
            ]
        srps_was_active = srps_active
        t_proposals = time.perf_counter()
        raw_cands = (
            cheap_cands
            + stack_cands
            + hybrid_coast_cands
            + candidate_local_recenter_cands
            + target_local_recovery_cands
            + target_local_anchor_bank_cands
            + srps_cands
        )
        if args.native_roi_score:
            assign_native_roi_scores(raw_cands, cur_full, args.downscale)
        preselect_score = (
            (lambda cand: candidate_obs(cand, args))
            if candidate_obs_sort_needed(args, router_applies(args))
            else None
        )
        if args.scenario_balance:
            pool_n = max(args.top_k_candidates, int(round(args.top_k_candidates * args.scenario_pool_factor)))
            cands = dedupe_candidates(raw_cands, pool_n, preselect_score)
        else:
            cands = dedupe_candidates(raw_cands, args.top_k_candidates, preselect_score)
        assign_attached_support(cands, cur_g, only_missing=router_applies(args))
        if args.native_roi_score:
            assign_native_roi_scores(cands, cur_full, args.downscale)
        candidate_router_counts: dict[str, int] = {}
        if router_is_active(args):
            assign_candidate_router_states(cands, cur_g, only_missing=router_applies(args))
            if not router_applies(args):
                assign_sky_context(cands, cur_g)
        else:
            assign_sky_context(cands, cur_g)
        t_context_router = time.perf_counter()
        effective_max_candidates = (
            frame_decision.max_candidates
            if runtime_budget_applies(args)
            else args.top_k_candidates
        )
        if args.scenario_balance:
            cands = scenario_balanced_candidates(cands, args, router_applies(args), effective_max_candidates)
        elif effective_max_candidates < len(cands):
            sort_key = (
                (lambda cand: candidate_obs(cand, args))
                if candidate_obs_sort_needed(args, router_applies(args))
                else (lambda cand: cand.score)
            )
            cands = sorted(cands, key=sort_key, reverse=True)[:effective_max_candidates]
        if router_is_active(args):
            candidate_router_counts = candidate_router_state_counts(cands)
            for state, count in candidate_router_counts.items():
                candidate_router_counts_total[state] = candidate_router_counts_total.get(state, 0) + count
        t_scenario = time.perf_counter()

        states = tbd.update(fno, cands, signed_diff, signed_sigma, w_img, h_img, chosen["h"], residual_blur, app_resp)
        if args.candidate_local_recenter_track_only_replay:
            replay_delay = max(1, int(getattr(args, "candidate_local_recenter_track_only_replay_delay", 1)))
            replay_seeds = track_only_replay_seed_candidates(
                states,
                tbd,
                fno,
                fno + replay_delay,
                w_img,
                h_img,
                args,
            )
            if replay_seeds:
                pending_track_only_replay_seeds.append((fno + replay_delay, replay_seeds))
        selected = tbd.raw_best_for_mask() if args.delayed_sequence_select else tbd.best()
        selected_source = "tbd"
        target_local_state_select_info: dict[str, object] = {"enabled": bool(args.target_local_state_select), "used": False}
        srps_verified_select_info: dict[str, object] = {
            "enabled": bool(args.srps_verified_candidate_priority),
            "used": False,
        }
        replay_handoff_select_info: dict[str, object] = {
            "enabled": bool(args.replay_handoff_select),
            "used": False,
        }
        if not args.delayed_sequence_select:
            selected, target_local_state_select_info = target_local_state_select_override(
                selected,
                states,
                target_local_recovery_seed,
                fno,
                w_img,
                h_img,
                tbd,
                args,
            )
            if target_local_state_select_info.get("used"):
                selected_source = "target_local_state_select"
                target_local_state_select_count += 1
            selected, srps_verified_select_info = srps_verified_state_select_override(
                selected,
                states,
                tbd,
                args,
            )
            if srps_verified_select_info.get("used"):
                selected_source = "srps_verified_select"
                srps_verified_select_count += 1
            if replay_handoff_selector is not None:
                selected, replay_handoff_select_info = replay_handoff_selector.choose(fno, selected, states, tbd)
                if replay_handoff_select_info.get("used"):
                    selected_source = "replay_handoff_select"
                    replay_handoff_select_count += 1
        t_tbd = time.perf_counter()

        selected_json = selected_state_json(fno, selected, tbd, args) if selected is not None else None
        if selected_json is not None:
            selected_json["source"] = selected_source
            selected_json["target_local_state_select"] = target_local_state_select_info
            selected_json["srps_verified_select"] = srps_verified_select_info
            selected_json["replay_handoff_select"] = replay_handoff_select_info
        telemetry_events: list[tuple[int, PathState | None, str, str]] = []
        if selected is not None and not args.delayed_sequence_select:
            selected_output_count += append_selected_output(
                fno,
                selected,
                tbd,
                args,
                selected_rows,
                selected_feature_rows,
                selected_jsonl_handle,
                emitted_at_frame=fno,
                selected_source=selected_source,
            )
        if not args.delayed_sequence_select:
            telemetry_events.append((fno, selected, selected_source, "selected" if selected is not None else "no_target"))
        feedback_selected = None if args.delayed_sequence_select or selected_source == "replay_handoff_select" else selected
        if args.target_local_recovery_proposals:
            if feedback_selected is not None:
                selected_verified = tbd.verified_score(feedback_selected)
                if (
                    feedback_selected.hit_count() >= args.target_local_recovery_min_hits
                    and selected_verified >= args.target_local_recovery_min_verified_score
                ):
                    target_local_recovery_seed = TargetLocalRecoverySeed.from_state(
                        fno,
                        feedback_selected,
                        selected_verified,
                    )
            elif (
                target_local_recovery_seed is not None
                and target_local_recovery_seed.age(fno) > args.target_local_recovery_max_seed_gap
            ):
                target_local_recovery_seed = None
        if args.target_local_anchor_bank_proposals:
            target_local_anchor_bank.confirm_selected(fno, feedback_selected)
        motion_model_mask_boxes = (
            [feedback_selected.bbox] if feedback_selected is not None and feedback_selected.misses == 0 else []
        )

        if delayed_selector is not None:
            delayed_selector.add_frame(fno, states, tbd)
            while delayed_selector.ready():
                out_frame, out_selected = delayed_selector.pop_ready()
                telemetry_events.append(
                    (
                        out_frame,
                        out_selected,
                        "delayed_sequence",
                        "selected" if out_selected is not None else "no_target",
                    )
                )
                selected_output_count += append_selected_output(
                    out_frame,
                    out_selected,
                    tbd,
                    args,
                    selected_rows,
                    selected_feature_rows,
                    selected_jsonl_handle,
                    emitted_at_frame=fno,
                    selected_source="delayed_sequence",
                )
                if args.target_local_recovery_proposals and out_selected is not None:
                    selected_verified = tbd.verified_score(out_selected)
                    if (
                        out_selected.hit_count() >= args.target_local_recovery_min_hits
                        and selected_verified >= args.target_local_recovery_min_verified_score
                    ):
                        target_local_recovery_seed = TargetLocalRecoverySeed.from_state(
                            out_frame,
                            out_selected,
                            selected_verified,
                        )
            if not telemetry_events:
                telemetry_events.append((fno, None, "delayed_sequence", "warming"))

        top_tubes_json: list[dict] = []
        if args.export_top_tubes > 0:
            ranked_scored = sorted(
                [(tbd.verified_score(st), st) for st in states],
                key=lambda item: item[0],
                reverse=True,
            )
            for rank, (score_cur, st) in enumerate(ranked_scored[: args.export_top_tubes], start=1):
                next_score = ranked_scored[rank][0] if rank < len(ranked_scored) else None
                margin = (score_cur - next_score) if next_score is not None else None
                payload, row = tube_state_payload(fno, rank, st, tbd, args, selected, margin)
                top_tubes_json.append(payload)
                if not args.stream_only:
                    top_tube_rows.append(row)

        t_export = time.perf_counter()
        timing_ms = {
            "capture_read": round((t_read - wall_start) * 1000.0, 3),
            "preprocess": round((t_preprocess - frame_start) * 1000.0, 3),
            "frame_router": round((t_frame_router - t_preprocess) * 1000.0, 3),
            "motion_model": round((t_motion_model - t_frame_router) * 1000.0, 3),
            "residual_masks": round((t_residual - t_motion_model) * 1000.0, 3),
            "cheap_proposals": round((t_cheap_proposals - t_residual) * 1000.0, 3),
            "surface_extra_proposals": round((t_proposals - t_cheap_proposals) * 1000.0, 3),
            "target_local_anchor_bank": round(target_local_anchor_bank_ms, 3),
            "context_router": round((t_context_router - t_proposals) * 1000.0, 3),
            "scenario_balance": round((t_scenario - t_context_router) * 1000.0, 3),
            "tbd_update": round((t_tbd - t_scenario) * 1000.0, 3),
            "export_selection": round((t_export - t_tbd) * 1000.0, 3),
        }
        dt_ms = (t_export - frame_start) * 1000.0
        wall_ms = (t_export - wall_start) * 1000.0
        for telemetry_frame, telemetry_selected, telemetry_source, telemetry_status in telemetry_events:
            write_telemetry_output(
                telemetry_jsonl_handle,
                emitted_at_frame=fno,
                selected_frame=telemetry_frame,
                selected=telemetry_selected,
                tbd=tbd,
                source=telemetry_source,
                status=telemetry_status,
                process_ms=dt_ms,
                wall_ms=wall_ms,
            )
        frame_mode_counts[frame_decision.mode] = frame_mode_counts.get(frame_decision.mode, 0) + 1

        frame_rec = {
            "frame": fno,
            "runtime_mode": frame_decision.mode,
            "runtime_mode_confidence": round(frame_decision.confidence, 3),
            "runtime_mode_features": {k: round(float(v), 3) for k, v in frame_decision.features.items()},
            "surface_extras_allowed": surface_extras_allowed,
            "surface_branch_reason": surface_branch_reason,
            "effective_max_candidates": effective_max_candidates,
            "model": chosen["name"],
            "n_features": int(len(g0)),
            "inlier_ratio": round(chosen["inlier_ratio"], 3),
            "median_feature_error": round(chosen["median_feature_error"], 3),
            "threshold": round(threshold, 2),
            "appearance_threshold": round(app_threshold, 2),
            "n_candidates": len(cands),
            "n_map_candidates": len(map_cands),
            "n_native_candidates": len(native_cands),
            "n_large_dark_candidates": len(large_dark_cands),
            "n_temporal_stack_candidates": len(stack_cands),
            "n_hybrid_coast_candidates": len(hybrid_coast_cands),
            "n_candidate_local_recenter_candidates": len(candidate_local_recenter_cands),
            "n_candidate_local_recenter_track_only_replay_seeds": len(track_only_replay_seed_cands),
            "n_candidate_local_recenter_track_only_replay_pending": sum(
                len(seeds) for _target_frame, seeds in pending_track_only_replay_seeds
            ),
            "n_target_local_recovery_candidates": len(target_local_recovery_cands),
            "n_target_local_anchor_bank_candidates": len(target_local_anchor_bank_cands),
            "n_srps_candidates": len(srps_cands),
            "srps_trace": srps_source.trace[-1].to_row() if srps_source.trace else None,
            "target_local_anchor_bank": {
                **target_local_anchor_bank_info,
                "events": target_local_anchor_bank.last_events,
                "active_anchors": len(target_local_anchor_bank.anchors),
                "quarantines": len(target_local_anchor_bank.quarantines),
            },
            "target_local_state_select": target_local_state_select_info,
            "srps_verified_select": srps_verified_select_info,
            "replay_handoff_select": replay_handoff_select_info,
            "candidate_router_counts": candidate_router_counts,
            "n_tracks": len(states),
            "selected": selected_json,
            "n_motion_model_mask_boxes": len(motion_model_mask_boxes),
            "kinematic_reject": None,
            "process_ms": round(dt_ms, 3),
            "wall_ms": round(wall_ms, 3),
            "timing_ms": timing_ms,
            "top_candidates": [c.to_json() for c in cands[: args.top_k_debug]],
        }
        if args.export_top_tubes > 0:
            frame_rec["top_tubes"] = top_tubes_json
        processed_count += 1
        process_stats.add(dt_ms)
        wall_stats.add(wall_ms)
        inlier_stats.add(float(chosen["inlier_ratio"]))
        candidate_stats.add(float(len(cands)))
        if selected is not None:
            selected_frames += 1
        if len(cands) > 10:
            noisy_frames += 1
        if len(cands) > 1:
            multi_candidate_frames += 1
        for key, value in timing_ms.items():
            timing_stats.setdefault(key, RunningWindowStats(args.stats_window)).add(float(value))
        if args.report_mode == "full":
            report.append(frame_rec)

        if args.save_every and fno % args.save_every == 0:
            cv2.imwrite(str(out_dir / f"residual_{fno:05d}.png"), residual_blur)
            debug_mask = motion_mask if args.appearance == "off" else cv2.bitwise_or(motion_mask, app_mask)
            cv2.imwrite(str(out_dir / f"mask_{fno:05d}.png"), debug_mask)
            ov = draw_overlay(cur, fno, chosen["name"], chosen["inlier_ratio"], threshold, cands, states, selected, args)
            cv2.imwrite(str(out_dir / f"overlay_{fno:05d}.png"), ov)
        if video_writer is not None:
            ov = draw_overlay(cur, fno, chosen["name"], chosen["inlier_ratio"], threshold, cands, states, selected, args)
            video_writer.write(ov)

        prev_mask = mask
        prev_g = cur_g
        prev_full = cur_full.copy()
        fno += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if delayed_selector is not None:
        for out_frame, out_selected in delayed_selector.flush():
            write_telemetry_output(
                telemetry_jsonl_handle,
                emitted_at_frame=fno,
                selected_frame=out_frame,
                selected=out_selected,
                tbd=tbd,
                source="delayed_sequence_flush",
                status="selected" if out_selected is not None else "no_target",
                process_ms=0.0,
                wall_ms=0.0,
            )
            selected_output_count += append_selected_output(
                out_frame,
                out_selected,
                tbd,
                args,
                selected_rows,
                selected_feature_rows,
                selected_jsonl_handle,
                selected_source="delayed_sequence_flush",
            )
    if selected_jsonl_handle is not None:
        selected_jsonl_handle.close()
    if telemetry_jsonl_handle is not None:
        telemetry_jsonl_handle.close()
    if processed_count == 0:
        raise SystemExit("no usable frame pairs")

    avg_ms = process_stats.mean()
    p90_ms = process_stats.percentile(90)
    p95_ms = process_stats.percentile(95)
    p99_ms = process_stats.percentile(99)
    max_ms = process_stats.max()
    avg_wall_ms = wall_stats.mean()
    p90_wall_ms = wall_stats.percentile(90)
    p95_wall_ms = wall_stats.percentile(95)
    p99_wall_ms = wall_stats.percentile(99)
    max_wall_ms = wall_stats.max()
    avg_inlier = inlier_stats.mean()
    avg_candidates = candidate_stats.mean()
    med_candidates = candidate_stats.percentile(50)
    p90_candidates = candidate_stats.percentile(90)
    selected_rate = selected_frames / processed_count
    timing_keys = list(timing_stats)
    avg_timing_ms = {
        key: round(timing_stats[key].mean(), 3)
        for key in timing_keys
    }
    p90_timing_ms = {
        key: round(timing_stats[key].percentile(90), 3)
        for key in timing_keys
    }
    p95_timing_ms = {
        key: round(timing_stats[key].percentile(95), 3)
        for key in timing_keys
    }
    p99_timing_ms = {
        key: round(timing_stats[key].percentile(99), 3)
        for key in timing_keys
    }
    max_timing_ms = {
        key: round(timing_stats[key].max(), 3)
        for key in timing_keys
    }

    result = {
        "video": args.video,
        "source_frames": n_total,
        "source_fps": fps_src,
        "downscale": args.downscale,
        "report_mode": args.report_mode,
        "stream_only": args.stream_only,
        "stats_window": args.stats_window,
        "args": vars(args),
        "summary": {
            "n_processed": processed_count,
            "report_frames_stored": len(report),
            "avg_ms_per_frame": round(avg_ms, 3),
            "p90_ms_per_frame": round(p90_ms, 3),
            "p95_ms_per_frame": round(p95_ms, 3),
            "p99_ms_per_frame": round(p99_ms, 3),
            "max_ms_per_frame": round(max_ms, 3),
            "avg_wall_ms_per_frame": round(avg_wall_ms, 3),
            "p90_wall_ms_per_frame": round(p90_wall_ms, 3),
            "p95_wall_ms_per_frame": round(p95_wall_ms, 3),
            "p99_wall_ms_per_frame": round(p99_wall_ms, 3),
            "max_wall_ms_per_frame": round(max_wall_ms, 3),
            "fits_30hz": avg_ms <= 33.3,
            "wall_fits_30hz": avg_wall_ms <= 33.3,
            "wall_p95_fits_30hz": p95_wall_ms <= 33.3,
            "wall_p99_fits_30hz": p99_wall_ms <= 33.3,
            "fits_60hz_on_this_machine": avg_ms <= 16.7,
            "avg_inlier_ratio": round(avg_inlier, 3),
            "avg_candidates_per_frame": round(avg_candidates, 3),
            "median_candidates_per_frame": round(med_candidates, 3),
            "p90_candidates_per_frame": round(p90_candidates, 3),
            "selected_frames": selected_frames,
            "selected_frame_rate": round(selected_rate, 3),
            "selected_output_rows": selected_output_count,
            "selected_output_frame_rate": round(selected_output_count / processed_count, 3),
            "target_local_state_select_count": target_local_state_select_count,
            "srps_verified_select_count": srps_verified_select_count,
            "replay_handoff_select_count": replay_handoff_select_count,
            "kinematic_gate_px_per_frame": round(px_per_frame, 3),
            "kinematic_rejections": 0,
            "multi_candidate_frames": multi_candidate_frames,
            "noisy_frames_gt10_candidates": noisy_frames,
            "model_counts": model_counts,
            "runtime_mode_counts": frame_mode_counts,
            "candidate_router_counts": candidate_router_counts_total,
            "avg_timing_ms": avg_timing_ms,
            "p90_timing_ms": p90_timing_ms,
            "p95_timing_ms": p95_timing_ms,
            "p99_timing_ms": p99_timing_ms,
            "max_timing_ms": max_timing_ms,
        },
        "frames": report,
    }
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "selected_tracks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x", "y", "w", "h", "score", "misses"])
        writer.writerows(selected_rows)
    if selected_feature_rows:
        fieldnames: list[str] = []
        for row in selected_feature_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "selected_tubes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_feature_rows)
    if top_tube_rows:
        fieldnames = []
        for row in top_tube_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "top_tubes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_tube_rows)
    if srps_runtime_residual_rows:
        with (out_dir / "srps_runtime_residual_proposals.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(srps_runtime_residual_rows[0].keys()))
            writer.writeheader()
            writer.writerows(srps_runtime_residual_rows)
    if srps_seed_rows:
        with (out_dir / "srps_seed_candidates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(srps_seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(srps_seed_rows)
    if srps_path_rows:
        fieldnames = []
        for row in srps_path_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (out_dir / "srps_path_candidates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(srps_path_rows)
    if timing_keys:
        with (out_dir / "timing_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["block", "avg_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"])
            writer.writeheader()
            for key in timing_keys:
                writer.writerow(
                    {
                        "block": key,
                        "avg_ms": avg_timing_ms[key],
                        "p90_ms": p90_timing_ms[key],
                        "p95_ms": p95_timing_ms[key],
                        "p99_ms": p99_timing_ms[key],
                        "max_ms": max_timing_ms[key],
                    }
                )
    (out_dir / "summary.md").write_text(
        f"""# candidate TBD summary

Video: `{args.video}`  
Source: {n_total} frames @ {fps_src:.2f} fps  
Processed: {processed_count} frame pairs at downscale {args.downscale}

| metric | value |
|---|---:|
| Avg time / frame | {avg_ms:.2f} ms |
| Avg wall time / frame | {avg_wall_ms:.2f} ms |
| P95 wall time / frame | {p95_wall_ms:.2f} ms |
| P99 wall time / frame | {p99_wall_ms:.2f} ms |
| Max wall time / frame | {max_wall_ms:.2f} ms |
| Avg RANSAC inlier ratio | {avg_inlier:.3f} |
| Avg candidates / frame | {avg_candidates:.2f} |
| Median candidates / frame | {med_candidates:.1f} |
| P90 candidates / frame | {p90_candidates:.1f} |
| Frames with selected box | {selected_frames}/{processed_count} ({selected_rate:.1%}) |
| Kinematic gate | {px_per_frame:.1f} px/frame |

Runtime mode counts: `{frame_mode_counts}`
Candidate router counts: `{candidate_router_counts_total}`

This is candidate-level track-before-detect: selected_frame_rate is not accuracy.
"""
    )
    print(
        f"done -> {out_dir / 'summary.md'}\n"
        f"avg {avg_ms:.2f} ms/frame | inlier {avg_inlier:.1%} | "
        f"cand {avg_candidates:.1f}/frame p90 {p90_candidates:.1f} | selected {selected_rate:.1%}"
    )


if __name__ == "__main__":
    run(parse_args())
