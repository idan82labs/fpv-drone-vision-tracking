from __future__ import annotations

import pytest

from scripts.apply_true_ground_profile_mux import (
    DEFAULT_WEIGHTS,
    CandidatePayload,
    bridge_selected_gaps,
    dedupe_items,
    recenter_emitted_boxes,
    recover_target_local_gaps,
    selected_track_score,
    top_tube_score,
)
from scripts.selector_core import SequenceItem


def test_large_dark_selected_track_scores_above_generic() -> None:
    generic = {
        "selected": "1",
        "rank": "2",
        "learned_score": "0.85",
        "verified_score": "80",
        "source": "map",
    }
    large_dark = dict(generic, source="large_dark")

    assert selected_track_score("score084", large_dark, DEFAULT_WEIGHTS) > selected_track_score(
        "score084", generic, DEFAULT_WEIGHTS
    )


def test_top_tube_score_prefers_target_probability_over_clutter_probability() -> None:
    base = {
        "rank": "1",
        "score": "30",
        "verified_score": "40",
        "cand_source": "map",
        "crop_t_prob": "0.8",
        "crop_s_prob": "0.05",
        "crop_e_prob": "0.05",
        "crop_h_prob": "0.05",
        "crop_g_prob": "0.1",
    }
    clutter = dict(base, crop_t_prob="0.2", crop_e_prob="0.9")

    assert top_tube_score(base, DEFAULT_WEIGHTS) > top_tube_score(clutter, DEFAULT_WEIGHTS)


def test_dedupe_keeps_highest_scoring_same_source_box() -> None:
    low_payload = CandidatePayload("clip", 1, (1.0, 2.0, 3.0, 4.0), "score084", 1.0)
    high_payload = CandidatePayload("clip", 1, (1.1, 2.1, 3.0, 4.0), "score084", 2.0)
    other_source = CandidatePayload("clip", 1, (1.0, 2.0, 3.0, 4.0), "cropw9", 1.5)
    items = [
        SequenceItem(1, low_payload.bbox, low_payload.mux_score, low_payload),
        SequenceItem(1, high_payload.bbox, high_payload.mux_score, high_payload),
        SequenceItem(1, other_source.bbox, other_source.mux_score, other_source),
    ]

    deduped = dedupe_items(items)

    assert len(deduped) == 2
    assert {item.payload.mux_source for item in deduped} == {"score084", "cropw9"}
    assert max(item.score for item in deduped if item.payload.mux_source == "score084") == 2.0


def test_bridge_selected_gaps_interpolates_short_high_score_gap() -> None:
    left = CandidatePayload("clip", 10, (10.0, 20.0, 4.0, 4.0), "score084", 4.0, track_id="a")
    right = CandidatePayload("clip", 13, (16.0, 26.0, 8.0, 8.0), "score084", 4.0, track_id="b")
    selected = {
        10: SequenceItem(10, left.bbox, left.mux_score, left),
        13: SequenceItem(13, right.bbox, right.mux_score, right),
    }

    bridged = bridge_selected_gaps(selected, max_gap_frames=2, score_floor=3.25)

    assert sorted(bridged) == [10, 11, 12, 13]
    assert bridged[11].payload.mux_source == "bridge"
    assert bridged[11].bbox == pytest.approx((12.0, 22.0, 5.333333333333333, 5.333333333333333))


def test_bridge_selected_gaps_does_not_fill_long_or_low_score_gaps() -> None:
    low = CandidatePayload("clip", 10, (10.0, 20.0, 4.0, 4.0), "score084", 2.0)
    high = CandidatePayload("clip", 13, (16.0, 26.0, 8.0, 8.0), "score084", 4.0)
    long_end = CandidatePayload("clip", 20, (20.0, 30.0, 8.0, 8.0), "score084", 4.0)

    low_gap = {
        10: SequenceItem(10, low.bbox, low.mux_score, low),
        13: SequenceItem(13, high.bbox, high.mux_score, high),
    }
    long_gap = {
        10: SequenceItem(10, high.bbox, high.mux_score, high),
        20: SequenceItem(20, long_end.bbox, long_end.mux_score, long_end),
    }

    assert sorted(bridge_selected_gaps(low_gap, max_gap_frames=2, score_floor=3.25)) == [10, 13]
    assert sorted(bridge_selected_gaps(long_gap, max_gap_frames=2, score_floor=3.25)) == [10, 20]


def test_recover_target_local_gaps_uses_recent_motion_prediction() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 4.0, 4.0), "score084", 4.0)
    second = CandidatePayload("clip", 11, (14.0, 10.0, 4.0, 4.0), "score084", 4.0)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
    }
    tubes = {
        12: [
            {"clip": "clip", "frame": "12", "rank": "1", "x": "18", "y": "10", "w": "4", "h": "4", "score": "14"},
            {"clip": "clip", "frame": "12", "rank": "2", "x": "50", "y": "50", "w": "4", "h": "4", "score": "40"},
        ]
    }

    recovered = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=3.45,
        max_recovery_frames=2,
        top_k=2,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=10.0,
    )

    assert 12 in recovered
    assert recovered[12].payload.mux_source == "target_local_recovery"
    assert recovered[12].bbox == (18.0, 10.0, 4.0, 4.0)


def test_recover_target_local_gaps_does_not_start_without_history() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 4.0, 4.0), "score084", 4.0)
    selected = {10: SequenceItem(10, first.bbox, first.mux_score, first)}
    tubes = {11: [{"clip": "clip", "frame": "11", "rank": "1", "x": "14", "y": "10", "w": "4", "h": "4", "score": "20"}]}

    recovered = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=3.45,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=10.0,
    )

    assert sorted(recovered) == [10]


def test_recover_target_local_respects_absolute_frame_horizon() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 4.0, 4.0), "score084", 4.0)
    second = CandidatePayload("clip", 11, (14.0, 10.0, 4.0, 4.0), "score084", 4.0)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
    }
    tubes = {
        20: [
            {"clip": "clip", "frame": "20", "rank": "1", "x": "50", "y": "10", "w": "4", "h": "4", "score": "30"},
        ]
    }

    recovered = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=3.45,
        max_recovery_frames=3,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=10.0,
    )

    assert sorted(recovered) == [10, 11]


def test_recover_target_local_gaps_can_use_last_anchor_for_close_pass_jitter() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 4.0, 4.0), "score084", 4.2)
    second = CandidatePayload("clip", 11, (34.0, 10.0, 4.0, 4.0), "score084", 4.2)
    low_score_current = CandidatePayload("clip", 12, (0.0, 0.0, 4.0, 4.0), "cropw9", 3.7)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
        12: SequenceItem(12, low_score_current.bbox, low_score_current.mux_score, low_score_current),
    }
    tubes = {
        12: [
            {"clip": "clip", "frame": "12", "rank": "1", "x": "30", "y": "10", "w": "4", "h": "4", "score": "18"},
        ]
    }

    without_anchor = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=10.0,
    )
    with_anchor = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=8.0,
        anchor_mux_sources={"cropw9"},
        raw_min=10.0,
    )

    assert without_anchor[12].payload.mux_source == "cropw9"
    assert with_anchor[12].bbox == (30.0, 10.0, 4.0, 4.0)
    assert with_anchor[12].payload.reason.startswith("anchor_error_")


def test_recover_target_local_can_replace_far_existing_selection() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 4.0, 4.0), "score084", 4.4)
    second = CandidatePayload("clip", 11, (14.0, 10.0, 4.0, 4.0), "score084", 4.4)
    wrong_current = CandidatePayload("clip", 12, (70.0, 70.0, 4.0, 4.0), "score084", 5.5)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
        12: SequenceItem(12, wrong_current.bbox, wrong_current.mux_score, wrong_current),
    }
    tubes = {
        12: [
            {"clip": "clip", "frame": "12", "rank": "7", "x": "18", "y": "10", "w": "4", "h": "4", "score": "9"},
            {"clip": "clip", "frame": "12", "rank": "1", "x": "70", "y": "70", "w": "4", "h": "4", "score": "40"},
        ]
    }

    recovered = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=2,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=5.0,
        replace_existing_error=20.0,
        replace_improvement_px=4.0,
    )

    assert recovered[12].payload.mux_source == "target_local_recovery"
    assert recovered[12].bbox == (18.0, 10.0, 4.0, 4.0)
    assert recovered[12].payload.reason.startswith("replace_current_error_")


def test_recover_target_local_does_not_replace_near_existing_selection() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 4.0, 4.0), "score084", 4.4)
    second = CandidatePayload("clip", 11, (14.0, 10.0, 4.0, 4.0), "score084", 4.4)
    near_current = CandidatePayload("clip", 12, (19.5, 10.0, 4.0, 4.0), "score084", 5.5)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
        12: SequenceItem(12, near_current.bbox, near_current.mux_score, near_current),
    }
    tubes = {
        12: [
            {"clip": "clip", "frame": "12", "rank": "1", "x": "18", "y": "10", "w": "4", "h": "4", "score": "20"},
        ]
    }

    recovered = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=5.0,
        replace_existing_error=20.0,
        replace_improvement_px=4.0,
    )

    assert recovered[12].payload.mux_source == "score084"
    assert recovered[12].bbox == near_current.bbox


def test_recover_target_local_replace_can_require_min_box_side() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 8.0, 8.0), "score084", 4.4)
    second = CandidatePayload("clip", 11, (18.0, 10.0, 8.0, 8.0), "score084", 4.4)
    wrong_current = CandidatePayload("clip", 12, (70.0, 70.0, 8.0, 8.0), "score084", 5.5)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
        12: SequenceItem(12, wrong_current.bbox, wrong_current.mux_score, wrong_current),
    }
    tubes = {
        12: [
            {"clip": "clip", "frame": "12", "rank": "1", "x": "26", "y": "10", "w": "4", "h": "4", "score": "30"},
        ]
    }

    recovered = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=5.0,
        replace_existing_error=20.0,
        replace_improvement_px=4.0,
        replace_min_side=8.0,
    )

    assert recovered[12].payload.mux_source == "score084"
    assert recovered[12].bbox == wrong_current.bbox


def test_recover_target_local_replace_can_use_deeper_lower_score_window() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 8.0, 8.0), "score084", 4.4)
    second = CandidatePayload("clip", 11, (18.0, 10.0, 8.0, 8.0), "score084", 4.4)
    wrong_current = CandidatePayload("clip", 12, (70.0, 70.0, 8.0, 8.0), "score084", 5.5)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
        12: SequenceItem(12, wrong_current.bbox, wrong_current.mux_score, wrong_current),
    }
    tubes = {
        12: [
            {"clip": "clip", "frame": "12", "rank": "1", "x": "70", "y": "70", "w": "8", "h": "8", "score": "40"},
            {"clip": "clip", "frame": "12", "rank": "2", "x": "64", "y": "64", "w": "8", "h": "8", "score": "35"},
            {"clip": "clip", "frame": "12", "rank": "80", "x": "26", "y": "10", "w": "8", "h": "8", "score": "-1.5"},
        ]
    }

    shallow = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=5.0,
        replace_existing_error=20.0,
        replace_improvement_px=4.0,
        replace_min_side=7.0,
    )
    deep = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=0.0,
        anchor_mux_sources=None,
        raw_min=5.0,
        replace_existing_error=20.0,
        replace_improvement_px=4.0,
        replace_min_side=7.0,
        replace_top_k=80,
        replace_raw_min=-2.0,
    )

    assert shallow[12].payload.mux_source == "score084"
    assert deep[12].payload.mux_source == "target_local_recovery"
    assert deep[12].bbox == (26.0, 10.0, 8.0, 8.0)


def test_recover_target_local_anchor_respects_low_score_source_allow_list() -> None:
    first = CandidatePayload("clip", 10, (10.0, 10.0, 4.0, 4.0), "score084", 4.2)
    second = CandidatePayload("clip", 11, (34.0, 10.0, 4.0, 4.0), "score084", 4.2)
    low_score_current = CandidatePayload("clip", 12, (0.0, 0.0, 4.0, 4.0), "cs_proposal", 1.0)
    selected = {
        10: SequenceItem(10, first.bbox, first.mux_score, first),
        11: SequenceItem(11, second.bbox, second.mux_score, second),
        12: SequenceItem(12, low_score_current.bbox, low_score_current.mux_score, low_score_current),
    }
    tubes = {
        12: [
            {"clip": "clip", "frame": "12", "rank": "1", "x": "30", "y": "10", "w": "4", "h": "4", "score": "18"},
        ]
    }

    recovered = recover_target_local_gaps(
        selected,
        tubes,
        min_emit_score=4.05,
        max_recovery_frames=2,
        top_k=1,
        max_pred_error=6.0,
        last_anchor_px=8.0,
        anchor_mux_sources={"cropw9"},
        raw_min=10.0,
    )

    assert recovered[12].payload.mux_source == "cs_proposal"


def test_recenter_emitted_boxes_tightens_existing_large_box() -> None:
    payload = CandidatePayload("clip", 12, (10.0, 10.0, 20.0, 20.0), "score084", 4.2)
    selected = {12: SequenceItem(12, payload.bbox, payload.mux_score, payload)}
    tubes = {
        12: [
            {
                "clip": "clip",
                "frame": "12",
                "rank": "4",
                "x": "16",
                "y": "16",
                "w": "6",
                "h": "6",
                "score": "12",
                "crop_t_prob": "0.2",
                "crop_pred_class": "E",
            }
        ]
    }

    recentered = recenter_emitted_boxes(
        selected,
        tubes,
        min_emit_score=4.05,
        top_k=5,
        max_delta_px=18,
        raw_min=-5,
        min_area_ratio=0.05,
        max_area_ratio=0.9,
    )

    assert recentered[12].bbox == (16.0, 16.0, 6.0, 6.0)
    assert recentered[12].payload.mux_source == "score084+recenter"


def test_recenter_emitted_boxes_does_not_create_or_recenter_low_score_items() -> None:
    payload = CandidatePayload("clip", 12, (10.0, 10.0, 20.0, 20.0), "score084", 3.0)
    selected = {12: SequenceItem(12, payload.bbox, payload.mux_score, payload)}
    tubes = {12: [{"clip": "clip", "frame": "12", "rank": "1", "x": "16", "y": "16", "w": "6", "h": "6", "score": "20"}]}

    recentered = recenter_emitted_boxes(
        selected,
        tubes,
        min_emit_score=4.05,
        top_k=5,
        max_delta_px=18,
        raw_min=-5,
        min_area_ratio=0.05,
        max_area_ratio=0.9,
    )

    assert recentered[12].bbox == payload.bbox


def test_recenter_emitted_boxes_respects_source_allow_list() -> None:
    payload = CandidatePayload("clip", 12, (10.0, 10.0, 20.0, 20.0), "cs_js1", 4.2)
    selected = {12: SequenceItem(12, payload.bbox, payload.mux_score, payload)}
    tubes = {12: [{"clip": "clip", "frame": "12", "rank": "1", "x": "16", "y": "16", "w": "6", "h": "6", "score": "20"}]}

    recentered = recenter_emitted_boxes(
        selected,
        tubes,
        min_emit_score=4.05,
        top_k=5,
        max_delta_px=18,
        raw_min=-5,
        min_area_ratio=0.05,
        max_area_ratio=0.9,
        allowed_sources={"score084"},
    )

    assert recentered[12].bbox == payload.bbox


def test_recenter_emitted_boxes_can_reject_tiny_clutter_candidate() -> None:
    payload = CandidatePayload("clip", 12, (10.0, 10.0, 12.0, 12.0), "score084", 4.2)
    selected = {12: SequenceItem(12, payload.bbox, payload.mux_score, payload)}
    tubes = {
        12: [
            {
                "clip": "clip",
                "frame": "12",
                "rank": "1",
                "x": "12",
                "y": "12",
                "w": "3",
                "h": "3",
                "score": "40",
                "crop_t_prob": "0.03",
                "crop_e_prob": "0.80",
                "crop_pred_class": "E",
            },
            {
                "clip": "clip",
                "frame": "12",
                "rank": "2",
                "x": "13",
                "y": "13",
                "w": "8",
                "h": "8",
                "score": "20",
                "crop_t_prob": "0.18",
                "crop_e_prob": "0.55",
                "crop_pred_class": "E",
            },
        ]
    }

    recentered = recenter_emitted_boxes(
        selected,
        tubes,
        min_emit_score=4.05,
        top_k=5,
        max_delta_px=18,
        raw_min=-5,
        min_area_ratio=0.05,
        max_area_ratio=0.9,
        t_margin_weight=0.9,
        clutter_weight=0.8,
        tiny_area_px=25,
        tiny_penalty=0.8,
    )

    assert recentered[12].bbox == (13.0, 13.0, 8.0, 8.0)


def test_recenter_emitted_boxes_preserves_high_confidence_source() -> None:
    payload = CandidatePayload(
        "clip",
        12,
        (10.0, 10.0, 12.0, 12.0),
        "score084",
        4.2,
        learned_score="0.90",
        source="large_dark",
    )
    selected = {12: SequenceItem(12, payload.bbox, payload.mux_score, payload)}
    tubes = {
        12: [
            {
                "clip": "clip",
                "frame": "12",
                "rank": "1",
                "x": "12",
                "y": "12",
                "w": "3",
                "h": "3",
                "score": "40",
                "crop_t_prob": "0.03",
                "crop_e_prob": "0.80",
            }
        ]
    }

    recentered = recenter_emitted_boxes(
        selected,
        tubes,
        min_emit_score=4.05,
        top_k=5,
        max_delta_px=18,
        raw_min=-5,
        min_area_ratio=0.05,
        max_area_ratio=0.9,
        allowed_sources={"score084"},
        preserve_sources={"large_dark"},
        preserve_min_learned=0.84,
    )

    assert recentered[12].bbox == payload.bbox
