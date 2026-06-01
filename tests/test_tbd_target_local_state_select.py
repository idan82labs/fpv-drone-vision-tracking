import argparse

from scripts import tbd_motion_detector as detector


class FakeTBD:
    def verified_score(self, st: detector.PathState) -> float:
        return st.score()


def make_args(**overrides):
    defaults = dict(
        target_local_state_select=True,
        target_local_state_select_error_px=9.0,
        target_local_state_select_improvement_px=6.0,
        target_local_state_select_max_pred_error_px=6.0,
        target_local_state_select_anchor_px=18.0,
        target_local_state_select_min_side=7.0,
        target_local_state_select_max_side=18.0,
        target_local_state_select_top_n=80,
        target_local_state_select_sources="target_local_recovery",
        target_local_state_select_allow_missed_anchor=False,
        target_local_state_select_missed_max_misses=1,
        target_local_recovery_max_seed_gap=5,
        target_local_recovery_min_hits=3,
        target_local_recovery_min_verified_score=6.0,
        target_local_recovery_predictor="clamped_velocity",
        target_local_recovery_max_velocity_px=3.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_candidate(source: str, bbox: tuple[int, int, int, int], score: float = 8.0) -> detector.base.Candidate:
    return detector.base.Candidate(
        source=source,
        bbox=bbox,
        area=max(1, bbox[2] * bbox[3]),
        fill=1.0,
        aspect=1.0,
        mean_residual=score,
        mean_appearance=score,
        local_contrast=score,
        texture=0.0,
        line_context=0.0,
        isolation=1.0,
        score=score,
    )


def make_state(
    sid: int,
    bbox: tuple[int, int, int, int],
    source: str = "target_local_recovery",
    score: float = 8.0,
    hits: int = 1,
) -> detector.PathState:
    cand = make_candidate(source, bbox, score)
    return detector.PathState(
        sid=sid,
        bbox=bbox,
        last_frame=2,
        misses=0,
        hits=hits,
        last_candidate=cand,
        contribs=[score],
        hit_flags=[True] * hits,
    )


def make_seed(**overrides) -> detector.TargetLocalRecoverySeed:
    values = dict(
        sid=1,
        bbox=(10, 10, 8, 8),
        vx=10.0,
        vy=0.0,
        frame_no=1,
        hits=3,
        verified_score=10.0,
    )
    values.update(overrides)
    return detector.TargetLocalRecoverySeed(**values)


def run_override(selected, states, seed=None, args=None):
    return detector.target_local_state_select_override(
        selected,
        states,
        seed if seed is not None else make_seed(),
        frame_no=2,
        w_img=100,
        h_img=80,
        tbd=FakeTBD(),
        args=args if args is not None else make_args(),
    )


def test_target_local_state_select_replaces_far_existing_lock():
    selected = make_state(10, (55, 10, 8, 8), source="large_dark", score=30.0)
    recovery = make_state(11, (21, 10, 8, 8), score=4.0)

    chosen, info = run_override(selected, [selected, recovery])

    assert chosen is recovery
    assert info["used"] is True
    assert info["reason"] == "target_local_state_select"


def test_target_local_state_select_keeps_near_existing_selection():
    selected = make_state(10, (20, 10, 8, 8), source="large_dark", score=30.0)
    recovery = make_state(11, (21, 10, 8, 8), score=4.0)

    chosen, info = run_override(selected, [selected, recovery])

    assert chosen is selected
    assert info["used"] is False
    assert info["reason"] == "selected_near_prediction"


def test_target_local_state_select_keeps_selection_near_anchor_when_velocity_is_bad():
    seed = make_seed(vx=30.0, vy=0.0)
    selected = make_state(10, (12, 10, 8, 8), source="appearance", score=10.0)
    recovery = make_state(11, (41, 10, 8, 8), score=4.0)
    args = make_args(target_local_recovery_predictor="state_velocity")

    chosen, info = run_override(selected, [selected, recovery], seed=seed, args=args)

    assert chosen is selected
    assert info["used"] is False
    assert info["reason"] == "selected_near_anchor"


def test_target_local_state_select_can_acquire_when_no_state_selected():
    recovery = make_state(11, (21, 10, 8, 8), score=4.0)

    chosen, info = run_override(None, [recovery])

    assert chosen is recovery
    assert info["used"] is True


def test_target_local_state_select_respects_source_allow_list():
    selected = make_state(10, (55, 10, 8, 8), source="large_dark", score=30.0)
    wrong_source = make_state(11, (21, 10, 8, 8), source="large_dark", score=4.0)

    chosen, info = run_override(selected, [selected, wrong_source])

    assert chosen is selected
    assert info["used"] is False
    assert info["reason"] == "no_candidate"


def test_target_local_state_select_can_preserve_near_anchor_missed_state_when_enabled():
    selected = make_state(10, (55, 10, 8, 8), source="large_dark", score=30.0)
    missed = make_state(11, (21, 10, 5, 5), source="appearance", score=20.0)
    missed.misses = 1
    missed.last_candidate = None
    args = make_args(
        target_local_state_select_allow_missed_anchor=True,
        target_local_state_select_min_side=4.0,
    )

    chosen, info = run_override(selected, [selected, missed], args=args)

    assert chosen is missed
    assert info["used"] is True
    assert info["candidate_source"] == "missed_anchor"


def test_target_local_state_select_does_not_use_missed_anchor_by_default():
    selected = make_state(10, (55, 10, 8, 8), source="large_dark", score=30.0)
    missed = make_state(11, (21, 10, 5, 5), source="appearance", score=20.0)
    missed.misses = 1
    missed.last_candidate = None
    args = make_args(target_local_state_select_min_side=4.0)

    chosen, info = run_override(selected, [selected, missed], args=args)

    assert chosen is selected
    assert info["used"] is False
    assert info["reason"] == "no_candidate"


def test_target_local_state_select_requires_min_side():
    selected = make_state(10, (55, 10, 8, 8), source="large_dark", score=30.0)
    tiny = make_state(11, (23, 12, 4, 4), score=4.0)

    chosen, info = run_override(selected, [selected, tiny])

    assert chosen is selected
    assert info["used"] is False
    assert info["reason"] == "no_candidate"


def test_target_local_state_select_rejects_stale_seed():
    selected = make_state(10, (55, 10, 8, 8), source="large_dark", score=30.0)
    recovery = make_state(11, (21, 10, 8, 8), score=4.0)
    stale_seed = make_seed(frame_no=-10)

    chosen, info = run_override(selected, [selected, recovery], seed=stale_seed)

    assert chosen is selected
    assert info["used"] is False
    assert info["reason"] == "seed_gap"
