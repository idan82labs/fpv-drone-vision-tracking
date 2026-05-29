import argparse
import io
import json
import unittest

from scripts.tbd_motion_detector import DelayedSequenceSelector, PathState, write_telemetry_output


class FakeTBD:
    def verified_score(self, st):
        return st.score()


def args():
    return argparse.Namespace(
        delayed_sequence_top_n=20,
        delayed_sequence_min_hits=1,
        delayed_sequence_window=1,
        delayed_sequence_max_jump_px=5.0,
        delayed_sequence_transition_weight=1.5,
        delayed_sequence_threshold=0.0,
        max_selected_misses=1,
        min_path_hits=1,
        selected_score=0.0,
        tube_verifier="off",
        delayed_sequence_require_floor=False,
        delayed_sequence_commit_prefix=False,
    )


def state(sid, frame, x, y, score):
    return PathState(
        sid=sid,
        bbox=(x, y, 4, 4),
        last_frame=frame,
        contribs=[score],
        hit_flags=[True],
    )


class DelayedSequenceSelectorTest(unittest.TestCase):
    def test_unreachable_late_birth_does_not_backfill_old_frame(self):
        selector = DelayedSequenceSelector(args())
        selector.add_frame(1, [state(1, 1, 0, 0, 9.0)], FakeTBD())
        selector.add_frame(2, [state(2, 2, 100, 100, 20.0)], FakeTBD())

        frame, selected = selector.pop_ready()

        self.assertEqual(frame, 1)
        self.assertIsNone(selected)

    def test_reachable_path_emits_oldest_state(self):
        selector = DelayedSequenceSelector(args())
        selector.add_frame(1, [state(1, 1, 0, 0, 9.0)], FakeTBD())
        selector.add_frame(2, [state(1, 2, 2, 0, 8.0)], FakeTBD())

        frame, selected = selector.pop_ready()

        self.assertEqual(frame, 1)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.sid, 1)

    def test_below_floor_state_is_not_emitted(self):
        a = args()
        a.selected_score = 10.0
        a.delayed_sequence_require_floor = True
        selector = DelayedSequenceSelector(a)
        selector.add_frame(1, [state(1, 1, 0, 0, 9.0)], FakeTBD())
        selector.add_frame(2, [state(1, 2, 2, 0, 8.0)], FakeTBD())

        frame, selected = selector.pop_ready()

        self.assertEqual(frame, 1)
        self.assertIsNone(selected)

    def test_committed_pops_stay_on_reachable_branch(self):
        a = args()
        a.delayed_sequence_commit_prefix = True
        selector = DelayedSequenceSelector(a)
        selector.add_frame(1, [state(1, 1, 0, 0, 9.0), state(2, 1, 60, 0, 7.0)], FakeTBD())
        selector.add_frame(2, [state(1, 2, 2, 0, 8.0), state(2, 2, 62, 0, 7.0)], FakeTBD())
        selector.add_frame(3, [state(3, 3, 120, 0, 20.0), state(1, 3, 4, 0, 6.0)], FakeTBD())

        _frame1, selected1 = selector.pop_ready()
        _frame2, selected2 = selector.pop_ready()

        self.assertIsNotNone(selected1)
        self.assertIsNotNone(selected2)
        self.assertLessEqual(abs(selected2.bbox[0] - selected1.bbox[0]), 5)

    def test_telemetry_emits_null_and_selected_records(self):
        sink = io.StringIO()
        write_telemetry_output(sink, 10, 9, None, FakeTBD(), "delayed_sequence", "warming", 1.2, 1.5)
        write_telemetry_output(sink, 11, 10, state(1, 10, 2, 3, 7.0), FakeTBD(), "delayed_sequence", "selected", 1.3, 1.6)

        rows = [json.loads(line) for line in sink.getvalue().splitlines()]

        self.assertFalse(rows[0]["selected"])
        self.assertEqual(rows[0]["status"], "warming")
        self.assertTrue(rows[1]["selected"])
        self.assertEqual(rows[1]["bbox"], [2, 3, 4, 4])
        self.assertEqual(rows[1]["selected_frame"], 10)


if __name__ == "__main__":
    unittest.main()
