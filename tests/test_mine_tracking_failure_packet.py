import unittest

import pandas as pd

from scripts import mine_tracking_failure_packet as miner


class MineTrackingFailurePacketTests(unittest.TestCase):
    def test_gap_events_find_long_no_selection_runs(self):
        rows = pd.DataFrame(
            [
                {"frame": 0, "selected": 1},
                {"frame": 1, "selected": 0},
                {"frame": 2, "selected": 0},
                {"frame": 3, "selected": 0},
                {"frame": 4, "selected": 1},
                {"frame": 8, "selected": 1},
            ]
        )

        events = miner.gap_events("clip", rows, frame_count=10, min_gap_frames=3)

        self.assertEqual(
            [(e.start_frame, e.end_frame, e.issue_type) for e in events],
            [(1, 3, "no_selection_gap"), (5, 7, "no_selection_gap")],
        )

    def test_jump_events_score_large_selected_box_motion(self):
        rows = pd.DataFrame(
            [
                {"frame": 10, "selected": 1, "x": 10, "y": 10, "w": 4, "h": 4, "router_bucket": "surface"},
                {"frame": 11, "selected": 1, "x": 80, "y": 10, "w": 4, "h": 4, "router_bucket": "surface"},
            ]
        )

        events = miner.jump_events("clip", rows, jump_px=28.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].issue_type, "selected_jump")
        self.assertEqual(events[0].start_frame, 10)
        self.assertEqual(events[0].end_frame, 11)
        self.assertGreater(events[0].score, 70.0)

    def test_choose_review_frames_samples_gap_edges_and_middle(self):
        event = miner.Event("clip", "no_selection_gap", 10, 20, 11.0, "gap")

        self.assertEqual(miner.choose_review_frames(event), [10, 15, 20])


if __name__ == "__main__":
    unittest.main()
