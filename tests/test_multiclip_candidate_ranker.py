import unittest

from scripts import evaluate_multiclip_candidate_ranker as multiclip


class MulticlipCandidateRankerTests(unittest.TestCase):
    def test_aggregate_counts_visible_hits_and_invisible_no_box(self):
        rows = [
            {"visible": 1, "selected": 1, "strict_hit": 1, "loose_hit": 1},
            {"visible": 1, "selected": 1, "strict_hit": 0, "loose_hit": 1},
            {"visible": 0, "selected": 0, "strict_hit": 0, "loose_hit": 0},
            {"visible": 0, "selected": 1, "strict_hit": 0, "loose_hit": 0},
        ]

        out = multiclip.aggregate(rows, "m", 0.5)

        self.assertEqual(out["visible_frames"], 2)
        self.assertEqual(out["strict_hits"], 1)
        self.assertEqual(out["strict_recall"], 0.5)
        self.assertEqual(out["loose_recall"], 1.0)
        self.assertEqual(out["invisible_no_box"], 1)
        self.assertEqual(out["invisible_no_box_rate"], 0.5)

    def test_oracle_summary_uses_positive_candidate_frames(self):
        labels = {
            1: {"visible": True},
            2: {"visible": True},
            3: {"visible": False},
        }
        examples = [
            {"frame": 1, "y": 1},
            {"frame": 1, "y": 0},
            {"frame": 2, "y": 0},
            {"frame": 3, "y": 0},
        ]

        out = multiclip.oracle_summary(labels, examples, "clip")

        self.assertEqual(out["visible_frames"], 2)
        self.assertEqual(out["visible_oracle_strict_frames"], 1)
        self.assertEqual(out["visible_oracle_strict_rate"], 0.5)
        self.assertEqual(out["candidate_label_frames"], 3)


if __name__ == "__main__":
    unittest.main()
