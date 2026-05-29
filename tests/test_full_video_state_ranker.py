import unittest

from scripts import train_full_video_state_ranker as ranker


class FullVideoStateRankerTests(unittest.TestCase):
    def test_make_examples_labels_visible_and_null_candidates(self):
        labels = {
            1: {"visible": False, "bbox": None},
            2: {"visible": True, "bbox": (10.0, 10.0, 4.0, 4.0)},
        }
        top_by_frame = {
            1: [{"frame": "1", "rank": "1", "x": "30", "y": "30", "w": "4", "h": "4"}],
            2: [
                {"frame": "2", "rank": "1", "x": "10", "y": "10", "w": "4", "h": "4"},
                {"frame": "2", "rank": "2", "x": "40", "y": "40", "w": "4", "h": "4"},
                {"frame": "2", "rank": "3", "x": "15", "y": "10", "w": "4", "h": "4"},
            ],
        }

        examples, ignored = ranker.make_examples(labels, top_by_frame, 3.0, 12.0)

        self.assertEqual(ignored, 1)
        self.assertEqual([ex["y"] for ex in examples], [0, 1, 0])
        self.assertEqual([ex["reason"] for ex in examples], ["no_target_negative", "target_positive", "far_negative"])

    def test_stratified_folds_keep_visible_frames_distributed(self):
        labels = {
            i: {"visible": i >= 6, "bbox": (0.0, 0.0, 1.0, 1.0) if i >= 6 else None}
            for i in range(10)
        }

        folds = ranker.make_frame_folds(labels, 2, "stratified_blocks")

        self.assertEqual(set(folds), set(labels))
        self.assertEqual({folds[i] for i in range(6)}, {0, 1})
        self.assertEqual({folds[i] for i in range(6, 10)}, {0, 1})


if __name__ == "__main__":
    unittest.main()
