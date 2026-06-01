import unittest

from scripts import train_surface_halo_ranker as train


class TrainSurfaceHaloRankerTests(unittest.TestCase):
    def test_assign_frame_folds_interleaves_sorted_frames(self):
        labels = [{"frame": str(frame)} for frame in [30, 10, 20, 40, 50]]

        folds = train.assign_frame_folds(labels, folds=3, mode="interleaved")

        self.assertEqual(folds[10], 0)
        self.assertEqual(folds[20], 1)
        self.assertEqual(folds[30], 2)
        self.assertEqual(folds[40], 0)
        self.assertEqual(folds[50], 1)

    def test_assign_frame_folds_blocked_uses_contiguous_ranges(self):
        labels = [{"frame": str(frame)} for frame in range(10)]

        folds = train.assign_frame_folds(labels, folds=3, mode="blocked")

        self.assertEqual([folds[i] for i in range(4)], [0, 0, 0, 0])
        self.assertEqual([folds[i] for i in range(4, 7)], [1, 1, 1])
        self.assertEqual([folds[i] for i in range(7, 10)], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
