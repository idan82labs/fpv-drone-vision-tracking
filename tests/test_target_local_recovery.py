import argparse
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts import evaluate_target_local_recovery as recovery


class TargetLocalRecoveryTests(unittest.TestCase):
    def test_constant_velocity_prediction_uses_prior_labels_only(self):
        history = [
            recovery.Label("c", 10, 10, 10, 4, 4),
            recovery.Label("c", 12, 14, 12, 4, 4),
        ]

        pred = recovery.predict_from_history(history, 14, max_seed_gap=5, predictor="constant_velocity")

        self.assertIsNotNone(pred)
        assert pred is not None
        pred_x, pred_y, seed_frame, seed_gap = pred
        self.assertEqual(seed_frame, 12)
        self.assertEqual(seed_gap, 2)
        self.assertAlmostEqual(pred_x, 20.0)
        self.assertAlmostEqual(pred_y, 16.0)

    def test_prediction_rejected_when_seed_gap_is_too_large(self):
        history = [recovery.Label("c", 10, 10, 10, 4, 4)]

        self.assertIsNone(recovery.predict_from_history(history, 20, max_seed_gap=5, predictor="previous"))

    def test_local_recovery_finds_dark_peak_near_prediction(self):
        gray = np.full((80, 80), 160, dtype=np.uint8)
        yy, xx = np.mgrid[:80, :80]
        gray[np.hypot(xx - 42, yy - 39) <= 2.0] = 30
        label = recovery.Label("c", 2, 19, 18, 4, 4)
        prediction = (20.0, 19.5, 1, 1)

        candidates = recovery.recover_candidates(
            label,
            prediction,
            gray,
            radii_orig=[2],
            detector_scale=0.5,
            search_radius_det_px=8,
            texture_weight=0.0,
            peaks_per_frame=3,
            box_size_det_px=4,
            shift_penalty=0.0,
        )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertLess(recovery.center_dist_label(candidates[0], label), 2.0)

    def test_end_to_end_writes_summary_and_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "tiny.mp4"
            labels = root / "labels.csv"
            out_dir = root / "out"

            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (80, 80),
                isColor=True,
            )
            for frame in range(4):
                img = np.full((80, 80, 3), 160, dtype=np.uint8)
                cv2.circle(img, (40 + frame, 40), 2, (25, 25, 25), -1)
                writer.write(img)
            writer.release()

            labels.write_text(
                "clip,frame,visible,det_x,det_y,det_w,det_h\n"
                "c,0,1,18,18,4,4\n"
                "c,1,1,18.5,18,4,4\n"
                "c,2,1,19,18,4,4\n"
                "c,3,1,19.5,18,4,4\n"
            )
            args = argparse.Namespace(
                labels=str(labels),
                video=str(video),
                out_dir=str(out_dir),
                clip="c",
                frame_min=-1,
                frame_max=-1,
                detector_scale=0.5,
                strict_tol_px=8.0,
                loose_tol_px=16.0,
                max_seed_gap=3,
                search_radius_det_px=8.0,
                radii_orig="2",
                texture_weight=0.0,
                peaks_per_frame=3,
                box_size_det_px=4.0,
                shift_penalty=0.0,
                predictor="constant_velocity",
            )

            summary = recovery.run(args)

            self.assertEqual(summary["labels"], 4)
            self.assertEqual(summary["seeded_frames"], 3)
            self.assertGreater(summary["topk_strict_rate"], 0.9)
            self.assertTrue((out_dir / "target_local_candidates.csv").exists())


if __name__ == "__main__":
    unittest.main()
