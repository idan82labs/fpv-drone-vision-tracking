import math
import unittest

import numpy as np

from scripts import augment_top_tubes_prop_flicker_features as prop


class PropFlickerFeatureTests(unittest.TestCase):
    def test_temporal_features_detect_periodic_ac_above_flat_trace(self):
        flat = [1.0] * 16
        wave = [math.sin(2.0 * math.pi * i / 4.0) for i in range(16)]

        flat_feat = prop.temporal_features(flat, fps=32.0)
        wave_feat = prop.temporal_features(wave, fps=32.0)

        self.assertEqual(flat_feat["samples"], 16.0)
        self.assertGreater(wave_feat["ac_rms"], flat_feat["ac_rms"])
        self.assertGreater(wave_feat["periodic_score"], flat_feat["periodic_score"])
        self.assertGreater(wave_feat["peak_ratio"], 0.25)

    def test_detrending_keeps_alternating_signal_but_removes_slow_ramp(self):
        ramp = np.asarray([float(i) for i in range(16)], dtype=np.float32)
        alternating = np.asarray([1.0 if i % 2 == 0 else -1.0 for i in range(16)], dtype=np.float32)

        ramp_resid = prop.detrend_trace(ramp)
        combo_resid = prop.detrend_trace(ramp + alternating)

        self.assertLess(float(np.std(ramp_resid)), 1e-4)
        self.assertGreater(float(np.std(combo_resid)), 0.8)

    def test_robust_gain_compares_target_to_local_controls(self):
        z, med, sigma = prop.robust_gain(1.5, [0.2, 0.2, 0.25, 5.0])

        self.assertAlmostEqual(med, 0.225, places=3)
        self.assertGreater(sigma, 0.0)
        self.assertGreater(z, 1.0)

    def test_crop_dark_signal_positive_for_dark_center(self):
        crop = np.full((17, 17), 100.0, dtype=np.float32)
        crop[7:10, 7:10] = 40.0

        self.assertGreater(prop.crop_dark_signal(crop), 1.0)

    def test_control_offsets_are_deterministic(self):
        offsets = prop.control_offsets(10.0, 8)

        self.assertEqual(len(offsets), 8)
        self.assertAlmostEqual(offsets[0][0], 10.0)
        self.assertAlmostEqual(offsets[0][1], 0.0)


if __name__ == "__main__":
    unittest.main()

