import unittest
from argparse import Namespace

import numpy as np

from scripts import make_surface_continuity_packet as packet


class SurfaceContinuityPacketTests(unittest.TestCase):
    def test_label_box_uses_det_coordinate_fallback(self):
        row = {"det_x": "10.5", "det_y": "20.0", "det_w": "3", "det_h": "4"}

        self.assertEqual(packet.label_box_fullres(row), (21, 40, 6, 8))

    def test_label_box_prefers_legacy_label_coordinates(self):
        row = {
            "label_x": "5",
            "label_y": "6",
            "label_w": "2",
            "label_h": "3",
            "det_x": "10",
            "det_y": "20",
            "det_w": "4",
            "det_h": "4",
        }

        self.assertEqual(packet.label_box_fullres(row), (10, 12, 4, 6))

    def test_crop_around_det_label_returns_image(self):
        frame = np.full((80, 100, 3), 127, dtype=np.uint8)
        row = {"det_x": "20", "det_y": "15", "det_w": "3", "det_h": "3"}

        crop = packet.crop_around_label(frame, row, half_w=20, half_h=15)

        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_select_rows_can_filter_clip_prefix(self):
        rows = [
            {"clip": "aa-111", "frame": "10", "bg_split": "textured_non_sky"},
            {"clip": "bb-222", "frame": "10", "bg_split": "textured_non_sky"},
        ]
        args = Namespace(
            clip="bb",
            split="textured_non_sky",
            start_frame=1,
            end_frame=20,
            frame_step=1,
            max_frames=10,
        )

        selected = packet.select_rows(rows, args)

        self.assertEqual([row["clip"] for row in selected], ["bb-222"])


if __name__ == "__main__":
    unittest.main()
