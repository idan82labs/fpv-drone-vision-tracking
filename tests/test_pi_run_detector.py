import tempfile
import unittest
import json
from pathlib import Path

from raspberry_pi_runtime.benchmark_pi_profiles import selected_jsonl_to_csv
from raspberry_pi_runtime.run_pi_detector import read_csv, render_selection_csv, write_csv
from scripts.tbd_motion_detector import video_capture_source


class PiRunDetectorTest(unittest.TestCase):
    def test_render_selection_csv_converts_detector_selected_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected_tracks.csv"
            write_csv(
                selected,
                [
                    {
                        "frame": 7,
                        "track_id": 3,
                        "x": 11,
                        "y": 12,
                        "w": 4,
                        "h": 5,
                        "score": 9.25,
                        "misses": 0,
                    }
                ],
            )

            out = render_selection_csv(selected, root / "render.csv", "clip-id")
            rows = read_csv(out)

            self.assertEqual(out.name, "render.csv")
            self.assertEqual(rows[0]["clip"], "clip-id")
            self.assertEqual(rows[0]["selected"], "1")
            self.assertEqual(rows[0]["rank"], "1")
            self.assertEqual(rows[0]["learned_score"], "9.25")

    def test_render_selection_csv_keeps_header_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected_tracks.csv"
            write_csv(selected, [], ["frame", "track_id", "x", "y", "w", "h", "score", "misses"])

            out = render_selection_csv(selected, root / "render.csv", "clip-id")

            self.assertIn("clip,frame,selected", out.read_text())

    def test_video_capture_source_accepts_camera_aliases(self):
        self.assertEqual(video_capture_source("0"), 0)
        self.assertEqual(video_capture_source("camera:1"), 1)
        self.assertEqual(video_capture_source("/tmp/0.mp4"), "/tmp/0.mp4")
        with self.assertRaises(ValueError):
            video_capture_source("camera:not-a-number")

    def test_selected_jsonl_to_csv_preserves_frame_and_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "selected.jsonl"
            jsonl.write_text(
                json.dumps(
                    {
                        "frame": 12,
                        "track_id": 4,
                        "bbox": [7, 8, 3, 2],
                        "score": 6.5,
                        "verified_score": 7.25,
                    }
                )
                + "\n"
            )

            out = selected_jsonl_to_csv(jsonl, root / "selected.csv", "clip-a")
            rows = read_csv(out)

            self.assertEqual(rows[0]["clip"], "clip-a")
            self.assertEqual(rows[0]["frame"], "12")
            self.assertEqual(rows[0]["x"], "7")
            self.assertEqual(rows[0]["learned_score"], "7.25")

    def test_selected_jsonl_to_csv_writes_header_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = selected_jsonl_to_csv(root / "missing.jsonl", root / "selected.csv", "clip-a")

            self.assertIn("clip,frame,selected", out.read_text())


if __name__ == "__main__":
    unittest.main()
