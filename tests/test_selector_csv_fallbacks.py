import csv
import tempfile
import unittest
from pathlib import Path

from scripts import analyze_selector_disagreements as disagreements
from scripts import evaluate_mode_supervisor as supervisor


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "selected", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)


class SelectorCsvFallbackTests(unittest.TestCase):
    def test_disagreement_reader_accepts_sequence_selected_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(
                root / "clip_a" / "sequence_selected_tracks.csv",
                [
                    {"frame": "10", "selected": "1", "x": "20", "y": "30", "w": "4", "h": "5"},
                    {"frame": "11", "selected": "0", "x": "40", "y": "50", "w": "4", "h": "5"},
                ],
            )

            selected = disagreements.selected_by_frame(root, "clip_a")

            self.assertEqual(sorted(selected), [10])
            self.assertEqual(selected[10]["x"], "20")

    def test_supervisor_reader_accepts_sequence_selected_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(
                root / "clip_b" / "sequence_selected_tracks.csv",
                [
                    {"frame": "4", "selected": "1", "x": "7", "y": "8", "w": "3", "h": "3"},
                ],
            )

            selected = supervisor.selected_by_frame(root, "clip_b")

            self.assertEqual(sorted(selected), [4])
            self.assertEqual(selected[4]["y"], "8")


if __name__ == "__main__":
    unittest.main()
