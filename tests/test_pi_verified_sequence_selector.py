import csv
import tempfile
from pathlib import Path

from raspberry_pi_runtime.verified_sequence_selector import load_rows, output_rows, viterbi_select, write_csv
from scripts.selector_core import SequenceItem, select_viterbi_sequence
import unittest


def candidate(frame, rank, x, y, score):
    return {
        "frame": frame,
        "rank": rank,
        "x": x,
        "y": y,
        "w": 4,
        "h": 4,
        "selector_score": score,
        "verified_score": score,
        "cand_source": "test",
        "track_id": rank,
    }


class VerifiedSequenceSelectorTest(unittest.TestCase):
    def test_viterbi_rejects_unreachable_late_jump_prefix(self):
        by_frame = {
            1: [candidate(1, 1, 0, 0, 9.0)],
            2: [candidate(2, 1, 100, 100, 20.0)],
        }

        selected = viterbi_select(by_frame, max_jump_px=5.0, transition_weight=1.5)

        self.assertNotIn(1, selected)
        self.assertEqual(selected[2]["x"], 100)

    def test_output_rows_omits_unselected_blank_boxes(self):
        by_frame = {
            1: [candidate(1, 1, 0, 0, -1.0)],
            2: [candidate(2, 1, 2, 0, 4.0)],
        }
        selected = {1: by_frame[1][0], 2: by_frame[2][0]}

        rows = output_rows("clip", by_frame, selected, threshold=0.0)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], 2)
        self.assertEqual(rows[0]["selected"], 1)
        self.assertEqual(rows[0]["x"], 2)

    def test_output_rows_can_emit_no_box_rows_for_parity(self):
        by_frame = {
            1: [candidate(1, 1, 0, 0, -1.0)],
            2: [candidate(2, 1, 2, 0, 4.0)],
            3: [candidate(3, 1, 100, 100, 10.0)],
        }
        selected = {1: by_frame[1][0], 2: by_frame[2][0]}

        rows = output_rows("clip", by_frame, selected, threshold=0.0, emit_no_box_rows=True)

        self.assertEqual([row["frame"] for row in rows], [1, 2, 3])
        self.assertEqual([row["selected"] for row in rows], [0, 1, 0])
        self.assertEqual(rows[0]["x"], "")
        self.assertEqual(rows[2]["x"], "")

    def test_viterbi_matches_selector_core_no_backfill_selected_no_box_parity(self):
        by_frame = {
            1: [candidate(1, 1, 0, 0, 9.0)],
            2: [candidate(2, 1, 100, 100, 20.0)],
            3: [candidate(3, 1, 102, 100, 15.0)],
        }

        selected = viterbi_select(by_frame, max_jump_px=5.0, transition_weight=1.5)
        core_selected = select_viterbi_sequence(
            [
                (
                    frame,
                    [
                        SequenceItem(
                            frame=frame,
                            bbox=(row["x"], row["y"], row["w"], row["h"]),
                            score=row["selector_score"],
                            payload=row,
                        )
                        for row in rows
                    ],
                )
                for frame, rows in sorted(by_frame.items())
            ],
            max_jump_px=5.0,
            transition_weight=1.5,
        )
        core_selected_payloads = {
            frame: item.payload
            for frame, item in core_selected.items()
            if isinstance(item.payload, dict)
        }
        rows = output_rows("clip", by_frame, selected, threshold=0.0, emit_no_box_rows=True)

        self.assertEqual(selected, core_selected_payloads)
        self.assertEqual([row["frame"] for row in rows], [1, 2, 3])
        self.assertEqual([row["selected"] for row in rows], [0, 1, 1])
        self.assertEqual(rows[0]["x"], "")
        self.assertEqual(rows[1]["track_id"], 1)

    def test_load_rows_filters_detector_ineligible_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "top_tubes.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["frame", "rank", "x", "y", "w", "h", "verified_score", "eligible", "passes_floor"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "frame": 1,
                        "rank": 1,
                        "x": 0,
                        "y": 0,
                        "w": 4,
                        "h": 4,
                        "verified_score": 10,
                        "eligible": 0,
                        "passes_floor": 1,
                    }
                )
                writer.writerow(
                    {
                        "frame": 1,
                        "rank": 2,
                        "x": 2,
                        "y": 0,
                        "w": 4,
                        "h": 4,
                        "verified_score": 5,
                        "eligible": 1,
                        "passes_floor": 1,
                    }
                )

            rows = load_rows(path, max_rank=20, score_column="verified_score")

            self.assertEqual(len(rows[1]), 1)
            self.assertEqual(rows[1][0]["rank"], 2)

    def test_write_csv_keeps_header_for_empty_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.csv"

            write_csv(path, [], ["clip", "frame", "selected"])

            self.assertEqual(path.read_text().strip(), "clip,frame,selected")


if __name__ == "__main__":
    unittest.main()
