import csv
import tempfile
import unittest
from pathlib import Path

from scripts import analyze_selector_disagreements as disagreements
from scripts import evaluate_mode_supervisor as supervisor
from scripts import seed_selector_disagreement_review_labels as seed_review


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "selected", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)


class SelectorCsvFallbackTests(unittest.TestCase):
    def test_seed_review_labels_map_binary_router_targets(self):
        self.assertEqual(
            seed_review.router_label({"category": "a_visible_hit_b_miss", "visible": "1"}),
            "protect_continuous_visible",
        )
        self.assertEqual(seed_review.binary_mode_target("protect_continuous_visible"), "viterbi")
        self.assertEqual(seed_review.router_label({"category": "a_false_b_suppressed", "visible": "0"}), "hard_null_use_hmm")
        self.assertEqual(seed_review.binary_mode_target("hard_null_use_hmm"), "hmm")
        self.assertEqual(
            seed_review.router_label({"category": "both_null_false_box", "visible": "0"}),
            "hard_null_needs_override",
        )
        self.assertEqual(seed_review.binary_mode_target("hard_null_needs_override"), "")

    def test_supervisor_crop_features_are_explicit(self):
        self.assertNotIn("crop_stack_score", supervisor.active_base_features(False))
        self.assertIn("crop_stack_score", supervisor.active_base_features(True))

    def test_supervisor_branch_context_is_explicit(self):
        selected = {
            7: {
                "frame": "7",
                "selected": "1",
                "rank": "2",
                "learned_score": "0.75",
                "x": "10",
                "y": "12",
                "w": "4",
                "h": "4",
            }
        }
        ranked = {
            7: [
                {"rank": "1", "x": "1", "y": "1", "w": "4", "h": "4", "crop_stack_score": "0.1"},
                {"rank": "2", "x": "10", "y": "12", "w": "4", "h": "4", "crop_stack_score": "0.9"},
            ]
        }
        without_context = supervisor.branch_features(
            selected,
            ranked,
            [7],
            "viterbi",
            supervisor.active_base_features(True),
            include_context=False,
        )
        with_context = supervisor.branch_features(
            selected,
            ranked,
            [7],
            "viterbi",
            supervisor.active_base_features(True),
            include_context=True,
        )

        self.assertNotIn("viterbi_sel_crop_stack_score", without_context[7])
        self.assertEqual(with_context[7]["viterbi_sel_crop_stack_score"], 0.9)

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

    def test_supervisor_reviewed_examples_use_binary_mode_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "clip",
                        "frame",
                        "binary_mode_target",
                        "router_label",
                        "false_lock_kind",
                        "review_confidence",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "clip": "clip_a",
                            "frame": "1",
                            "binary_mode_target": "hmm",
                            "router_label": "hard_null_use_hmm",
                            "false_lock_kind": "cloud_sky_speck",
                            "review_confidence": "high",
                        },
                        {
                            "clip": "clip_a",
                            "frame": "2",
                            "binary_mode_target": "viterbi",
                            "router_label": "protect_continuous_visible",
                            "false_lock_kind": "visible_mode_error",
                            "review_confidence": "high",
                        },
                        {
                            "clip": "clip_a",
                            "frame": "3",
                            "binary_mode_target": "",
                            "router_label": "hard_null_needs_override",
                            "false_lock_kind": "tree_edge",
                            "review_confidence": "medium_high",
                        },
                    ]
                )

            examples, y, clips, frames = supervisor.reviewed_examples(
                path,
                {("clip_a", 1): {}, ("clip_a", 2): {}, ("clip_a", 3): {}},
            )

            self.assertEqual([row["category"] for row in examples], ["hard_null_use_hmm", "protect_continuous_visible"])
            self.assertEqual(y.tolist(), [1, 0])
            self.assertEqual(clips.tolist(), ["clip_a", "clip_a"])
            self.assertEqual(frames.tolist(), [1, 2])

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
