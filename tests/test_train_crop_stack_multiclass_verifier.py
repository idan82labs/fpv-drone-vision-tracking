import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import train_crop_stack_multiclass_verifier as multi


class MultiClassCropStackVerifierTests(unittest.TestCase):
    def test_weak_clutter_class_prefers_target_for_positive(self):
        self.assertEqual(multi.weak_clutter_class({"hard_label": "1"}), "T")

    def test_weak_clutter_class_splits_boundary_attached_static(self):
        boundary = {
            "hard_label": "0",
            "cand_router_state": "boundary_mixed",
            "tube_router_boundary_rate": "0.8",
        }
        attached = {
            "hard_label": "0",
            "cand_router_state": "line_attached",
            "cand_line_context": "0.9",
            "cand_attached_support": "15",
        }
        static = {
            "hard_label": "0",
            "clba_bg_static_likelihood": "2.0",
            "clba_bg_q": "1.5",
            "clba_path_bg_dist_mean": "0.5",
        }

        self.assertEqual(multi.weak_clutter_class(boundary), "H")
        self.assertEqual(multi.weak_clutter_class(attached), "E")
        self.assertEqual(multi.weak_clutter_class(static), "S")

    def test_weak_clutter_class_keeps_ambiguous_negative_generic(self):
        self.assertEqual(multi.weak_clutter_class({"hard_label": "0", "cand_router_state": "unknown"}), "G")

    def test_normalize_taxonomy_label_maps_review_labels(self):
        self.assertEqual(multi.normalize_taxonomy_label("target"), "T")
        self.assertEqual(multi.normalize_taxonomy_label("static_hotspot"), "S")
        self.assertEqual(multi.normalize_taxonomy_label("attached_tree_branch_terrain"), "E")
        self.assertEqual(multi.normalize_taxonomy_label("skyline_boundary_parallax"), "H")
        self.assertEqual(multi.normalize_taxonomy_label("noise"), "G")
        self.assertIsNone(multi.normalize_taxonomy_label("near_target_wrong_center"))
        self.assertEqual(multi.normalize_taxonomy_label(""), "")

    def test_load_taxonomy_labels_uses_human_fallback_and_ignores_uncertain(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text(
                "clip,frame,rank,taxonomy_label,human_label\n"
                "c,1,2,,target\n"
                "c,1,3,uncertain,\n"
                "c,1,4,terrain_texture,\n"
            )

            labels, counts = multi.load_taxonomy_labels(path, "taxonomy_label")

        self.assertEqual(labels[("c", 1, 2)], "T")
        self.assertIsNone(labels[("c", 1, 3)])
        self.assertEqual(labels[("c", 1, 4)], "E")
        self.assertEqual(counts["T"], 1)
        self.assertEqual(counts["ignored"], 1)

    def test_append_taxonomy_examples_adds_unsampled_rank(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_dir = root / "results" / "clip"
            clip_dir.mkdir(parents=True)
            (clip_dir / "top_tubes.csv").write_text(
                "clip,frame,rank,x,y,w,h,score\n"
                "clip,10,7,1,2,3,4,0.5\n"
            )
            packet = root / "packet.csv"
            packet.write_text(
                "clip,frame,rank,taxonomy_label\n"
                "clip,10,7,target\n"
            )
            labels, _ = multi.load_taxonomy_labels(packet, "taxonomy_label")
            examples: list[dict[str, str]] = []

            added = multi.append_taxonomy_examples(examples, packet, labels, root / "results", 80)

        self.assertEqual(added, 1)
        self.assertEqual(examples[0]["hard_label"], "1")
        self.assertEqual(examples[0]["hard_kind"], "taxonomy_packet_appended")
        self.assertEqual(examples[0]["x"], "1")

    def test_pairwise_t_summary_uses_target_logit(self):
        rows = [
            {"clip": "c", "frame": "1", "weak_class": "T", "crop_t_logit": "2"},
            {"clip": "c", "frame": "1", "weak_class": "E", "crop_t_logit": "1"},
            {"clip": "c", "frame": "2", "weak_class": "T", "crop_t_logit": "0"},
            {"clip": "c", "frame": "2", "weak_class": "S", "crop_t_logit": "3"},
        ]

        summary = multi.pairwise_t_summary(rows)

        self.assertEqual(summary["pairwise_total"], 2)
        self.assertEqual(summary["pairwise_wins"], 1)
        self.assertEqual(summary["pairwise_win_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
