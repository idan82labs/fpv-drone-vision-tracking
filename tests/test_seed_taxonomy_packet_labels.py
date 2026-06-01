import unittest

from scripts import seed_taxonomy_packet_labels as seed


class SeedTaxonomyPacketLabelsTests(unittest.TestCase):
    def test_geometry_target_and_near_target_labels(self):
        labels = {("clip", 10): (True, (10.0, 10.0, 4.0, 4.0))}

        target, target_reason = seed.seed_row(
            {"clip": "clip", "frame": "10", "bbox": "[10, 10, 4, 4]"},
            labels,
            8.0,
            16.0,
            24.0,
            0.62,
        )
        near, near_reason = seed.seed_row(
            {"clip": "clip", "frame": "10", "bbox": "[24, 10, 4, 4]"},
            labels,
            8.0,
            16.0,
            24.0,
            0.62,
        )

        self.assertEqual(target, "target")
        self.assertIn("geometry_target", target_reason)
        self.assertEqual(near, "near_target_wrong_center")
        self.assertIn("geometry_near", near_reason)

    def test_model_clutter_label_uses_confident_generic(self):
        label, reason = seed.model_clutter_label(
            {
                "crop_s_prob": "0.02",
                "crop_e_prob": "0.03",
                "crop_h_prob": "0.04",
                "crop_g_prob": "0.91",
                "crop_pred_class": "G",
            },
            0.62,
        )

        self.assertEqual(label, "noise")
        self.assertIn("model_prob", reason)

    def test_model_clutter_label_falls_back_to_texture(self):
        label, reason = seed.model_clutter_label(
            {
                "candidate_source": "large_dark",
                "cand_texture": "60",
                "crop_s_prob": "0.2",
                "crop_e_prob": "0.2",
                "crop_h_prob": "0.2",
                "crop_g_prob": "0.2",
            },
            0.62,
        )

        self.assertEqual(label, "terrain_texture")
        self.assertEqual(reason, "large_dark_textured")


if __name__ == "__main__":
    unittest.main()
