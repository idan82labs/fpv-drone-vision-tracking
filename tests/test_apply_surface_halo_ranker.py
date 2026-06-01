import unittest

import numpy as np

from scripts import apply_surface_halo_ranker as apply


class DummyModel:
    def predict_proba(self, x):
        # Score is the first feature scaled into [0, 1].
        p = np.clip(x[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class ApplySurfaceHaloRankerTests(unittest.TestCase):
    def test_score_rows_uses_bundle_feature_schema(self):
        rows = [
            {"rank": "1", "feature_a": "0.2", "cand_source": "map", "proposal_variant": "original"},
            {"rank": "2", "feature_a": "0.8", "cand_source": "recenter", "proposal_variant": "recentered"},
        ]
        bundle = {
            "model": DummyModel(),
            "numeric_features": ["feature_a"],
            "source_features": ["src_map", "src_recenter"],
            "variant_features": ["variant_original", "variant_recentered"],
            "best_model_interleaved": "dummy",
        }

        scored = apply.score_rows(rows, bundle, overwrite_scores=True)

        self.assertEqual(scored[0]["surface_halo_score"], 0.2)
        self.assertEqual(scored[1]["surface_halo_score"], 0.8)
        self.assertEqual(scored[1]["score"], 0.8)
        self.assertEqual(scored[1]["surface_halo_model"], "dummy")

    def test_rerank_by_frame_keeps_top_n_and_parent_rank(self):
        rows = [
            {"frame": "1", "rank": "7", "surface_halo_score": "0.1"},
            {"frame": "1", "rank": "8", "surface_halo_score": "0.9"},
            {"frame": "1", "rank": "9", "surface_halo_score": "0.8"},
        ]

        ranked = apply.rerank_by_frame(rows, top_per_frame=2)

        self.assertEqual([r["surface_halo_parent_rank"] for r in ranked], ["8", "9"])
        self.assertEqual([r["rank"] for r in ranked], ["1", "2"])

    def test_rerank_can_mark_exported_rows_selected(self):
        rows = [{"frame": "1", "rank": "3", "surface_halo_score": "0.7", "selected": "0"}]

        ranked = apply.rerank_by_frame(rows, top_per_frame=1, mark_selected=True)

        self.assertEqual(ranked[0]["selected"], "1")

    def test_blank_clip_rows_match_requested_clip(self):
        self.assertTrue(apply.row_matches_clip({"clip": ""}, "clip-a"))
        self.assertTrue(apply.row_matches_clip({"clip": "clip-a"}, "clip-a"))
        self.assertFalse(apply.row_matches_clip({"clip": "clip-b"}, "clip-a"))


if __name__ == "__main__":
    unittest.main()
