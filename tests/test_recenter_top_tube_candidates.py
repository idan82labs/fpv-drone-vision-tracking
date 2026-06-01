import argparse
import unittest

import numpy as np

from scripts import recenter_top_tube_candidates as recenter


class RecenterTopTubeCandidatesTests(unittest.TestCase):
    def test_recenter_moves_loose_candidate_to_dark_high_res_peak(self):
        gray = np.full((80, 80), 150, dtype=np.uint8)
        yy, xx = np.mgrid[:80, :80]
        dot = np.hypot(xx - 42, yy - 39) <= 2.0
        gray[dot] = 35
        score_maps = {2: recenter.compact_dark_map(gray, radius=2, texture_weight=0.0)}
        row = {
            "frame": "10",
            "rank": "1",
            "track_id": "a",
            "x": "16.5",
            "y": "15.5",
            "w": "3",
            "h": "3",
            "score": "0.0",
            "verified_score": "0.0",
        }

        outs = recenter.recenter_row(
            row,
            score_maps,
            detector_scale=0.5,
            search_radius_det_px=8.0,
            box_size_det_px=3.0,
            recenter_score_weight=1.0,
            shift_penalty=0.0,
            peaks_per_candidate=1,
            grid_step_det_px=0.0,
            grid_per_candidate=0,
        )

        self.assertEqual(len(outs), 1)
        out = outs[0]
        self.assertIsNotNone(out)
        cx = recenter.safe_float(out["x"]) + 0.5 * recenter.safe_float(out["w"])
        cy = recenter.safe_float(out["y"]) + 0.5 * recenter.safe_float(out["h"])
        self.assertLess(abs(cx - 21.0), 0.6)
        self.assertLess(abs(cy - 19.5), 0.6)
        self.assertEqual(out["proposal_variant"], "recenter_highres_dark_ring")

    def test_dedupe_and_rank_prefers_highest_score_near_same_center(self):
        rows = [
            {"frame": "1", "rank": "7", "x": "10", "y": "10", "w": "3", "h": "3", "recenter_combined_score": "1.0"},
            {"frame": "1", "rank": "8", "x": "10.5", "y": "10.25", "w": "3", "h": "3", "recenter_combined_score": "3.0"},
            {"frame": "1", "rank": "9", "x": "30", "y": "30", "w": "3", "h": "3", "recenter_combined_score": "2.0"},
        ]

        kept = recenter.dedupe_and_rank(rows, nms_det_px=2.5, max_output=10)

        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["rank"], "1")
        self.assertEqual(kept[0]["x"], "10.5")
        self.assertEqual(kept[1]["rank"], "2")

    def test_recenter_rows_preserves_originals_and_adds_variant(self):
        gray = np.full((60, 60), 150, dtype=np.uint8)
        gray[30:34, 30:34] = 30
        args = argparse.Namespace(
            keep_originals=True,
            texture_weight=0.0,
            detector_scale=0.5,
            search_radius_det_px=6.0,
            box_size_det_px=3.0,
            recenter_score_weight=1.0,
            shift_penalty=0.0,
            max_recenter_per_frame=10,
            peaks_per_candidate=1,
            grid_step_det_px=0.0,
            grid_per_candidate=0,
            nms_det_px=0.1,
            max_output_per_frame=10,
            router_include="",
            max_rank=80,
            search_radius_det_px_unused=None,
        )
        rows = [
            {
                "frame": "1",
                "rank": "1",
                "track_id": "a",
                "x": "12",
                "y": "12",
                "w": "3",
                "h": "3",
                "verified_score": "0.0",
                "cand_router_state": "surface_backed",
            }
        ]

        out, summary = recenter.recenter_rows(rows, {1: gray}, [2], args)

        variants = {r.get("proposal_variant") for r in out}
        self.assertIn("original", variants)
        self.assertIn("recenter_highres_dark_ring", variants)
        self.assertEqual(summary["recentered_rows_before_nms"], 1)


if __name__ == "__main__":
    unittest.main()
