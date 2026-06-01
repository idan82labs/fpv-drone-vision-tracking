import unittest

from scripts.apply_pi_surface_false_lock_gate import gate_rows, surface_false_lock_risk, target_support_score


def row(frame, x=10, y=20, **kw):
    base = {
        "frame": str(frame),
        "selected": "1",
        "rank": "1",
        "track_id": str(frame),
        "x": str(x),
        "y": str(y),
        "w": "7",
        "h": "7",
        "verified_score": "10",
        "cand_source": "map",
        "cand_texture": "12",
        "cand_sky_like": "0.35",
        "tube_appearance_only_rate": "0",
        "tube_mean_texture": "12",
        "tube_mean_sky_like": "0.35",
        "tube_mean_pair_score": "1.4",
        "tube_positive_pair_rate": "0.8",
        "tube_mean_line_context": "0.05",
        "tube_mean_attached_support": "0",
    }
    base.update({k: str(v) for k, v in kw.items()})
    return base


class PiSurfaceFalseLockGateTests(unittest.TestCase):
    def test_false_lock_risk_marks_high_texture_appearance_only(self):
        risk, reason = surface_false_lock_risk(
            row(
                1,
                y=330,
                cand_source="appearance",
                cand_texture=80,
                cand_sky_like=0,
                tube_appearance_only_rate=1,
                tube_mean_texture=82,
                tube_mean_sky_like=0,
                tube_mean_pair_score=0.1,
                tube_positive_pair_rate=0.2,
            )
        )

        self.assertGreaterEqual(risk, 1.0)
        self.assertIn("high_texture", reason)
        self.assertIn("appearance_only", reason)

    def test_target_support_marks_map_sky_candidate(self):
        support, reason = target_support_score(row(1, cand_source="map", cand_texture=8, cand_sky_like=0.4))

        self.assertGreaterEqual(support, 1.0)
        self.assertIn("sky", reason)
        self.assertIn("map", reason)

    def test_gate_suppresses_surface_risk_without_recent_support(self):
        rows = [
            row(
                1,
                y=335,
                cand_source="appearance",
                cand_texture=85,
                cand_sky_like=0,
                tube_appearance_only_rate=1,
                tube_mean_texture=85,
                tube_mean_sky_like=0,
                tube_mean_pair_score=0.1,
                tube_positive_pair_rate=0.2,
            )
        ]

        accepted, decisions, events = gate_rows(rows, "clip")

        self.assertEqual(accepted, [])
        self.assertEqual(len(events), 1)
        self.assertEqual(decisions[0]["reason"], "surface_risk_without_recent_continuity")

    def test_gate_accepts_sky_then_suppresses_impossible_surface_jump(self):
        rows = [
            row(10, x=40, y=20, cand_source="map", cand_texture=8, cand_sky_like=0.45),
            row(
                11,
                x=420,
                y=330,
                cand_source="appearance",
                cand_texture=85,
                cand_sky_like=0,
                tube_appearance_only_rate=1,
                tube_mean_texture=85,
                tube_mean_sky_like=0,
                tube_mean_pair_score=0.1,
                tube_positive_pair_rate=0.2,
            ),
        ]

        accepted, _, events = gate_rows(rows, "clip", max_supported_jump_px=60)

        self.assertEqual([r["frame"] for r in accepted], [10])
        self.assertEqual(len(events), 1)

    def test_gate_keeps_plausible_continuation_near_recent_support(self):
        rows = [
            row(10, x=40, y=20, cand_source="map", cand_texture=8, cand_sky_like=0.45),
            row(
                11,
                x=45,
                y=25,
                cand_source="map",
                cand_texture=50,
                cand_sky_like=0,
                tube_appearance_only_rate=1,
                tube_mean_texture=50,
                tube_mean_sky_like=0,
                tube_mean_pair_score=0.5,
                tube_positive_pair_rate=0.5,
            ),
        ]

        accepted, _, events = gate_rows(rows, "clip", max_supported_jump_px=60)

        self.assertEqual([r["frame"] for r in accepted], [10, 11])
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
