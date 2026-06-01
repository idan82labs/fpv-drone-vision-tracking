import unittest

from scripts.selector_core import (
    SequenceItem,
    StreamingViterbiSelector,
    select_viterbi_sequence,
    surface_gate_low_confidence,
    surface_rescue_risk,
    trace_router_bucket,
)


def item(frame, x, y, score, payload=None):
    return SequenceItem(frame=frame, bbox=(x, y, 4, 4), score=score, payload=payload)


class SelectorCoreTest(unittest.TestCase):
    def test_full_path_selector_prefers_late_restart_without_backfill(self):
        selected = select_viterbi_sequence(
            [
                (1, [item(1, 0, 0, 9.0, "old")]),
                (2, [item(2, 100, 100, 20.0, "late")]),
            ],
            max_jump_px=5.0,
            transition_weight=1.5,
        )

        self.assertNotIn(1, selected)
        self.assertEqual(selected[2].payload, "late")

    def test_full_path_selector_can_backfill_unreachable_prefix(self):
        selected = select_viterbi_sequence(
            [
                (1, [item(1, 0, 0, 9.0, "old")]),
                (2, [item(2, 100, 100, 20.0, "late")]),
            ],
            max_jump_px=5.0,
            transition_weight=1.5,
            backfill_unreachable=True,
        )

        self.assertEqual(selected[1].payload, "old")
        self.assertEqual(selected[2].payload, "late")

    def test_size_jump_weight_allows_large_candidate_motion(self):
        selected = select_viterbi_sequence(
            [
                (1, [SequenceItem(frame=1, bbox=(10, 10, 20, 20), score=0.9, payload="first")]),
                (
                    2,
                    [
                        SequenceItem(frame=2, bbox=(23, 10, 20, 20), score=0.9, payload="large"),
                        SequenceItem(frame=2, bbox=(18, 18, 4, 4), score=0.85, payload="small"),
                    ],
                ),
            ],
            max_jump_px=4.0,
            transition_weight=0.0,
            size_jump_weight=0.5,
            backfill_unreachable=True,
        )

        self.assertEqual(selected[2].payload, "large")

    def test_streaming_oldest_selection_does_not_emit_late_restart(self):
        core = StreamingViterbiSelector(max_jump_px=5.0, transition_weight=1.5)
        core.add_layer(1, [item(1, 0, 0, 9.0, "old")])
        core.add_layer(2, [item(2, 100, 100, 20.0, "late")])

        frame, selected, score, path = core.first_selection()

        self.assertEqual(frame, 1)
        self.assertIsNone(selected)
        self.assertIsNone(score)
        self.assertEqual(path, [])

    def test_streaming_oldest_selection_emits_reachable_path(self):
        core = StreamingViterbiSelector(max_jump_px=5.0, transition_weight=1.5)
        core.add_layer(1, [item(1, 0, 0, 9.0, "first")])
        core.add_layer(2, [item(2, 2, 0, 8.0, "second")])

        frame, selected, score, path = core.first_selection()

        self.assertEqual(frame, 1)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.payload, "first")
        self.assertEqual(score, 9.0)
        self.assertEqual(path, [0, 0])

    def test_commit_prefix_prunes_remaining_layers(self):
        core = StreamingViterbiSelector(max_jump_px=5.0, transition_weight=1.5)
        core.add_layer(1, [item(1, 0, 0, 9.0, "a1"), item(1, 60, 0, 7.0, "b1")])
        core.add_layer(2, [item(2, 2, 0, 8.0, "a2"), item(2, 62, 0, 7.0, "b2")])
        core.add_layer(3, [item(3, 4, 0, 6.0, "a3"), item(3, 64, 0, 7.0, "b3")])

        _frame, selected, _score, path = core.first_selection()
        self.assertIsNotNone(selected)
        core.pop_first()
        core.commit_prefix(path[1:])

        self.assertEqual([layer.items[0].payload for layer in core.layers], ["a2", "a3"])

    def test_shared_surface_gate_normalizes_router_and_low_confidence(self):
        self.assertEqual(trace_router_bucket({"router_bucket": "line_attached"}), "line")
        active, reason = surface_gate_low_confidence(
            {
                "state": "T",
                "selected": "1",
                "rank": "12",
                "target_margin": "2.0",
                "router_bucket": "surface_backed",
            },
            gate_states={"A", "P", "C", "S", "E"},
            gate_routers={"surface", "line", "boundary", "unknown"},
            gate_rank_min=10,
            gate_margin_max=0.8,
            gate_raw_score_max=-999.0,
        )

        self.assertTrue(active)
        self.assertIn("rank_ge_10", reason)

    def test_shared_surface_risk_blocks_easy_clean_track(self):
        score, reason = surface_rescue_risk(
            {"state": "T", "selected": "1", "rank": "1", "target_margin": "2.0", "router_bucket": "clean_sky"},
            [],
            [{"cand_texture": "80"}],
        )

        self.assertLess(score, 0.0)
        self.assertIn("router_clean", reason)


if __name__ == "__main__":
    unittest.main()
