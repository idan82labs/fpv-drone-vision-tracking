# Learned Verifier Calibration

Scores: `results/tube_verifier_final_extra_trees_applied/learned_tube_scores.csv`
Labels: `results/tube_alternative_review_packet_top16/tube_alternatives_to_label.csv`
Full scored frames: 5876
Packet labeled rows: 268
Packet checkpoint frames: 35

Packet-balanced threshold from current local labels:
`0.55`
`0.55`: packet 12/15 target hits, 0 wrong target-frame selections, 20/20 no-target frames suppressed; full export selects 1814/5876 frames

Strict full-topK checkpoint threshold:
`0.6`
`0.6`: packet 11/15 target hits, 0 wrong target-frame selections, 20/20 no-target frames suppressed; full export selects 1396/5876 frames

This is a calibration report, not proof of dense full-video accuracy. Full-video rows outside the review packet remain unlabeled.

See `threshold_sweep.csv` for packet-only and full-topK metrics.
