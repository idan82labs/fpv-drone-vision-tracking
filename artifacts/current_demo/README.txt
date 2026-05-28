Drone-Strike v4 proposal-recovery notes
Generated 2026-05-29

Scope
-----
This package continues from v3 tail-extended labels. It does not replace the
v3 videos as a better autonomous demo. It adds the next algorithm experiment:
direct temporal-stack proposal recovery on the e271 tail, where v3 exposed many
visible-drone misses.

Files copied here
-----------------
- v3 tail labels and selection CSVs
- v3 accepted overlay sheets
- e271 proposal-recovery summaries and failure sheets
- integrated TBD summaries/ranker audits for composed, direct, union, and
  temporal-stack-only variants

Main result
-----------
The new temporal-stack proposal layer can see more of the e271 tail, but the
current beam/ranker still chooses clutter.

Standalone proposal recovery on e271 v3 labels:
- best source: temporal_dark, past-only wide offsets
- R@80 = 0.539 overall
- R@200 = 0.730 overall
- R@500 = 0.910 overall
- high-confidence R@80 = 0.755

Integrated TBD audits on e271 labels:
- composed temporal stack:
  - oracle = 0.517 overall, 0.122 high-confidence
  - ExtraTrees strict = 37/89, loose = 44/89
  - runtime ~= 187.5 ms/frame
- direct-warp temporal stack:
  - oracle = 0.629 overall, 0.327 high-confidence
  - ExtraTrees strict = 28/89, loose = 43/89
  - runtime ~= 207.3 ms/frame
- direct + composed union:
  - oracle = 0.629 overall, 0.327 high-confidence
  - ExtraTrees strict = 34/89, loose = 43/89
- direct temporal-stack-only, wider beam:
  - oracle = 0.685 overall, 0.510 high-confidence
  - ExtraTrees strict = 16/89, loose = 33/89
  - runtime ~= 379.2 ms/frame

Interpretation
--------------
This is progress on proposal recovery, not progress on final autonomous
tracking. More of the target is present somewhere in the proposal pool, but the
path scorer prefers cloud/terrain/skyline clutter. Simple raw-proposal features
(rank, score, radius, position) are not enough; a quick cross-validated check
only reached about 10/89 strict even though oracle@500 was 0.91.

Next meaningful algorithm step
------------------------------
Train or implement crop/tube visual evidence:
- target-aligned crop stack versus background-aligned crop stack
- local matched-filter/dipole residual normalized by local MAD
- hard-negative classes from the failure sheets: cloud specks, terrain marks,
  skyline boundary points, and static hot spots

Do not spend more time broadening thresholds or just increasing candidate count.
The candidate pool already contains useful target proposals; the current blocker
is selecting the target tube over visually plausible clutter.

Honest status
-------------
The v3 visual labels/videos are useful for presentation and training data. The
v4 proposal work is a research diagnostic. It is not yet a better real-time
tracker and is too slow for onboard use in its direct-warp form.

Vision-assisted e271 demo addendum
----------------------------------
After the v4 proposal-recovery work, I added a separate vision-assisted label
run for the visible e271 tail section before the prior lock. This is not an
autonomous detector result. It is a cleaned target track made from the visible
frames so the team can see what the tracker should learn to recover.

New files:
- e271_vision_assisted_clean_9p0-12p8s_v2.mp4
  Best team-facing cut. The target is visible and the box stays on the quad
  without the previous dropout.
- e271_vision_assisted_medium_9p0-13p2s_v2.mp4
  Longer cut; still mostly clean, but the tail gets lower contrast.
- e271_vision_assisted_full_9p0-14p0s_v2_clean.mp4
  Full diagnostic cut. Useful for inspection, less clean for presentation.
- e271_vision_assisted_450_698_selection_v2_gapfilled.csv
  Selection data used for the rendered videos. Frames 606-621 were filled after
  visual inspection confirmed the target was plainly visible through the gap.

Use the v2 gap-filled file as weak/vision-assisted training data only. Keep it
separate from stricter human-reviewed labels unless manually rechecked frame by
frame.
