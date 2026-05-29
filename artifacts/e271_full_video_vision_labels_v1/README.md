# e271 Full-Video Vision Labels v1

Vision-assisted labels for `e2711620-6d4e-4f9c-8922-b1b2d1fb74f2.MP4`.

## Contents

- `e271_full_video_vision_labels_v1.csv` - one row per source frame, frames `0..698`.
- `contact_labels_every20_tailfix.jpg` - full-frame visual audit sampled every 20 frames, plus tail checks.
- `tail_corrected_zoom_contact.jpg` - zoomed audit for the corrected `580..698` tail segment.

## Scope

This clip is treated as visible-target throughout the full video. It is useful for continuity/ranking evaluation on a surface-backed target, especially the late ridge/grass segment. It is not a no-target false-positive benchmark because it does not contain confirmed empty frames.

## Label Method

The labels are manual/vision keyframe annotations with interpolation between keyframes. The early and middle segment uses manually reviewed keyframes. The `580..698` tail was corrected after the inherited labels drifted right of the drone around frames `660..698`.

Confidence is stored per row. Frames around `460..500` remain lower confidence because the target is faint against textured terrain. Use these labels as a practical training/evaluation set, not as a pixel-perfect ground-truth dataset.
