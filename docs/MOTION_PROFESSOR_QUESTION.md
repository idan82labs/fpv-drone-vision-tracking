# Moving-Camera Small-Object Detection Question

We are testing a class-agnostic computer-vision detector for small independently
moving objects in low-resolution outdoor video from a moving camera.

## Data

- 9 short MP4 clips, mostly 640x480 at about 50 Hz.
- Current processing resolution: 320x240 (`downscale=0.5`).
- Scene types: sky, hills, roads, vegetation, clouds, sun glare, strong parallax.
- Desired operational output for the algorithmic test: one bounding box on the
  independently moving object in about 80% of frames, at 30 Hz or better.
- No formal ground-truth labels yet. We are visually reviewing sampled overlays.

## Current Pipeline

1. Detect Shi-Tomasi corners in frame `t-1`.
2. Track to frame `t` with pyramidal Lucas-Kanade.
3. Reject tracks with forward/backward LK error.
4. Fit global camera motion with RANSAC:
   - partial affine
   - full affine
   - homography
   - auto-select by feature reprojection error and inlier ratio
5. Warp frame `t-1` into frame `t`.
6. Compute residual image.
7. Adaptive threshold residual with median/MAD and high percentile.
8. Connected components, shape/area/fill/aspect filtering.
9. Lightweight tracker with temporal persistence.
10. Fallback local-contrast appearance cue for stabilized objects whose
    inter-frame residual is weak.
11. Optional kinematic gating: reject track association and selected-box jumps
    that imply impossible image-plane displacement under assumed FOV, frame
    rate, max relative speed, and minimum range.

## Observed Results

Original simple affine residual baseline:

- Fast: about 0.7-1.6 ms/frame on Mac.
- Too noisy: some clips produced 20-42 candidate blobs/frame.

Current best compromise:

- Runtime: about 3-6 ms/frame on Mac.
- RANSAC inlier ratio: usually 0.90-0.99.
- Selected box continuity: about 98-100% on most clips.
- Candidate count: about 3.7-15/frame depending on clutter.
- Visual review: often catches the compact object, but can still select
  terrain/tree/cloud-edge clutter in textured scenes.

Kinematic-gate experiment:

- Assumption: max relative speed 10 m/s.
- At 320 px wide, 50 Hz, 120 deg horizontal FOV, min range 2 m:
  allowed object motion is about 9 px/frame before detector slack.
- A strict gate plus 8 px slack was too aggressive on clutter-heavy clips.
- A wider gate, about 24-29 px/frame depending on clip dimensions, preserved
  selected-box continuity near or above 80% in all test clips while rejecting
  many selected-box switches.
- This improves plausibility, but cannot reject a persistent false track on a
  static tree/terrain edge.

## Hypotheses To Challenge

1. A single homography is often too good at explaining the scene and can erase
   useful residual for stabilized or distant objects. It is clean, but it misses
   some visually obvious compact objects.
2. Partial/full affine leaves more residual motion, which helps detection but
   leaks parallax and terrain edge clutter.
3. Appearance-only local contrast rescues stabilized objects but creates false
   positives on small high-contrast background texture.
4. The right model may be a layered motion model: dominant background
   homography plus residual segmentation, or plane-plus-parallax, rather than a
   single global transform.
5. Without camera IMU/gyro, there may be an observability limit: small object
   motion and parallax clutter can be mathematically ambiguous at 320x240.

## Questions

1. For a moving monocular camera over non-planar terrain, what is the simplest
   model beyond one homography that meaningfully reduces parallax false
   positives without overfitting away small independently moving objects?
2. Would a two-layer RANSAC model, such as dominant homography plus local
   residual flow clustering, be mathematically preferable to full-frame
   homography/affine selection?
3. Is there a principled likelihood-ratio or Bayesian test we should use for:
   "this compact blob is independently moving" vs "this is registration error
   from parallax/rolling shutter/terrain texture"?
4. Given only grayscale video at 320x240 or 640x480 and no IMU, what lower bound
   can we estimate for detectable angular/object motion relative to camera
   jitter and scene parallax?
5. Would gyro-assisted rotation compensation plus translational parallax
   residual be expected to materially improve this class of footage compared
   with vision-only homography?
6. What is the right way to convert a physical speed constraint such as
   10 m/s into an image-plane gating prior when object range is unknown and
   camera ego-motion compensation is imperfect?

## Suggested Data To Send

- `results/motion_v2_relaxed_gate/summary_table.csv`
- `results/motion_v2_relaxed_gate/review_crops.jpg`
- `results/motion_v2_relaxed_gate/review_overlays.jpg`
- A few raw frames around both good catches and clutter catches.
