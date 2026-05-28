# Follow-up on your local background parallax recommendation

Thank you. Your recommendation was directionally useful: it correctly identified that our global registration is usually not the main failure mode, and that the next problem is objectness against local parallax/clutter rather than a better full-frame homography.

I implemented several first-pass versions of your suggested idea and want to ask a sharper follow-up based on what failed empirically.

## What we tried from your answer

Baseline pipeline:

1. LK tracks between adjacent frames.
2. Auto-selected partial affine / full affine / homography with RANSAC.
3. Residual-image motion proposals.
4. Dark-blob appearance proposals.
5. Lightweight tracker with a soft kinematic gate.

Then I added variants inspired by your recommendation:

1. **Local residual-flow annulus score**
   - Compute residual flow after global warp.
   - For each candidate box, fit the local background residual from an annulus.
   - Compare inside-box residual flow against annulus residual flow.

2. **Local background compensation**
   - Use the annulus median residual flow to locally re-warp the already globally warped previous frame.
   - Score whether the candidate residual survives local background compensation.

3. **Stabilized-center motion**
   - Accumulate the frame-to-frame homographies into a stabilized reference frame.
   - Score tracks by whether selected centers move in stabilized coordinates.

4. **Short temporal median background**
   - Warp recent frames into the current view.
   - Use a median background residual as a multi-frame cue.

5. **Local structure/line-context penalty**
   - Penalize candidates embedded in a strongly one-dimensional local gradient field, e.g. pole tops, road edges, field rows.

## Empirical result

The advice helped the diagnosis, but the first implementations did not yet give a clean precision/recall improvement.

On a small 3-clip probe set with 17 provisional review checkpoints:

| Variant | Gold retained | Clutter suppressed | Empty suppressed | Miss recovered | Notes |
|---|---:|---:|---:|---:|---|
| Current baseline | 7/8 | 0/7 | 0/1 | 0/1 | Good recall, poor precision |
| Local residual-flow annulus | 6/8 | 0/7 | 0/1 | 0/1 | Flow inside tiny boxes unreliable |
| Local background compensation | 6/8 | 0/7 | 0/1 | 0/1 | Boosts some true boxes but clutter remains |
| Stabilized-center motion | 6/8 | 0/7 | 0/1 | 0/1 | Static clutter not reliably separated |
| Temporal median background | 5-6/8 | 1-2/7 | 0/1 | sometimes 1/1 | Adds many candidates and loses some true boxes |
| Local line/structure context | 7/8 | 0/7 | 1/1 | 0/1 | Helps one empty false positive only |

So the local background idea is not disproven, but our implementation is not enough.

## Observed failure mode

The target-like object is often only **3-10 px wide at 320x240**.

This creates a practical problem:

- Sparse or grid LK support inside the candidate box is usually background support, not object support.
- The false positives are also compact, dark, high-contrast, and sometimes temporally persistent.
- Appearance-only candidates often dominate the tracker.
- Hard appearance gates remove false positives but also lose true positives.
- Temporal median background helps a little, but parallax/rolling-shutter/model error creates too many new candidates.

In other words, comparing local flow inside the box to local flow in the annulus is fragile because the object is below reliable optical-flow support.

## Main follow-up question

Given this target scale, should we stop thinking in terms of **per-frame candidate classification** and switch to **track-before-detect in x-y-t**?

The version I am considering:

1. Ego-stabilize a short frame window with the current affine/homography model.
2. Build a likelihood map from residual intensity, dark-blob appearance, local texture/line penalties, and local background statistics.
3. Search for physically plausible trajectories over 5-15 frames using a soft velocity prior.
4. Emit one current-frame box from the best accumulated trajectory.

This would let weak evidence accumulate over time instead of requiring each 3-8 px object candidate to win a per-frame ranking contest.

## Specific questions

1. At 3-10 px target size, is local residual-flow likelihood inside the candidate box fundamentally the wrong primitive?

2. Would you recommend an x-y-t matched filter, Viterbi/dynamic programming, particle filter, or another track-before-detect formulation?

3. What should the likelihood model be when optical-flow support on the object is unreliable? Photometric residual persistence along a trajectory? Signed dark-blob appearance? Local contrast normalized by background texture? Something else?

4. How would you model static high-gradient background points that survive stabilization? Should we maintain a per-stabilized-location residual covariance over time, or is that likely to be too unstable under homography drift?

5. Is multi-frame robust background modeling in a stabilized frame worth pursuing without IMU/gyro, or will accumulated homography error and rolling shutter dominate?

6. If compute must stay lightweight for onboard use, what are the top three algorithmic primitives you would test next?

7. Offline, would you use a large grounding model only as a pseudo-labeling teacher, or would you also use it to train/design a small crop-level verifier?

## My current hypothesis

The next best direction is:

**high-recall cheap proposals + short-window track-before-detect + tiny crop/trajectory verifier**

rather than trying to perfect one-frame candidate scoring.

I would like your view on whether that is the right pivot, and if so, what exact statistical formulation would be hardest for terrain/tree/cloud clutter to fool.
