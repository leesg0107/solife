---
layout: post
title: "Visual SLAM vs Occupancy Networks"
subtitle: "Vision-based mapping without LiDAR: trade-offs and when to use each"
tags: [slam, computer-vision, mapping]
category: study
mathjax: true
---

Both approaches map an environment from cameras alone: visual SLAM estimates camera pose and map jointly, while a learned occupancy network predicts a dense free-space grid in the ego frame and takes its pose from elsewhere (odometry, SLAM, or GNSS/INS). LiDAR remains the standard sensor for robotic mapping, but it is heavy, power-hungry, and expensive, while cameras are cheap and already present on most platforms. This note compares visual SLAM and learned occupancy prediction along six criteria. There is no universal winner; the trade-offs decide which fits a given system.

## Definitions

- Visual SLAM
  - Tracks image features (e.g., ORB‑SLAM family) or uses direct photometric alignment to jointly estimate camera pose and a map (points/keyframes).
  - Inputs: Monocular / Stereo / RGB‑D / VIO (with IMU). Pure monocular has scale ambiguity; stereo/depth/IMU resolves it.
  - Outputs: Accurate trajectory, sparse/semi‑dense map, loop‑closure to rein in drift.
  - Canon: ORB‑SLAM2/3, DSO, VINS‑Fusion, OKVIS.

- Occupancy Networks (learned occupancy prediction: BEV / 3D-voxel free-space)
  - Name clash worth flagging: the original *Occupancy Networks* (Mescheder et al., 2019) is an implicit-field method for reconstructing watertight 3D **surfaces**. In robotics and driving the same term gets reused loosely for learned **occupancy-grid prediction** — free-space vs. obstacles, e.g. Tesla's camera-only occupancy net — and that's the sense this comparison uses.
  - A network predicts occupancy probabilities over 3D coordinates or a BEV grid, yielding free‑space vs obstacles. Needs training data; infers 3D from multi‑view or sequential camera frames.
  - Pros: Data‑driven robustness to poor texture/lighting, easy to fuse semantics, dense map outputs.
  - Cons: Training cost and inference compute, harder real‑time, generalization concerns.

## 1. Setup and Cost
**Advantage: visual SLAM.**  
No training data or large‑scale modeling required. Calibrate and go. Occupancy adds training and serving infra, increasing upfront cost.

## 2. Compute and Real-Time Operation
**Advantage: visual SLAM, usually.**  
SLAM often runs light with low latency. Occupancy can be real‑time with quantization/pruning and smaller BEV, but embedded budgets make it harder.

## 3. Map Density and Semantics
**Advantage: occupancy.**  
SLAM excels at pose and sparse maps; occupancy excels at dense free‑space/obstacle representation and semantic fusion.

- Visual SLAM
  - Pros: Low latency, consistent trajectory, relatively light compute. Very accurate with good features/texture.
  - Limits: Weak texture, repetitive patterns, harsh lighting, and dynamic objects can hurt. Monocular has scale ambiguity; long runs accumulate drift.

- Occupancy Networks
  - Pros: Dense 3D occupancy/free‑space, easy semantic fusion, wide FoV via multi‑camera.
  - Limits: Inference latency/compute, dataset bias and OOD generalization, stability for online updates.

## 4. Robustness
**No clear advantage.**  
Feature‑poor or lighting‑harsh scenes break classic SLAM (stereo/IMU helps). Occupancy can learn around it, but out-of-distribution inputs remain a failure mode.

## 5. Developer Experience
**Advantage: visual SLAM.**  
Start without labels or a training pipeline; Occupancy requires a data and training pipeline.

## 6. Scaling and Planning Fit
**Advantage: occupancy.**  
BEV occupancy makes free‑space explicit, which planners consume directly.

---

## Scorecard

| Criterion | Advantage |
|---|---|
| Setup and cost | Visual SLAM |
| Compute and real-time | Visual SLAM (usually) |
| Map density and semantics | Occupancy |
| Robustness | Depends on the environment |
| Developer experience | Visual SLAM |
| Planning fit | Occupancy |

## When to Use Which

- Tiny budget/embedded; stability now → Visual SLAM (+IMU).
- Wide‑FoV free‑space/obstacle/semantic richness → Occupancy/BEV (with quantization or offloading).
- Most practical: hybrid—SLAM for pose/loop‑closure; occupancy for dense space/semantics.

## Pipeline sketches

- Visual SLAM
  1) Camera/IMU calibration → 2) Feature tracking / pose estimation → 3) Loop closure / BA → 4) Local map → 5) Planning

- Occupancy Network (BEV example)
  1) Multi‑view/depth alignment → 2) Spatiotemporal encoding (backbone) → 3) 3D/BEV transform → 4) Occupancy field prediction → 5) Post‑processing/planning

## Minimal setups

- RPi 5 + monocular: ORB‑SLAM3 (mono+IMU) or VINS‑Fusion for VIO; start slow.
- Multi‑camera (or RGB‑D) + GPU: Light BEV occupancy model (e.g., Fast‑BEV/BEVFormer‑style), stabilized with SLAM poses.

## Summary

- The question is not which is universally better, but which fits the current requirements.
- Need lightweight, stable navigation now → Visual SLAM.
- Want richer scene understanding (free‑space/semantics/dense map) → Occupancy.
- In most practical systems the two are combined: SLAM for pose, occupancy for space.

## References

- ORB‑SLAM3: [https://arxiv.org/abs/2007.11898](https://arxiv.org/abs/2007.11898)
- VINS‑Fusion: [https://arxiv.org/abs/1712.00036](https://arxiv.org/abs/1712.00036)
- Occupancy Networks — Mescheder et al., 2019 (implicit 3D surface reconstruction; origin of the name): [https://arxiv.org/abs/1812.03828](https://arxiv.org/abs/1812.03828)
- Camera-based occupancy for driving / BEV — Tesla AI (overview): [https://www.tesla.com/AI](https://www.tesla.com/AI)
