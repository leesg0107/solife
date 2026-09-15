---
layout: post
title: "Semantic VSLAM"
subtitle: "Stereo visual SLAM with YOLO object detection for a drone, tested in UE5"
tags: [slam, computer-vision, drone, ros2]
category: project
thumbnail-img: "/assets/img/semantic_vslam_pipeline.png"
---
Visual SLAM estimates a robot's pose and builds a map from cameras alone. LiDAR is the standard sensor for robotic mapping, but it is expensive and heavy, while cameras are cheap and already carried by most platforms. The cost is that depth must be inferred from images rather than measured directly, so camera-only maps are less precise than LiDAR maps. This project adds an object detector (YOLO) on top of a stereo VSLAM pipeline so that the map contains labeled objects with 3D positions, not only geometry. The target platform is a drone with three stereo cameras (front-left, front-right, and rear); the pipeline was developed and tested in Unreal Engine 5 with ROS 2.

![Semantic VSLAM Pipeline](/assets/img/semantic_vslam_pipeline.png)
*Semantic VSLAM pipeline overview*

## Tech Stack

- **RTAB-Map** for RGB-D SLAM backend
- **OpenCV StereoSGBM** for calculating Disparity
- **WLS Filter** for refining Disparity
- **robot_localization** for EKF sensor fusion
- **YOLOv8n** for object detection
- **TF2** for transforming coordinates

---

## 1. Stereo Depth Estimation

Depth is estimated from stereo image pairs using the standard relation:

```
depth = (fx × baseline) / disparity
```

with

- **fx** = 554.25 (focal length in pixels)
- **baseline** = 0.12m (distance between the two cameras)
- **disparity** = how much a pixel shifts between left and right images

For this camera configuration the depth range is:

- **Minimum depth** (disparity=128): about 0.52m
- **Maximum depth** (disparity=1): about 66m
- **Practical range**: 0.3m ~ 15m

---

## 2. StereoSGBM Algorithm

Semi-global block matching has three components:

### 2.1 Block Matching

The front-left and front-right cameras provide an image pair. The algorithm compares patches between the two images using a `blockSize × blockSize` window and searches for the best-matching disparity.

### 2.2 Cost Function

Each disparity assignment is scored by an energy function:

```
E(D) = Σ(C(p, D_p) + Σ P1·T[|D_p - D_q| = 1] + P2·T[|D_p - D_q| > 1])
```

P1 penalizes small disparity changes between neighboring pixels and P2 penalizes large ones. The values used:

- **P1** = 8 × 3 × blockSize² (small change penalty)
- **P2** = 32 × 3 × blockSize² (big jump penalty)

### 2.3 Semi-Global Optimization

Instead of just looking at one direction, the algorithm accumulates costs along 8 different paths (up, down, left, right, and diagonals). This makes the depth map much more consistent than simple block matching.

---

## 3. WLS Filtering

Raw disparity maps are noisy. The weighted least squares (WLS) filter solves:

```
minimize: Σ(u_i - d_i)² + λ Σ w_ij(u_i - u_j)²
```

The first term keeps the output close to the input and the second smooths it, with weights that suppress smoothing across image edges. **λ = 8000** sets the smoothing strength and **σ_color = 1.5** the edge sensitivity.

The result is a depth map that is smooth in uniform regions and sharp at object boundaries.

---

## 4. 3D Back-Projection

Pixel coordinates and depth are converted to 3D camera coordinates by inverting the pinhole model:

```
X = (px - cx) × depth / fx
Y = (py - cy) × depth / fy
Z = depth
```

where (cx, cy) is the principal point and (fx, fy) are the focal lengths.

The physical size of a detected object follows from its bounding box:

```
object_width = bbox_width × depth / fx
object_height = bbox_height × depth / fy
```

A detection is therefore reported with a distance and an approximate physical size, not only a label.

---

## 5. Extended Kalman Filter (EKF) Fusion

Visual odometry alone is insufficient for flight control: it runs at only 10 to 15 Hz, too slow for the control loop, and it accumulates drift over a trajectory. The IMU runs at 250 Hz, but it is noisy and dead-reckoning from it drifts within seconds. The two error profiles are complementary, so an extended Kalman filter (EKF) fuses them.

### State Vector (15 dimensions)

```
x = [x, y, z, roll, pitch, yaw, vx, vy, vz, ωx, ωy, ωz, ax, ay, az]ᵀ
```

Position, orientation, linear velocity, angular velocity, and linear acceleration.

### Prediction and Update

**Prediction (when IMU arrives at 250Hz):**

```
x̂_predicted = f(x̂_previous, IMU_data)
P_predicted = F × P_previous × Fᵀ + Q
```

**Update (when VSLAM arrives at 10-15Hz):**

```
K = P_predicted × Hᵀ × (H × P_predicted × Hᵀ + R)⁻¹
x̂_updated = x̂_predicted + K × (measurement - expected)
P_updated = (I - K × H) × P_predicted
```

The Kalman gain K weights the VSLAM measurement against the IMU prediction according to their covariances: a confident VSLAM measurement gives a large gain, a noisy one a small gain. The filter outputs a 50 Hz pose estimate.

---

## 6. Coordinate Frame Transformation

ROS and PX4 use different body-frame conventions.

- **ROS (FLU)**: x-forward, y-left, z-up
- **PX4 (FRD)**: x-forward, y-right, z-down

To convert, we rotate 180° around the x-axis:

```
# Position transformation
x_px4 = x_ros
y_px4 = -y_ros  (flip sign)
z_px4 = -z_ros  (flip sign)

# Quaternion transformation
R_convert = Rotation.from_euler('x', 180, degrees=True)
q_px4 = R_convert × q_ros × R_convert.inverse()
```

Without this transformation the z axis is inverted and the controller receives an upside-down pose.

---

## 7. RTAB-Map Visual Odometry

RTAB-Map provides the SLAM backend.

### Feature Extraction

GFTT (good features to track) keypoints with BRIEF descriptors, up to 5000 features per frame, with a minimum of 5 inliers for a valid pose estimate.

### Motion Estimation

Frame-to-Map matching (Odom/Strategy: 0):

1. Extract features from current frame
2. Match against accumulated map features
3. Use PnP algorithm to estimate camera pose
4. RANSAC removes outliers

### Loop Closure Detection

When the drone returns to a previously visited place, RTAB-Map detects it using Bag-of-Words visual similarity. This corrects accumulated drift and keeps the map consistent.

---

## 8. Semantic Mapping Pipeline

Detections are combined with depth as follows:

```
RGB Image → YOLO → 2D Bounding Boxes
                      ↓
              + Depth Image
                      ↓
              3D Position (Back-projection)
                      ↓
              SemanticObject
                      ↓
         ┌───────────┴───────────┐
         ↓                       ↓
   RViz MarkerArray      Central LLM (JSON)
```

### Depth Sampling Strategy

Depth at the exact center of a bounding box is noisy, so the depth is taken as the median of the valid depths (0.3–15 m) inside the bounding box.

```python
depth_region = depth_image[y1:y2, x1:x2]
valid_depths = depth_region[(depth_region > 0.3) & (depth_region < 15.0)]
depth = np.median(valid_depths)
```

---

## 9. Testing in UE5

The pipeline was tested in Unreal Engine 5 using a car dealership scene. The scene was chosen because it is rich in visual features (cars, signs, windows, reflections), which feature-based SLAM depends on; blank walls and open fields are the failure cases.

<video width="100%" controls>
  <source src="/assets/img/semantic_vslam_demo.webm" type="video/webm">
  Your browser does not support the video tag.
</video>
*Semantic VSLAM demo: drone navigation with real-time 3D mapping and object detection*

In the demo the drone builds a 3D map while flying, and YOLO detections (cars, people) are projected into it. The result is a semantic map: each detected object carries a class label as well as a 3D position.

---

## Summary

The pipeline consists of:

1. **Stereo → Depth**: StereoSGBM + WLS filter generates high-quality depth maps
2. **Visual Odometry**: RTAB-Map RGB-D mode provides stable pose estimation
3. **Sensor Fusion**: EKF combines VSLAM (10Hz) + IMU (250Hz) → smooth 50Hz output
4. **Semantic Mapping**: YOLO detection + depth projection → 3D semantic objects
5. **PX4 Integration**: FLU→FRD coordinate transformation for drone control

The next step is fusing GPS into the same filter: VSLAM provides local accuracy and GPS a global reference, which should bound long-term drift and allow mapping over larger areas.
