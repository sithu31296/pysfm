# Python Only Structure-from-Motion and 3D Computer Vision

## Progress

- [x] Focal Length Estimation (DepthPro, MoGe, GeoCalib)
- [x] Monocular Depth Estimation (DepthPro, MoGe)
- [x] Feature Detection and Matching (LoFTR, RoMa)
- [x] Relative Pose Estimation
- [x] Absolute Pose Estimation
- [ ] Triangulation
- [ ] Bundle Adjustment
- [ ] Solvers (Essential, Fundamental, PnP, RANSAC)
- [ ] Structure-from-Motion
- [ ] Visual Localization


## Tasks

Single-View
* Monocular Depth Estimation (From Models)
* Focal Length Estimation (From Metadata, Models)

Two-View (Stereo)
* Stereo Depth Estimation
* Local Feature Matching 
* Relative Pose Estimation
* Absolute Pose Estimation
* Triangulation
* Visual Localization

More Views 
* Structure-from-Motion
* Bundle Adjustment

Structure Post-Processing
* Real Metric-scale Injection

Pose Optimization
* Joint Optimization of Pose and NeRF/GS


## Installation

```bash
bash install.sh
```

## Single-View

### Monocular Depth Estimation

The goal is to get the pixel-wise depth values (and bonus: focal length).

Supported Metric-Depth Models:
* [DepthPro](https://github.com/apple/ml-depth-pro) (ArXiv 2024)

Supported Relative-Depth Models:
* [MoGe](https://github.com/microsoft/moge) (CVPR 2025)


Run this command to get the estimated depth and focal length (if not exist in metadata):

```bash
python scripts/estimate_depth.py
```
![mondepth_results](./assets/depth_with_cam.png)
> Notes: For outdoor images, mask the sky.

### Global Image Retrieval

The goal is to search for the most similar image from a database with respect to the given image.

Supported Models:


### Local Descriptor or Image Matching

Given two images, to find the matches (pixel locations) based on features correspondences.

Supported Feature Detectors/Matching Models:
* [LoFTR](https://github.com/zju3dv/LoFTR) (CVPR 2021)
* [RoMa](https://github.com/Parskatt/RoMa) (CVPR 2024)

Run this command to match the two images:

```bash
python scripts/match_views.py
```

Matched Keypoints             |  Matched 3D Effect
:-------------------------:|:-------------------------:
![matched_kpts](./assets/matched_kpts.png)  |  ![matched_3d](./assets/matched_3d.png)

> Notes: You can also specify the number of matches for dense methods.

### Pose Estimation (Relative/Absolute)

Given 2D correspondences, to find the relative or absolute pose between two images.

Relative Pose Estimation
* Up to an unknown scale
* 2D-2D Correspondences
* Pose relative to an another image

Absolute Pose Estimation or Camera Relocalization
* Up to a real scale
* 2D-3D Correspondences
* Pose relative to a world/map

> With relative pose from 2D-2D correspondences, you can only recover the direction of motion (translation vector is only correct up to scale), but not how far you moved. So the resulting pose is in an arbitary scale.

Pose Solvers:
* 2D-2D Correspondences
  * Essential Matrix (if you know the camera intrinsics)
  * Fundamental Matrix (if you don't know the camera intrinsics)
  * Homography
* 2D-3D Correspondences
  * PnP (if you know the camera intrinsics and depth of the first image)
* 3D-3D Corrspondences
  * Procrustes (if you know the camera intrinsics and depth of both images)

No matter which algorithm you choose, the initial correspondences will be pixel correspondences.

Run this command to match the two images:

```bash
python scripts/find_pose.py
```

Relative Pose Estimation   |  Absolute Pose Estimation
:-------------------------:|:-------------------------:
![relative_pose](./assets/relative_pose.png)  |  ![absolute_pose](./assets/absolute_pose.png)

> Here you can see that in relative pose estimation, the reconstructed point clouds are not aligned.

<!-- 
## Testing Datasets

Download the testing datasets from [here](https://colmap.github.io/datasets.html#datasets).

Name | #images | Intrinsics | Lens
--- | --- | --- | ---
Gerrard Hall | 100 | Same | Wide-angle
Graham Hall | 1273 | Same | Wide-angle
Person Hall | 330 | Same | Wide-angle
South Building | 128 | Same | -

## Two-View

### Global Matching

### Local Matching


### Relative Pose Estimation


### Absolute Pose Estimation

## More Views

### Structure-from-Motion

### Visual Localization


## Image Matching

* SIFT
* LoFTR

## Image Retrieval

* R2Former
* CosPlace
* NetVLAD
* SALAD


## Depth Estimation


## Structure-from-Motion

* COLMAP
* Detector-free SfM
* GLOMAP

### Structure

Usually, structure is a scene represention, that is implicit (defined with a neural network) or explicit (3D model that can be visualized directly, e.g. point cloud, meshes).
The structure can be recovered by
1. SfM approaches (unknown scale, correspondences between multiple images)
2. Scene Coordinate Regression (absolute or relative scale, directly estimate 3D points in scene space with a neural network)
3. LiDAR scanners (absolute scale, point clouds)

The goal is to get the 3D representation of the scene that may or may not have the absolute scale.

### Motion

Usually, motion is represented by:
1. Unstructured collection of images (no order, randomly captured with different cameras)
2. Sequences of images or Video (ordered images, sequentially captured with a single camera)

The goal is to estimate the pose (position + orientation) of the images relative to a scene's origin.

SfM is the task of recovering the scene structure from sufficient number of captured images.


## Absolute/Relative Pose Regression

Represent the scene with an implicit NN, which is trained end-to-end.
At test time, regress an absolute or relative pose from a query image.

Limitations of APR methods:
* w/o geometric constraints, they do not generalize well to novel viewpoints or appearances.
* They do not scale well when limiting network capacity. 

Limitations of RPR methods:
* These methods regress a camera pose relative to one or more database images. While being scene-agnostic, they are often limited in accuracy.


## Scene Coordinate Regression

Represents the scene within the weights of a NN.
Regresses corresponding 3D scene coordinates for all pixels in the query image.
First predict 2D-3D correspondences and then solve for the pose with PnP-RANSAC.
Usually, the network is supervised with ground truth 3D scene coordinates, (from a depth sensor or an SfM point cloud).
But recent works train w/o ground truth scene coordinates using a reprojection loss with ground truth poses and calibration parameters.

Limitations:
* Limited on small-scale scenes.


## Visual Localization or Re-localization


## Visual Odometry

## SLAM

## 3D Reconstruction -->


## References

* https://github.com/colmap/colmap
* https://github.com/naver/dust3r
* https://github.com/apple/ml-depth-pro
* https://github.com/microsoft/moge
