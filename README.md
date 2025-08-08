# Python Only 3D Computer Vision

![teaser](./assets/teaser.png)

## Supported Features

* Camera Model
  - [x] [Focal Length Estimation](#monocular-depth-estimation) (Metadata, MonoDepth)
* Camera Calibration
  - [ ] Single-Camera Calibration
  - [ ] Multi-Camera Calibration
  - [ ] PTZ-Camera Calibration
  - [ ] LiDAR-Camera Calibration
* Depth Estimation
  - [x] [Monocular Depth Estimation](#monocular-depth-estimation) ([DepthPro](https://github.com/apple/ml-depth-pro), [MoGe](https://github.com/microsoft/MoGe))
  - [x] [Stereo Depth Estimation](#stereo-depth-estimation) ([FoundationStereo](https://github.com/NVlabs/FoundationStereo))
* Camera Pose Estimation
  - [x] [Relative Pose Estimation](#pose-estimation-relativeabsolute) (Essential, Fundamental, Homography)
  - [x] [Absolute Pose Estimation](#pose-estimation-relativeabsolute) (PnP)
  - [x] [Visual Localization](#pose-estimation-relativeabsolute) (PnP)
  - [ ] Pose Optimization from Scene Representation (iNeRF, iCoMa)
* Point-set Registration
  - [x] [Rigid Registration](#rigid-registration) (Procrustes, ICP)
  - [x] [Non-Rigid Registration](#non-rigid-or-deformable-registration) (CPD)
* 3D Reconstruction
  - [ ] Triangulation
  - [x] [Global Image Retrieval](#global-image-retrieval) ([NetVLAD](https://github.com/Relja/netvlad), [SALAD](https://github.com/serizba/salad))
  - [x] [Local Feature Detection and Matching](#local-feature-detection-and-matching) ([RoMa](https://github.com/Parskatt/RoMa), [UFM](https://github.com/UniFlowMatch/UFM))
  - [x] [Point Tracking and Flow Matching]() (Cotracker)
  - [ ] Incremental Structure-from-Motion (COLMAP)
  - [ ] Global Structure-from-Motion (GLOMAP)
  - [ ] Feed-Forward Structure-from-Motion (VGGT)
* Global Alignment
  - [ ] Bundle Adjustment
  - [ ] Differentiable Bundle Adjustment
  - [ ] Video Depth Alignment
* Post Processing
  - [ ] Mesh Reconstruction (SDF, Marching Cubes, PSR)
  - [ ] Voxelization
  - [ ] Point Cloud Completion
* Libraries
  - [x] [OpenCV](https://github.com/opencv/opencv)
  - [x] [PoseLib](https://github.com/PoseLib/PoseLib)
  - [x] [PyCOLMAP](https://colmap.github.io/pycolmap/index.html)
  - [x] [SupeRANSAC](https://github.com/danini/superansac)
> WARNING: Some features or methods may be missing for now. They will be implemented slowly. The codebase may have significant changes.
<!-- 
A lot of 3D computer vision tasks will be supported with a simple inference script with SOTA models.
However, the following tasks will not be supported:
* Object-Level 3D Tasks (Classification, Detection, Segmentation)
* Scene Flow & Motion (Optical Flow) 
* Learning-based SfM (DUSt3R, MASt3R, VGGSfM, etc.) (may be later)
* Visual SLAM (may be later)
* Absolute Pose Regression
* Scene Coordinate Regression -->


## Installation

```bash
bash install.sh
```

## Monocular Depth Estimation

The goal is to get the pixel-wise depth values (and bonus: focal length).

Supported Metric-Depth Models:
* [DepthPro](https://github.com/apple/ml-depth-pro) (ArXiv 2024)

Supported Relative-Depth Models:
* [MoGe](https://github.com/microsoft/moge) (CVPR 2025)


Run this command to get the estimated depth and focal length (if not exist in metadata):

```bash
python scripts/estimate_depth.py
```
![monodepth_results](./assets/depth_with_cam.png)
> Notes: For outdoor images, mask the sky.

## Stereo Depth Estimation

The goal is to get the pixel-wise depth values given stereo images.
Usually the depth values are in metric-scale and accurate compared to monocular depth methods.

Supported Models:
* [FoundationStereo](https://github.com/NVlabs/FoundationStereo) (CVPR 2025 Oral)

> Notes: For FoundationStereo, download the pre-trained model from official repo and put them in checkpoints/ folder.

Run this command to get the estimated depth for the left image:

```bash
python scripts/estimate_stereo_depth.py
```
Ground-truth                |  Estimated
:-------------------------:|:-------------------------:
![gt_stereo](./assets/gt_stereo_depth.png)  |  ![est_stereo](./assets/est_stereo_depth.png)

Image from [MiddleBury 2005 Dataset](https://vision.middlebury.edu/stereo/data/scenes2005/).

## Global Image Retrieval

The goal is to search for the most similar image from a database with respect to the given image.

Supported Models:
* [SALAD](https://github.com/serizba/salad) (CVPR 2024)

```bash
python scripts/match_global.py
```

> Output: A JSON file with queries as keys and Top-3 matched candidates.


## Local Feature Detection and Matching

Given two images, to find the matches (pixel locations) based on features correspondences.

Supported Feature Detectors/Matching Models:
<!-- * [LoFTR](https://github.com/zju3dv/LoFTR) (CVPR 2021) -->
* [RoMa](https://github.com/Parskatt/RoMa) (CVPR 2024)
* [UFM](https://github.com/UniFlowMatch/UFM) (ArXiv 2025)

Run this command to match the two images:

```bash
python scripts/match_views.py
```

Matched Keypoints             |  Matched 3D Effect
:-------------------------:|:-------------------------:
![matched_kpts](./assets/matched_kpts.png)  |  ![matched_3d](./assets/matched_3d.png)

> Notes: You can also specify the number of matches for dense methods.

## Pose Estimation (Relative/Absolute)

Given 2D correspondences, to find the relative or absolute pose between two images.

Relative Pose Estimation
* Up to an unknown scale
* 2D-2D Correspondences
* Pose relative to an another image
* Algorithms
  * 5 points
  * 8 points

Absolute Pose Estimation or Camera Relocalization
* Up to a real scale
* 2D-3D Correspondences
* Pose relative to a world/map
* Algorithms
  * P3P (3 points)
  * P4Pf (4 points)

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
python scripts/estimate_pose.py
```

Relative Pose Estimation   |  Absolute Pose Estimation
:-------------------------:|:-------------------------:
![relative_pose](./assets/relative_pose.png)  |  ![absolute_pose](./assets/absolute_pose.png)

<!-- > Here you can see that in relative pose estimation, the reconstructed point clouds are not aligned. -->


## Point Cloud Registration

### Rigid Registration

Given two point clouds, find the global transformation and align them.

Global Registration:
* Procrustes or Kabsch

Local Registration:
* ICP

Feature-based Registration:
* [Deep Gloabl Registration](https://github.com/chrischoy/DeepGlobalRegistration) (CVPR 2020 Oral)

Run this command to align the point clouds:

```bash
python scripts/align_pcd.py
```

Gloabl Registration   |  Local Registration
:-------------------------:|:-------------------------:
![before_regist](./assets/global_regist.png)  |  ![after_regist](./assets/global_regist.png)

> Before registration/alignment, the two point clouds are in their own coordinate system.

### Non-Rigid or Deformable Registration

Given two point clouds, find the transformation of each point and align them.

Supported Methods:
* [CPD](https://github.com/siavashk/pycpd/tree/master) (TPAMI 2010)


Run this command to align the point clouds:

```bash
python scripts/align_pcd_non_rigid.py
```

Gloabl Registration   |  Non-Rigid Registration
:-------------------------:|:-------------------------:
![global_regist](./assets/global_regist.png)  |  ![deform_regist](./assets/deform_regist.png)

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
