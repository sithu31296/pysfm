from pysfm import *
import poselib



class PnPSolver:
    def __init__(self, reproj_threshold=0.5, ransac_iters=10000, method='opencv'):
        self.reproj_threshold = reproj_threshold
        self.ransac_iters = ransac_iters
        self.method = method
        self.rotation = np.full((3, 3), np.nan)
        self.translation = np.full((3, 1), np.nan)
        self.num_inliers = 0
        self.inliers_mask = None

    def __call__(self, pts2d, pts3d, K, image_size=None):
        self.inliers_mask = np.zeros(pts2d.shape[0]).astype(bool)
        if len(pts2d) < 5:
            return self.rotation, self.translation, self.num_inliers, self.inliers_mask
        
        if self.method == 'opencv':
            R, t, inliers, mask = self.solve_pnp_opencv(pts2d, pts3d, K)
        elif self.method == 'poselib':
            R, t, inliers, mask = self.solve_pnp_poselib(pts2d, pts3d, K, image_size)
        return R, t, inliers, mask

    def solve_pnp_opencv(self, pts2d, pts3d, K):
        pts2d = pts2d.astype(np.float32)
        success, r_vec, t_vec, inliers = cv2.solvePnPRansac(pts3d, pts2d, K, None, 
                                                      flags=cv2.SOLVEPNP_SQPNP,
                                                      iterationsCount=self.ransac_iters,
                                                      reprojectionError=self.reproj_threshold,
                                                      confidence=0.999)
        if not success:
            return self.rotation, self.translation, self.num_inliers, self.inliers_mask
        
        R = cv2.Rodrigues(r_vec)[0]
        inlier_mask = self.inliers_mask
        inlier_mask[inliers.squeeze()] = 1
        return R, t_vec, np.sum(inlier_mask), inlier_mask
    
    def solve_pnp_poselib(self, pts2d, pts3d, K, image_size=None):
        """Robust estimators from Poselib

        Poselib estimators first normalize the data, calls the RANSAC and runs a post-RANSAC non-linear refinement.
        The RANSAC implementation is LO-RANSAC
        """
        pts2d, pts3d = pts2d.astype(int), pts3d.astype(np.float32)
        K = K.astype(int)
        # robust estimator options
        ransac_options = {
            "max_iterations": 100000,
            "min_iterations": 1000,
            "dyn_num_trials_mult": 3.0,
            "success_prob": 0.9999,                     
            "max_reproj_error": self.reproj_threshold,                   # used for 2D-3D matches (default=12.0)
            # "max_epipolar_error": 1.0,                  # used for 2D-2D matches (default=1.0)
            "seed": 0,

            # PROSAC parameters
            "progressive_sampling": False,              # data sorting for PROSAC sampling
            "max_prosac_iterations": 100000,

            # whether to use real focal length checking for F estimation
            # assumes that principal points of both cameras are at origin
            "real_focal_check": False,

            # whether to treat the input "best_model" as an initial model
            # and score it before running the main RANSAC loop
            "score_initial_model": False
        }
        # non-linear refinement options
        bundle_options = {
            "max_iterations": 100,
            "loss_type": "CAUCHY",      # TRIVIAL, TRUNCATED, HUBER, CAUCHY, TRUNCATED_LE_ZACH
            "loss_scale": 1.0,
            "gradient_tol": 1e-8,
            "step_tol": 1e-08,
            "initial_lambda": 1e-3,
            "min_lambda": 1e-10,
            "max_lambda": 1e10,
            "verbose": False
        }
        if image_size is None:
            image_size = (int(K[1, 2]*2), int(K[0, 2]*2))
        # camera model (COLMAP style)
        cam = {
            "model": "SIMPLE_PINHOLE",              # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE
            "width": image_size[1],                 # defined but not used anywhere in poselib
            "height": image_size[0],                # defined but not used anywhere in poselib
            "params": [K[0, 0], K[0, 2], K[1, 2]]   # [f, cx, cy]
        }
        pose, info = poselib.estimate_absolute_pose(pts2d, pts3d, cam,
                                                 ransac_options, bundle_options,
                                                 initial_pose=None)

        return np.array(pose.R), np.array(pose.t)[:, None], info['num_inliers'], np.array(info['inliers']).astype(bool)
        

    
    
    