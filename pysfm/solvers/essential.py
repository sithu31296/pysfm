from pysfm import *
import poselib



class EssentialSolver:
    """Estimate relative pose (up to scale) given a set of 2D-2D correspondences
    """
    def __init__(self, ransac_pix_threshold=0.5, ransac_iters=10000, method='opencv'):
        self.ransac_pix_threshold = ransac_pix_threshold
        self.ransac_confidence = 0.999
        self.ransac_iters = ransac_iters
        self.method = method
        self.rotation = np.full((3, 3), np.nan)
        self.translation = np.full((3, 1), np.nan)
        self.num_inliers = 0
        self.inliers_mask = None

    
    def __call__(self, kpts1, kpts2, K1, K2):
        self.inliers_mask = np.zeros(kpts1.shape[0]).astype(bool)
        if len(kpts1) < 5:
            return self.rotation, self.translation, self.num_inliers, self.inliers_mask
        
        if self.method == 'opencv':
            R, t, inliers, mask = self.solve_E_opencv(kpts1, kpts2, K1, K2)
        elif self.method == 'poselib':
            R, t, inliers, mask = self.solve_E_poselib(kpts1, kpts2, K1, K2)
    
        return R, t, inliers, mask


    def solve_E_opencv(self, kpts1, kpts2, K1, K2):
        # normalize keypoints
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        kpts2 = (kpts2 - K2[[0, 1], [2, 2]][None]) / K2[[0, 1], [0, 1]][None]

        # normalize ransac threshold
        ransac_thr = self.ransac_pix_threshold / np.mean([K1[0, 0], K2[1, 1], K1[1, 1], K2[0, 0]])

        E, mask = cv2.findEssentialMat(kpts1, kpts2, np.eye(3),
                                       method=cv2.USAC_MAGSAC, prob=self.ransac_confidence,
                                       threshold=ransac_thr, maxIters=self.ransac_iters)
        mask = mask.ravel() == 1
        if E is None:
            return self.rotation, self.translation, self.num_inliers, mask
        
        kpts1_f = kpts1[mask]
        kpts2_f = kpts2[mask]

        _, R, t, _ = cv2.recoverPose(E, kpts1_f, kpts2_f, np.eye(3), 1e9)
        return R, t, np.sum(mask), mask
    

    def solve_E_poselib(self, kpts1, kpts2, K1, K2, shared_focal=False):
        """Robust estimators from Poselib

        Poselib estimators first normalize the data, calls the RANSAC and runs a post-RANSAC non-linear refinement.
        The RANSAC implementation is LO-RANSAC
        """
        kpts1, kpts2 = kpts1.astype(np.float32), kpts2.astype(np.float32)
        K1, K2 = K1.astype(int), K2.astype(int)
        # robust estimator options
        ransac_options = {
            "max_iterations": 100000,
            "min_iterations": 1000,
            "dyn_num_trials_mult": 3.0,
            "success_prob": 0.9999,                     
            # "max_reproj_error": 12.0,                   # used for 2D-3D matches (default=12.0)
            "max_epipolar_error": 0.75,     # used for 2D-2D matches (default=1.0)
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
        # camera model (COLMAP style)
        cam1 = {
            "model": "SIMPLE_PINHOLE",  # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE
            "width": K1[0, 2]*2,                 # defined but not used anywhere in poselib
            "height": K1[1, 2]*2,                # defined but not used anywhere in poselib
            "params": [K1[0, 0], K1[0, 2], K1[1, 2]]
        }
        cam2 = {
            "model": "SIMPLE_PINHOLE",  # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE
            "width": K2[0, 2]*2,                 # defined but not used anywhere in poselib
            "height": K2[1, 2]*2,                # defined but not used anywhere in poselib
            "params": [K2[0, 0], K2[0, 2], K2[1, 2]]
        }
        E, info = poselib.estimate_relative_pose(kpts1, kpts2, cam1, cam2, ransac_options, bundle_options, initial_pose=None)

        if shared_focal:
            E, info = poselib.estimate_shared_focal_relative_pose(kpts1, kpts2, (0, 0), ransac_options, bundle_options, initial_image_pair=None)
        
        # refinements = info['refinements']   # refinement iterations
        # iterations = info['iterations']
        # inlier_ratio = info['inlier_ratio'] # in decimals
        # model_score = info['model_score']       # cost
        return np.array(E.R), np.array(E.t)[:, None], info['num_inliers'], np.array(info['inliers']).astype(bool)
        
