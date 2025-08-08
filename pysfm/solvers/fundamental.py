from pysfm import *
import poselib



class FundamentalSolver:
    """Estimate relative pose (up to scale) given a set of 2D-2D correspondences with unknown intrinsics
    """
    def __init__(self, ransac_pix_threshold=0.1, ransac_iters=1000, method='opencv'):
        self.ransac_pix_threshold = ransac_pix_threshold
        self.ransac_iters = ransac_iters
        self.method = method
        self.rotation = np.full((3, 3), np.nan)
        self.translation = np.full((3, 1), np.nan)
        self.num_inliers = 0
        self.inliers_mask = None


    def __call__(self, kpts1, kpts2, K, K2=None):
        self.inliers_mask = np.zeros(kpts1.shape[0]).astype(bool)
        if len(kpts1) < 5:
            return self.rotation, self.translation, self.num_inliers, self.inliers_mask
        
        if self.method == 'opencv':
            F, inliers, mask = self.solve_F_opencv(kpts1, kpts2)
        else:
            F, inliers, mask = self.solve_F_poselib(kpts1, kpts2)
        
        R, t = self.recover_pose(F, kpts1, kpts2, mask, K)
        return R, t, inliers, mask

    def recover_pose(self, F, kpts1, kpts2, mask, K):
        kpts1, kpts2 = kpts1[mask], kpts2[mask]
        E = K.T @ F @ K
        _, R, t, _ = cv2.recoverPose(E, kpts1, kpts2, K)
        return R, t

    def solve_F_opencv(self, kpts1, kpts2):
        F, mask = cv2.findFundamentalMat(kpts1, kpts2,
                                            method=cv2.USAC_MAGSAC,
                                            ransacReprojThreshold=self.ransac_pix_threshold,
                                            confidence=0.999999,
                                            maxIters=self.ransac_iters)
        mask = mask.ravel() == 1
        return F, np.sum(mask), mask
    

    def solve_F_poselib(self, kpts1, kpts2):
        """Robust estimators from Poselib

        Poselib estimators first normalize the data, calls the RANSAC and runs a post-RANSAC non-linear refinement.
        The RANSAC implementation is LO-RANSAC
        """
        kpts1, kpts2 = kpts1.astype(np.float32), kpts2.astype(np.float32)
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
        F, info = poselib.estimate_fundamental(kpts1, kpts2, ransac_options, bundle_options, initial_F=None)
        return F, info['num_inliers'], np.array(info['inliers']).astype(bool)