from pysfm import *

def backproject_3d(uv, depth, K):
    '''
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [N,2]
    :param depth: array [N]
    :param K: array [3,3]
    :return: xyz: array [N,3]
    '''

    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    xyz = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
    return xyz


class EssentialSolver:
    """Estimate relative pose (up to scale) given a set of 2D-2D correspondences
    """
    def __init__(self, reproj_threshold=0.5, ransac_iters=10000):
        self.reproj_threshold = reproj_threshold
        self.ransac_iters = ransac_iters
        self.mask = None

    def __call__(self, kpts1, kpts2, K1, K2):
        R = np.full((3, 3), np.nan)
        t = np.full((3, 1), np.nan)
        inliers = 0
        if len(kpts1) < 5:
            return R, t, inliers
        
        # normalize keypoints
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        kpts2 = (kpts2 - K2[[0, 1], [2, 2]][None]) / K2[[0, 1], [0, 1]][None]

        # normalize ransac threshold
        ransac_thr = self.reproj_threshold / np.mean([K1[0, 0], K2[1, 1], K1[1, 1], K2[0, 0]])

        E, mask = cv2.findEssentialMat(kpts1, kpts2, 
                                       np.eye(3), 
                                       cv2.USAC_MAGSAC, 
                                       0.9999, 
                                       ransac_thr, 
                                       self.ransac_iters)
        self.mask = mask
        if E is None:
            return R, t, inliers

        best_n_inliers = 0
        ret = R, t, 0
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts1, kpts2, np.eye(3), 1e9, mask=mask)
            if n > best_n_inliers:
                best_n_inliers = n
                ret = (R, t[:, 0], n)
        return ret
    

class EssentialMetricSolver(EssentialSolver):
    """Estimate relative pose (with scale) given a set of 2D-2D correspondences

    The scale of the translation vector is obtained using RANSAC over the possible scales recovered from 3D-3D correspondences.
    """
    def __init__(self, ransac_scale_threshold=0.1, reproj_threshold=0.5, ransac_iters=10000):
        super().__init__(reproj_threshold, ransac_iters)
        self.ransac_scale_threshold = ransac_scale_threshold

    def __call__(self, kpts1, kpts2, K1, K2, depth1, depth2):
        R, t, inliers = super().__call__(kpts1, kpts2, K1, K2)
        if inliers == 0:
            return R, t, inliers
        
        mask = self.mask.ravel() == 1

        inlier_kpts1 = np.int32(kpts1[mask])
        inlier_kpts2 = np.int32(kpts2[mask])

        inlier_depth1 = depth1[inlier_kpts1[:, 1], inlier_kpts1[:, 0]]
        inlier_depth2 = depth2[inlier_kpts2[:, 1], inlier_kpts2[:, 0]]

        # check for valid depth
        valid = (inlier_depth1 > 0) * (inlier_depth2 > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers
        
        points1 = backproject_3d(inlier_kpts1[valid], inlier_depth1[valid], K1)
        points2 = backproject_3d(inlier_kpts2[valid], inlier_depth2[valid], K2)

        # rotate points1 to points2 coordinate system (so that axes are parallel)
        # points1 = (R @ points1.T).T
        points2 = (R.T @ points2.T).T

        # get individual scales (for each 3D-3D correspondence)
        scales = np.dot(points2 - points1, t.reshape(3, 1))

        # RANSAC loop
        best_inliers = 0
        best_scale = None
        for scale_hyp in scales:
            inliers_hyp = (np.abs(scales - scale_hyp) < self.ransac_scale_threshold).sum().item()
            if inliers_hyp > best_inliers:
                best_inliers = inliers_hyp
                best_scale = scale_hyp

        t_metric = best_scale * t
        return R, t_metric, best_inliers