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


class RelativePoseScaleSolver:
    """Estimate metric scale of a relative pose given known depths
    """
    def __init__(self, ransac_scale_threshold=0.1, use_ransac=True) -> None:
        self.ransac_scale_threshold = ransac_scale_threshold
        self.use_ransac = use_ransac

    def __call__(self, 
                 kpts1: np.ndarray, kpts2: np.ndarray, 
                 depth1: np.ndarray, depth2: np.ndarray,
                 K1: np.ndarray, K2: np.ndarray,
                 R: np.ndarray, t: np.ndarray, inliers_mask: np.ndarray) -> float:
        if np.sum(inliers_mask) < 2:
            return 1
        
        inlier_kpts1, inlier_kpts2 = kpts1[inliers_mask], kpts2[inliers_mask]
        inlier_depth1 = depth1[inlier_kpts1[:, 1], inlier_kpts1[:, 0]]
        inlier_depth2 = depth2[inlier_kpts2[:, 1], inlier_kpts2[:, 0]]

        # check for valid depth
        valid = (inlier_depth1 > 0) & (inlier_depth2 > 0)
        if valid.sum() < 2:
            return 1
        
        points1 = backproject_3d(inlier_kpts1[valid], inlier_depth1[valid], K1)
        points2 = backproject_3d(inlier_kpts2[valid], inlier_depth2[valid], K2)

        # rotate points1 to points2 coordinate system
        points1 = (R @ points1.T).T

        if self.use_ransac:
            return self.ransac_scale(points1, points2, t)
        else:
            return self.mean_scale(points1, points2, t)
        
    def mean_scale(self, points1, points2, t):
        # get average point for each camera
        points1_mean = np.mean(points1, axis=0)
        points2_mean = np.mean(points2, axis=0)

        # find scale as the length of the translation vector that minimizes the 3D distance
        # between projected points from 1 and the corresponding points in 2
        scale = (points2_mean - points1_mean) @ t
        return scale

    
    def ransac_scale(self, points1, points2, t):
        # get individual scales (for each 3D-3D correspondence)
        scales = (points2 - points1) @ t

        # RANSAC loop
        best_inliers = 0
        best_scale = 1
        for scale in scales:
            inliers = (np.abs(scales - scale) < self.ransac_scale_threshold).sum().item()
            if inliers > best_inliers:
                best_inliers = inliers
                best_scale = scale
        return best_scale