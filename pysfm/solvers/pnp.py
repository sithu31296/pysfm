from pysfm import *
from pysfm.utils.geometry import *


def compute_P(p2d, p3d):
    """Compute camera matrix from pairs of 2D-3D correspondences in homo coordinates.
    """
    n = p2d.shape[1]

    # create matrix for DLT solution
    M = np.zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = p3d[:, i]
        M[3 * i + 1, 4:8] = p3d[:, i]
        M[3 * i + 2, 8:12] = p3d[:, i]
        M[3 * i:3 * i + 3, i + 12] = -p2d[:, i]

    U, S, V = np.linalg.svd(M)
    return V[-1, :12].reshape((3, 4))


class PnPSolver:
    def __init__(self, method='opencv', reproj_threshold=0.5, ransac_iters=10000):
        self.reproj_threshold = reproj_threshold
        self.ransac_iters = ransac_iters
        self.method = method

    def __call__(self, kpts1, kpts2, K):
        if self.method == 'opencv':
            E, mask = cv2.findEssentialMat(kpts1, kpts2, K, cv2.RANSAC, 0.9999, self.reproj_threshold, self.ransac_iters)
        return E, mask.ravel()
    
    def recover_pose(self, kpts1, kpts2, K):
        E, mask = self(kpts1, kpts2, K)
        kpts1, kpts2 = kpts1[mask == 1], kpts2[mask == 1]
        if self.method == 'opencv':
            _, R, t, _ = cv2.recoverPose(E, kpts1, kpts2, K)
        return R, t