from pysfm import *
from pysfm.utils.geometry import inv, to_homo, compute_epipole, skew



def scale_and_translate_points(points):
    """Scale and translate image points so that the centroid of points are at the origin
        and average distance to the origin is equal to sqrt(2)
    Args:
        points: homogeneous matrix with shape (3, #points)
    Returns:
        matrix of same shape and its normalization matrix
    """
    x, y = points[0], points[1]
    center = points.mean(axis=1)    # mean of each row
    cx = x - center[0]
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])
    return np.dot(norm3d, points), norm3d


def correspondence_matrix(p1, p2):
    """
    Each row in the A matrix below is constructed as
    [x'*x, x'*y, x', 
     y'*x, y'*y, y', 
     x, y, 1]
    """
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]
    return np.array([
        p1x * p2x, p1x * p2y, p1x, 
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T


def calc_fundamental_matrix(K: np.ndarray, pt1: np.ndarray, pt2: np.ndarray):
    """Calculate E with 2D corresponded points using normalized 8-point algorithm
        Result will be up to a scale
    Args:
        K: (3, 3)
        pt1: (#points, 2)
        pt2: (#points, 2)
    Returns:
        Essential matrix (3, 3)
    """
    # first normalize points
    pt1_normed = np.dot(inv(K), to_homo(pt1.T)) # (3, #points)
    pt2_normed = np.dot(inv(K), to_homo(pt2.T)) # (3, #points)

    # normalize points
    pt1n, T1 = scale_and_translate_points(pt1_normed)
    pt2n, T2 = scale_and_translate_points(pt2_normed)

    # compute E with 8-point algorithm
    A = correspondence_matrix(pt1n, pt2n)

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    
    # reverse preprocessing of coordinates
    # we know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))
    return F / F[2, 2]


def compute_P_from_F(F):
    """Compute the second camera matrix (assuming P1 = [I 0]) from F
    """
    e = compute_epipole(F.T)    # left epipole
    Te = skew(e)
    return np.vstack((np.dot(Te, F.T).T, e)).T


class FundamentalSolver:
    def __init__(self, method='opencv', reproj_threshold=0.2, ransac_iters=1000):
        self.reproj_threshold = reproj_threshold
        self.ransac_iters = ransac_iters
        self.method = method

    def __call__(self, kpts1, kpts2):
        if self.method == 'opencv':
            F, mask = cv2.findFundamentalMat(kpts1, kpts2,
                                             method=cv2.USAC_MAGSAC,
                                             ransacReprojThreshold=self.reproj_threshold,
                                             confidence=0.999999,
                                             maxIters=self.ransac_iters)
        return F, mask.ravel()
    
    def recover_pose(self, kpts1, kpts2, K):
        F, mask = self(kpts1, kpts2)
        kpts1, kpts2 = kpts1[mask == 1], kpts2[mask == 1]

        E = K.T @ F @ K

        # # normalize E 
        # U, S, Vt = np.linalg.svd(E)
        # S = [1, 1, 0]   # enforce singular values for essential matrix
        # E = U @ np.diag(S) @ Vt

        if self.method == 'opencv':
            _, R, t, _ = cv2.recoverPose(E, kpts1, kpts2, K)
        
        return R, t