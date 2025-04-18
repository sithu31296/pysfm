from pysfm import *
from pysfm.utils.geometry import inv, to_homo



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


# def calc_essential_matrix(K: np.ndarray, pt1: np.ndarray, pt2: np.ndarray):
#     """Calculate E with 2D corresponded points using normalized 8-point algorithm
#         Result will be up to a scale
#     Args:
#         K: (3, 3)
#         pt1: (#points, 2)
#         pt2: (#points, 2)
#     Returns:
#         Essential matrix (3, 3)
#     """
#     # normalize points
#     pt1n, T1 = scale_and_translate_points(pt1)
#     pt2n, T2 = scale_and_translate_points(pt2)

#     # compute E with 8-point algorithm
#     A = correspondence_matrix(pt1n, pt2n)

#     # compute linear least square solution
#     U, S, V = np.linalg.svd(A)
#     F = V[-1].reshape(3, 3)

#     # constrain F. Make rank 2 by zeroing out last singular value
#     U, S, V = np.linalg.svd(F)
#     S = [1, 1, 0]   # force rank 2 and equal eigenvalues
#     F = np.dot(U, np.dot(np.diag(S), V))
    
#     # reverse preprocessing of coordinates
#     # we know that P1' E P2 = 0
#     F = np.dot(T1.T, np.dot(F, T2))
#     return F / F[2, 2]


#     # E, mask = cv2.findEssentialMat(pt1, pt2, K, cv2.RANSAC, prob=0.999, threshold=1.0)
#     # pass_count, R, T, mask, s = cv2.recoverPose(pt1, pt2, K, None, K, None, E=E, mask=mask)
#     # print(pass_count)
#     # print(s)
#     # return R, T, mask


def compute_E(K: np.ndarray, pt1: np.ndarray, pt2: np.ndarray):
    """Calculate E with 2D corresponded points using normalized 8-point algorithm
        Result will be up to a scale
    Args:
        K: (3, 3)
        pt1: (#points, 2)
        pt2: (#points, 2)
    Returns:
        Essential matrix (3, 3)
    """
    E, mask = cv2.findEssentialMat(pt1, pt2, K, cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask


def compute_P_from_E(E: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, K: np.ndarray, mask: np.ndarray):
    """Compute the second camera matrix (assuming P1 = [I 0]) from E
    E = [t]R
    Returns:
        list of 4 possible camera matrices
    """
    pass_count, R, T, mask = cv2.recoverPose(E, pt1, pt2, K, mask=mask)
    return R, T, mask


# def compute_P_from_E(E):
#     """Compute the second camera matrix (assuming P1 = [I 0]) from E
#     E = [t]R
#     Returns:
#         list of 4 possible camera matrices
#     """
#     U, S, V = np.linalg.svd(E)
#     # ensure rotation matrix are right-handed with positive determinant
#     if np.linalg.det(np.dot(U, V)) < 0:
#         V = -V
    
#     # create 4 possible camera matrices
#     W = np.array([
#         [0, -1, 0],
#         [1, 0, 0],
#         [0, 0, 1]
#     ])
#     P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
#           np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
#           np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
#           np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]
#     return P2s

#     pass_count, R, T, mask, s = cv2.recoverPose(pt1, pt2, K, None, K, None, E=E, mask=mask)
#     print(pass_count)
#     print(s)