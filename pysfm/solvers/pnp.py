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