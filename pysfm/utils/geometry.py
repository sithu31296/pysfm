from pysfm import *


def to_homo(matrix):
    # matrix in shape [3, #points]
    return np.vstack([matrix, np.ones((1, matrix.shape[1]))])


def to_cart(matrix):
    # matrix in shape [3 or 4, #points]
    dim = matrix.shape[0]
    return matrix[:dim-1] / matrix[-1]

def inv(matrix):
    if isinstance(matrix, torch.Tensor):
        return torch.linalg.inv(matrix)
    return np.linalg.inv(matrix)


def skew(x: np.ndarray):
    """Create a skew symmetric matrix A from a 3D vector x
    Property: np.cross(A, v) == np.dot(x, v)
    Args:
        x: 3D vector
    Returns:
        3x3 skew symmetric matrix from x
    """
    return np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],     0]
    ])


def reconstruct_one_point(p1, p2, m1, m2):
    """
        p1 and m1*X are parallel and cross product = 0
        p1 x m1*X = p2 x m2*X = 0
    """
    A = np.vstack([
        np.dot(skew(p1), m1),
        np.dot(skew(p2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])
    return P / P[3]


def reconstruct_points(p1, p2, m1, m2):
    num_points = p1.shape[1]
    results = np.ones((4, num_points))
    for i in range(num_points):
        results[:, i] = reconstruct_one_point(p1[:, i], p2[:, i], m1, m2)
    return results



def linear_triangulation(p1, p2, m1, m2):
    """Linear triangulation to find the 3D point X where p1 = m1*X and p2 = m2*X
    Solve AX = 0
    Args:
        p1, p2: 2D points in homogeneous coordinates (3, #points)
        m1, m2: camera matrices associated with p1 and p2 (3, 4)
    Returns:
        (4, n) homogeneous 3D triangulated points
    """
    num_points = p1.shape[1]
    results = np.ones((4, num_points))
    for i in range(num_points):
        A = np.array([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])
        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        results[:, i] = X / X[3]
    return results


def compute_epipole(F):
    """Computes the right epipole from a fundamental matrix F.
    Use with F.T for left epipole
    """
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]


def backprojection(depth, focal, cu=None, cv=None, mask=None):
    H, W = depth.shape
    if cu is None:
        cu = (W - 1) * 0.5
    if cv is None:
        cv = (H - 1) * 0.5

    grid_x, grid_y = np.meshgrid(*[np.arange(0, s) for s in (W, H)])
    pts3d = np.zeros((H, W, 3))
    pts3d[..., 0] = depth * (grid_x - cu) / focal     # (u - cu) / f * d
    pts3d[..., 1] = depth * (grid_y - cv) / focal     # (v - cv) / f * d
    pts3d[..., 2] = depth
    if mask is not None:
        pts3d = pts3d[mask]
    return pts3d
    