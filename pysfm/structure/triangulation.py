from pysfm import *
from pysfm.utils.geometry import to_cart


def triangulate_points(P1: np.ndarray, P2: np.ndarray, px1: np.ndarray, px2: np.ndarray):
    points4D = cv2.triangulatePoints(P1, P2, px1.T, px2.T)
    points3D = to_cart(points4D)
    return points3D.T