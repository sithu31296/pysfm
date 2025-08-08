from pysfm import *

EPS = 1e-15

def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + EPS)