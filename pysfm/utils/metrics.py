from pysfm import *



def compute_reprojection_error(gt_pixels, proj_pixels):
    # euclidean distance per point
    errors = np.linalg.norm(gt_pixels - proj_pixels, axis=1)
    # mean reprojection error
    avg_error = np.mean(errors)
    return avg_error