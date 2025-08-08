from pysfm import *
from pysfm.utils.algebra import normalize


def compute_reprojection_error(gt_pixels, proj_pixels):
    # euclidean distance per point
    errors = np.linalg.norm(gt_pixels - proj_pixels, axis=1)
    # mean reprojection error
    avg_error = np.mean(errors)
    return avg_error



def quat_angle_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    assert gt.shape == (4,)
    assert pred.shape == (4,)

    if len(gt.shape) == 1:
        gt = np.expand_dims(gt, axis=0) # (1, 4)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)

    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = gt / np.linalg.norm(gt, axis=1, keepdims=True)

    d = np.abs(np.sum(np.multiply(q1, q2), axis=1, keepdims=True))
    d = np.clip(d, a_min=-1, a_max=1)
    angle = 2. * np.degrees(np.arccos(d))
    return angle


def precision_recall(inliers, tp, failures):
    """
        Computes P/R plot for a set of poses given inliers (confidence) and 
    whether the estimated pose error (whatever it may be) is within a threshold.

    Each point in the plot is obtained by choosing a threshold for inliers (i.e. inlier_thr).

    Recall measures how many images have inliers >= inlier_thr
    Precision measures how many images that have inliers >= inlier_thr have estimated pose error <= pose_threshold (measured by counting tps)

    where, pose_threshold = (trans_thr[m], rot_thr[deg])
    """
    assert len(inliers) == len(tp)

    # sort by inliers (descending order)
    inliers = np.array(inliers)
    sort_idx = np.argsort(inliers)[::-1]
    inliers = inliers[sort_idx]
    tp = np.array(tp).reshape(-1)[sort_idx]

    # get idxs where inliers change (avoid tied up values)
    distinct_value_indices = np.where(np.diff(inliers))[0]
    threshold_idxs = np.r_[distinct_value_indices, inliers.size - 1]

    # compute P/R
    N = inliers.shape[0]
    rec = np.arange(N, dtype=np.float32) + 1
    cum_tp = np.cumsum(tp)
    prec = cum_tp[threshold_idxs] / rec[threshold_idxs]
    rec = rec[threshold_idxs] / (float(N) + float(failures))

    # invert order and ensures (prec=1, rec=0) point
    last_ind = rec.searchsorted(rec[-1])
    sl = slice(last_ind, None, -1)
    prec = np.r_[prec[sl], 1]
    rec = np.r_[rec[sl], 0]

    # compute average precision (AUC) as the weighted average of precisions
    average_precision = np.abs(np.sum(np.diff(rec) * np.array(prec)[:-1]))

    return prec, rec, average_precision


def pose_error(R_gt: np.ndarray, t_gt: np.ndarray, R: np.ndarray, t: np.ndarray, scale=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15
    err_abs_t = 0

    if scale != None:
        t_gt = scale * t_gt
        t = np.linalg.norm(t_gt) * normalize(t)
        err_abs_t = np.linalg.norm(t - t_gt)
    
    R2R1 = R_gt @ R.T
    cos_angle = max(min(1.0, 0.5 * (np.trace(R2R1 - 1.0))), -1.0)
    err_r = math.acos(cos_angle)

    t = normalize(t)
    t_gt = normalize(t_gt)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    return err_r, err_t, err_abs_t