from pysfm import *
from roma import rigid_points_registration



def rigid_points_registration_numpy(src_pts: np.ndarray, tgt_pts: np.ndarray, weights: np.ndarray = None, compute_scaling: bool = False):
    if weights is None:
        weights = np.ones(src_pts.shape[0])
    
    src_pts_mean = np.mean(weights[..., None] * src_pts, axis=0)
    tgt_pts_mean = np.mean(weights[..., None] * tgt_pts, axis=0)

    src_pts_centered = src_pts - src_pts_mean
    tgt_pts_centered = tgt_pts - tgt_pts_mean
    weights /= (weights.sum() + 1e-12)

    cov = (weights[:, None] * src_pts_centered).T @ tgt_pts_centered
    U, S, Vh = np.linalg.svd(cov)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2, :] *= -1
        R = Vh.T @ U.T

    if compute_scaling:
        scale = np.sum(S) / np.trace((weights[:, None] * src_pts_centered).T @ src_pts_centered)
        t = tgt_pts_mean - scale * (src_pts_mean @ R.T)
        return R, t, scale
    else:
        t = tgt_pts_mean - (src_pts_mean @ R.T)
        return R, t
    



class RigidRegistration:
    def __init__(self, source, target, compute_scale=False):
        self.source = source
        self.target = target
        self.source_tf = source
        self.dim, _ = self.source.shape
        self.N, self.D = self.target.shape

        self.R = np.eye(self.dim)
        self.t = np.zeros((self.D, 1))
        self.s = 1
        self.compute_scale = compute_scale


    def register(self):
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
        
        return self.source_tf, self.get_registration_parameters()

    def update_transform(self):
        pass

    def transform_point_cloud(self, source=None):
        if source is not None:
            source_tf = self.s * (self.R @ source.T) + self.t
            return source_tf.T
        else:
            source_tf = self.s * (self.R @ self.source.T) + self.t
            self.source_tf = source_tf.T

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf

        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.target, self.target), axis=1))
        yPy = np.dot(np.transpose(self.P1),  np.sum(
            np.multiply(self.source_tf, self.source_tf), axis=1))
        trPXY = np.sum(np.multiply(self.source_tf, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        trf = np.eye(4)
        trf[:3, :3] = self.s * self.R
        trf[:3, -1] = self.t[:, 0]
        return trf
    
    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        P = np.sum((self.target[None, :, :] - self.source_tf[:, None, :])**2, axis=2) # (M, N)
        P = np.exp(-P/(2*self.sigma2))
        c = (2*np.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

        den = np.sum(P, axis = 0, keepdims = True) # (1, N)
        den = np.clip(den, np.finfo(self.target.dtype).eps, None) + c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.target)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()