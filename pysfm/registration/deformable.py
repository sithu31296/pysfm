from pysfm import *



def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))


def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).

    Attributes
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)




class DeformableRegistration:
    """
    Modified from https://github.com/siavashk/pycpd/blob/master/pycpd/deformable_registration.py

    """
    def __init__(self, source, target):
        self.alpha = 1  # higher -> more rigid, lower -> more flexible
        self.beta = 10
        
        self.low_rank = False
        self.num_eig = 50
        self.source = source
        self.target = target
        self.source_tf = source
        self.sigma2 = initialize_sigma2(target, source) 
        self.N, self.D = self.target.shape
        self.M, _ = self.source.shape
        self.tolerance = 0.001
        self.w = 0.0
        self.max_iterations = 300
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N,))
        self.P1 = np.zeros((self.M,))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0

        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.source, self.beta)
        
        if self.low_rank:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1 / self.S)
            self.S = np.diag(self.S)
            self.E = 0.

    def register(self):
        self.transform_point_cloud()

        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
        
        return self.source_tf, self.get_registration_parameters()

    def update_transform(self):
        if self.low_rank:
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.source)

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

        else:
            A = np.dot(np.diag(self.P1), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.source)
            self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, source=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
            
        """
        if source is not None:
            G = gaussian_kernel(X=source, beta=self.beta, Y=self.source)
            return source + np.dot(G, self.W)
        else:
            if self.low_rank is True:
                self.source_tf = self.source + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
            else:
                self.source_tf = self.source + np.dot(self.G, self.W)

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

    def get_registration_parameters(self, new_source=None):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        if new_source is not None:
            G = gaussian_kernel(X=new_source, beta=self.beta, Y=self.source)
            return G @ self.W
        return self.G @ self.W
    
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