import numpy as np

from ..utils.validation import check_array, check_is_fitted
from ..metrics.pairwise import linear_kernel, rbf_kernel, sigmoid_kernel, polynomial_kernel
from ..QuantumUtility.Utility import tomography
from ._base import BaseEstimator




class QLSSVC(BaseEstimator):
    """
    LSSVC implementation. 
    "Least squares Support Vector Machine Classifier" Suykens, Vandewalle

    Parameters
    ----------

    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='linear'
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid'.
         If none is given, 'linear' will be used.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    penalty: float, default=0.1
        relative weight of the training error.
    """

    def __init__(self, kernel='linear', penalty=0.1, degree=3, gamma='scale', coef0=0.0, k_eff=100) -> None:
        super().__init__()
        self.kernel = kernel
        self.penalty = penalty
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.k_eff = k_eff
        
        self.alpha = None
        self.b = None
        self.is_fitted_ = False
        self.X = None


    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            An instance of the estimator.
        """

        X, y = self._validate_data(X, y)

        self.X = X

        M = len(X)
        #construction of matrix F
        Z = self.get_kernel(X) + (self.penalty ** -1) * np.identity(M)
        
        F = np.block([[0, np.ones((1,M))],
                      [np.ones((M,1)), Z]])


        F_hat = F/np.trace(F)
        self.k_eff = np.linalg.cond(F_hat)

        u, sigma, v = np.linalg.svd(F_hat)
        sigma = np.where(sigma < 1/self.k_eff, 0, sigma)

        F_hat = u @ np.diag(sigma) @ v
    
        # solution is [b a] = F^-1 * [0 y]
        y = np.concatenate(([0], y))

        F_inv = np.linalg.pinv(F)

        sol = np.dot(F_inv, y)

        self.b = sol[0]
        self.alpha = sol[1:]

        self.is_fitted_ = True

        return self
        

    
    def predict(self, X):
        """Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """

        check_array(X)
        check_is_fitted(self)

        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            y_pred[i] = np.sign(np.dot(self.alpha, self.get_kernel([X[i][:]], self.X)[0]) + self.b)

        return y_pred


    
    def get_kernel(self, X, Y=None):
        if self.kernel == 'linear':
            return linear_kernel(X=X, Y=Y)

        elif self.kernel == 'poly':
            gamma = self._get_gamma()
            return polynomial_kernel(X=X, Y=Y, degree=self.degree, gamma=gamma, coef0=self.coef0)
        
        elif self.kernel == 'rbf':
            gamma = self._get_gamma()
            return rbf_kernel(X=X, Y=Y, gamma=gamma)

        elif self.kernel == 'sigmoid':
            gamma = self._get_gamma()
            return sigmoid_kernel(X=X, Y=Y, gamma=gamma, coef0=self.coef0)



    def _get_gamma(self, X):
        gamma=None
        if self.gamma == 'scale':
            gamma = 1 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            gamma = 1 / self.n_features_in_
        
        return gamma