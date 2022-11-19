from tabnanny import verbose
import numpy as np

from ..utils.validation import check_array, check_is_fitted
from ..metrics.pairwise import linear_kernel, rbf_kernel, sigmoid_kernel, polynomial_kernel
from ..QuantumUtility.Utility import tomography
from ._base import BaseEstimator



class QLSSVC(BaseEstimator):
    """
    QLSSVC implementation. 
    "Quantum Least squares Support Vector Machine Classifier" Rebentrost, TODO

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

    def __init__(self, kernel='linear', penalty=0.1, degree=3, gamma='scale', coef0=0.0, k_eff=100, delta=0.01, verbose=False) -> None:
        super().__init__()
        self.kernel = kernel
        self.penalty = penalty
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.k_eff = k_eff
        self.delta = delta
        self.verbose = verbose
        
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
        #
        #        [ (X * X^T  +  gamma^-1 * I)^-1  *  X^T ]   * y
        #     ------------------------------------------------------  ~= H^-1 * y
        #       || (X * X^T  +  gamma^-1 * I)^-1  *  X^T    * y ||
        #

        X, y = self._validate_data(X, y)
        self.X = X

        M = len(X)
        H = self.get_kernel(X) + (self.penalty ** -1) * np.identity(M)
        condition_number = self.get_condition_number()
        if self.verbose:
            print(f"condition number: {condition_number}")
        H_pinv = np.linalg.pinv(H, rcond=1/condition_number)
        
        if self.verbose:
            print("Computing eta...")
        # eta = (H^-1 * y)/|| (H^-1)*y || -- following corollary 36
        # where H = X^T * X   +   1/gamma * I
        eta_ = H_pinv @ y
        eta_norm = np.linalg.norm(eta_, ord=2)
        eta = eta_ / eta_norm
        if self.verbose:
            print("Computing nu...")
        # nu = (H^-1 * vec(1))/|| (H^-1)*vec(1) || -- following corollary 36
        nu_ = H_pinv @ np.ones(M)
        nu_norm = np.linalg.norm(nu_, ord=2)
        nu = nu_ / nu_norm

        s = np.inner(y, eta)

        if self.verbose:
            print("Computing b and alpha...")
        self.b = (np.inner(eta, np.ones(M))) / s

        self.alpha = nu - (eta * self.b)

        if self.verbose:
            print(f"Intercept: {self.b}\nFitted!")
        return self



    
    def fit_fix(self, X, y):
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
        #
        #        [ (X * X^T  +  gamma^-1 * I)^-1  *  X^T ]   * y
        #     ------------------------------------------------------  ~= H^-1 * y = eta
        #       || (X * X^T  +  gamma^-1 * I)^-1  *  X^T    * y ||
        #

        X, y = self._validate_data(X, y)
        self.X = X

        M = len(X)
        H = self.get_kernel(X) + (self.penalty ** -1) * np.identity(M)
        condition_number = self.get_condition_number()
        if self.verbose:
            print(f"condition number: {condition_number}")
        H_pinv = np.linalg.pinv(H)
        
        if self.verbose:
            print("Computing eta...")
        # eta = (H^-1 * y)/|| (H^-1)*y || -- following corollary 36
        # where H = X^T * X   +   1/gamma * I
        eta = H_pinv @ y
        eta_norm = np.linalg.norm(eta, ord=2)
        eta = eta / eta_norm
        if self.verbose:
            print("Computing nu...")
        # nu = (H^-1 * vec(1))/|| (H^-1)*vec(1) || -- following corollary 36
        nu = H_pinv @ np.ones(M)
        nu_norm = np.linalg.norm(nu, ord=2)
        nu = nu / nu_norm

        s = np.inner(y, eta)

        if self.verbose:
            print("Computing b and alpha...")
        self.b = (np.inner(eta, np.ones(M))) / s

        self.alpha = eta - (nu * self.b)

        if self.verbose:
            print(f"Intercept: {self.b}\nFitted!")
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
            y_pred[i] = np.sign(np.dot(self.alpha, self.get_kernel(self.X, [X[i][:]])) + self.b)

        return y_pred


    def get_condition_number(self):
        if self.verbose:
            print("Start evaluating condition number:")
        norm = np.linalg.norm(x=self.X,ord=2)
        if self.verbose:
            print(f"\tThe norm of the matrix X is: {norm}")
        return self.k_eff * np.sqrt((norm**2 + self.penalty)/(norm**2 + self.penalty * self.k_eff**2))

    # TODO evaluate error
    def get_error(self):

        return
    
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