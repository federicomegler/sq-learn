from tabnanny import verbose
import numpy as np

from ..utils.validation import check_array, check_is_fitted
from ..metrics.pairwise import linear_kernel, rbf_kernel, sigmoid_kernel, polynomial_kernel
from ..QuantumUtility.Utility import tomography
from ._base import BaseEstimator
from ..metrics import accuracy_score



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

    def __init__(self, kernel='linear', penalty=0.1, degree=3, gamma='scale', coef0=0.0, verbose=False, algorithm='classic') -> None:
        super().__init__()
        self.kernel = kernel
        self.penalty = penalty
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.verbose = verbose
        self.algorithm = algorithm
        
        self.alpha = None
        self.b = None
        self.is_fitted_ = False
        self.X = None
        self.normX = None
        self.coef_ = None
        self.n_features_in_ = None


    def _classical_fit(self, y):
        N = len(self.X)

        #construction of matrix F
        Z = self.get_kernel(self.X) + (self.penalty ** -1) * np.identity(N)
        F = np.r_[[np.append(0,np.ones(N))], np.c_[np.ones(N), Z]]
        
        # solution is [b a] = F^-1 * [0 y]
        y = np.append(0,y)
        F_inv = np.linalg.pinv(F)

        sol = np.dot(F_inv, y)
        
        return sol[0], sol[1:]
    
    def _cg_fit():
        return

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

        if self.algorithm == 'classic':
            self.b, self.alpha = self._classical_fit(y)
        elif self.algorithm == '':
            return


        if self.kernel == 'linear':
            N, d = self.X.shape
            self.coef_ = np.zeros(d)
            for i in range(N):
                ay = self.alpha[i]
                w = ay * self.X[i]
                self.coef_ = np.add(self.coef_, w)
        

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

        P = self.get_P(X)
        y_pred = np.where(P <= 0.5, 1., -1.)

        return y_pred
    
    def classical_predict(self, X):
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
    
    def get_h(self, X):
        check_array(X)
        check_is_fitted(self)

        h = np.zeros(len(X))
        for i in range(len(X)):
            h[i] = np.dot(self.alpha, self.get_kernel(self.X, [X[i][:]])) + self.b

        return h
    
    def get_betas(self, X):

        return
    
    def get_P(self, X):
        check_array(X)
        check_is_fitted(self)
        N = len(self.X)

        P = np.zeros(len(X))
        for i in range(len(X)):
            h = np.dot(self.alpha, self.get_kernel(self.X, [X[i][:]])) + self.b
            Nx = N*np.linalg.norm(X[i], ord=2) + 1
            Nu = self.b ** 2
            for j in range(len(self.X)):
                Nu = Nu + np.sum((self.alpha[j]**2) * (np.linalg.norm(self.X[i])**2))
            beta = np.sqrt(Nx * Nu)
            P[i] = 0.5 * (1 - h / beta)
        
        return P

    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        check_is_fitted(self)

        y_pred = self.predict(X=X)
        return accuracy_score(y_true=y, y_pred=y_pred)
        



    
    def get_kernel(self, X, Y=None):
        if self.kernel == 'linear':
            return linear_kernel(X=X, Y=Y)

        elif self.kernel == 'poly':
            gamma = self._get_gamma(X)
            return polynomial_kernel(X=X, Y=Y, degree=self.degree, gamma=gamma, coef0=self.coef0)
        
        elif self.kernel == 'rbf':
            gamma = self._get_gamma(X)
            return rbf_kernel(X=X, Y=Y, gamma=gamma)

        elif self.kernel == 'sigmoid':
            gamma = self._get_gamma(X)
            return sigmoid_kernel(X=X, Y=Y, gamma=gamma, coef0=self.coef0)


    def _get_gamma(self, X):
        X = np.asarray(X)
        gamma=None
        if self.gamma == 'scale':
            gamma = 1 / (X.shape[1] * X.var())
            if self.verbose:
                print(f"Gamma: {gamma}, shape: {X.shape[1]}, Variance: {X.var()}")
        elif self.gamma == 'auto':
            gamma = 1 / self.n_features_in_
            if self.verbose:
                print(f"Gamma: {gamma}")
        return gamma
