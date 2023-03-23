from tabnanny import verbose
import numpy as np
from scipy.linalg import ishermitian

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
        self.Nu = None


    def _classical_fit(self, y):
        N = len(self.X)
        
        y = np.append(0,y)
        F = np.linalg.pinv(np.r_[[np.append(0,np.ones(N))], np.c_[np.ones(N), self.get_kernel(self.X) + (self.penalty ** -1) * np.identity(N)]]
                           , hermitian=True)
        sol = np.dot(F, y)
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

        self.Nu = self.b ** 2 + np.sum([self.alpha[index]**2 * np.linalg.norm(x)**2 for index, x in enumerate(self.X)])
            

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

        return np.asarray([(np.dot(self.alpha, self.get_kernel(self.X, [x])) + self.b)[0] for x in X])
    
    def get_betas(self, X):
        check_array(X)
        check_is_fitted(self)
        N = len(self.X)
        return np.asarray([np.sqrt((N*np.linalg.norm(x) + 1) * self.Nu) for x in X])
    
    def get_P(self, X):
        check_array(X)
        check_is_fitted(self)
        N = len(self.X)
        h = self.get_h(X)
        beta = self.get_betas(X)
        P = 0.5 * (1 - h/beta)

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
