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
        self.cond = None


    def _classical_fit(self, y):
        N = len(self.X)
        
        y = np.append(0,y)
        F = np.r_[[np.append(0,np.ones(N))], np.c_[np.ones(N), self.get_kernel(self.X) + (self.penalty ** -1) * np.identity(N)]]
        self.cond = np.linalg.cond(F)
        F = np.linalg.pinv(F, hermitian=True)
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
        
        spectral_norm = np.linalg.norm(X, ord=2)
        fro_norm = np.linalg.norm(X)
        p = spectral_norm**2 / fro_norm**2
        self.alpha_F = 1/p

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
    
    def get_classifcation_complexity(self, X, relative_error=False, epsilon=1):
        if relative_error:
            betas = self.get_betas(X)
            hs = np.abs(self.get_h(X))
            Ps = self.get_P(X)

            return (self.cond * (betas - hs) * self.alpha_F) / (epsilon * hs * np.sqrt(Ps))
        else:
            betas = self.get_betas(X)
            hs = np.abs(self.get_h(X))
            Ps = self.get_P(X)

            return (self.cond * betas * self.alpha_F) / epsilon

    def get_all_attributes(self, X):
        betas = self.get_betas(X)
        hs = self.get_h(X)
        Ps = self.get_P(X)
        cond = self.cond
        rel_comp = (cond * (betas - np.abs(hs)) * self.alpha_F) / (np.abs(hs) * np.sqrt(Ps))
        abs_comp = (cond * betas * self.alpha_F)

        return (betas, hs, Ps, cond, rel_comp, abs_comp)



    
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



class VapnikLSSVC():
    """A class that implements the Least Squares Support Vector Machine 
    for classification tasks.

    It uses Numpy pseudo-inverse function to solve the dual optimization 
    problem with ordinary least squares. In multiclass classification 
    problems the approach used is one-vs-all, so, a model is fit for each 
    class while considering the others a single set of the same class.
    
    # Parameters:
    - gamma: float, default = 1.0
        Constant that control the regularization of the model, it may vary 
        in the set (0, +infinity). The closer gamma is to zero, the more 
        regularized the model will be.
    - kernel: {'linear', 'poly', 'rbf'}, default = 'rbf'
        The kernel used for the model, if set to 'linear' the model 
        will not take advantage of the kernel trick, and the LSSVC maybe only
        useful for linearly separable problems.
    - kernel_params: **kwargs, default = depends on 'kernel' choice
        If kernel = 'linear', these parameters are ignored. If kernel = 'poly',
        'd' is accepted to set the degree of the polynomial, with default = 3. 
        If kernel = 'rbf', 'sigma' is accepted to set the radius of the 
        gaussian function, with default = 1. 
     
    # Attributes:
    - All hyperparameters of section "Parameters".
    - alpha: ndarray of shape (1, n_support_vectors) if in binary 
             classification and (n_classes, n_support_vectors) for 
             multiclass problems
        Each column is the optimum value of the dual variable for each model
        (using the one-vs-all approach we have n_classes == n_classifiers), 
        it can be seen as the weight given to the support vectors 
        (sv_x, sv_y). As usually there is no alpha == 0, we have 
        n_support_vectors == n_train_samples.
    - b: ndarray of shape (1,) if in binary classification and (n_classes,) 
         for multiclass problems 
        The optimum value of the bias of the model.
    - sv_x: ndarray of shape (n_support_vectors, n_features)
        The set of the supporting vectors attributes, it has the shape 
        of the training data.
    - sv_y: ndarray of shape (n_support_vectors, n)
        The set of the supporting vectors labels. If the label is represented 
        by an array of n elements, the sv_y attribute will have n columns.
    - y_labels: ndarray of shape (n_classes, n)
        The set of unique labels. If the label is represented by an array 
        of n elements, the y_label attribute will have n columns.
    - K: function, default = rbf()
        Kernel function.
    """
    
    def __init__(self, gamma=1): 
        # Hyperparameters
        self.gamma = gamma
        
        # Model parameters
        self.alpha = None
        self.b = None
        self.sv_x = None
        self.sv_y = None
        self.y_labels = None
        
    
    def _optimize_parameters(self, X, y_values):
        """Help function that optimizes the dual variables through the 
        use of the kernel matrix pseudo-inverse.
        """
        sigma = np.multiply(y_values*y_values.T, self.get_kernel(X))
        
        A = np.block([
            [0, y_values.T],
            [y_values, sigma + self.gamma**-1 * np.eye(len(y_values))]
        ])
        B = np.array([0]+[1]*len(y_values))
        
        A_cross = np.linalg.pinv(A)

        solution = np.dot(A_cross, B)
        b = solution[0]
        alpha = solution[1:]
        
        return (b, alpha)
    
    def fit(self, X, y):
        """Fits the model given the set of X attribute vectors and y labels.
        - X: ndarray of shape (n_samples, n_attributes)
        - y: ndarray of shape (n_samples,) or (n_samples, n)
            If the label is represented by an array of n elements, the y 
            parameter must have n columns.
        """
        y_reshaped = y.reshape(-1,1) if y.ndim==1 else y

        self.sv_x = X
        self.sv_y = y_reshaped
        self.y_labels = np.unique(y_reshaped, axis=0)
        
        if len(self.y_labels) == 2: # binary classification
            # converting to -1/+1
            y_values = np.where(
                (y_reshaped == self.y_labels[0]).all(axis=1)
                ,-1,+1)[:,np.newaxis] # making it a column vector
            
            self.b, self.alpha = self._optimize_parameters(X, y_values)
        
        else: # multiclass classification, one-vs-all approach
            n_classes = len(self.y_labels)
            self.b = np.zeros(n_classes)
            self.alpha = np.zeros((n_classes, len(y_reshaped)))
            
            for i in range(n_classes):
                # converting to +1 for the desired class and -1 for all 
                # other classes
                y_values = np.where(
                    (y_reshaped == self.y_labels[i]).all(axis=1)
                    ,+1,-1)[:,np.newaxis]
  
                self.b[i], self.alpha[i] = self._optimize_parameters(X, y_values)
        
    def predict(self, X):
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        if self.alpha is None:
            raise Exception(
                "The model doesn't see to be fitted, try running .fit() method first"
            )

        X_reshaped = X.reshape(1,-1) if X.ndim==1 else X
        KxX = self.K(self.sv_x, X_reshaped)
        
        if len(self.y_labels)==2: # binary classification
            y_values = np.where(
                (self.sv_y == self.y_labels[0]).all(axis=1),
                -1,+1)[:,np.newaxis]

            y = np.sign(np.dot(np.multiply(self.alpha, y_values.flatten()), KxX) + self.b)
            
            y_pred_labels = np.where(y==-1, self.y_labels[0], self.y_labels[1])
        
        else: # multiclass classification, one-vs-all approach
            y = np.zeros((len(self.y_labels), len(X)))
            for i in range(len(self.y_labels)):
                y_values = np.where(
                    (self.sv_y == self.y_labels[i]).all(axis=1),
                    +1, -1)[:,np.newaxis]
                y[i] = np.dot(np.multiply(self.alpha[i], y_values.flatten()), KxX) + self.b[i]
            
            predictions = np.argmax(y, axis=0)
            y_pred_labels = np.array([self.y_labels[i] for i in predictions])
            
        return y_pred_labels
    
    def get_kernel(self, X, Y=None):
        return linear_kernel(X=X, Y=Y)
        