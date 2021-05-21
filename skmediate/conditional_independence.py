# scikit-learn class for cross-covariance statistic

import numpy as np
from scipy import linalg


class ConditionalCrossCovariance(object):
    """
    Implements conditional dependence testing between multivariate quantities.

    """

    def __init__(
        self,
        regression_estimator=None,
        covariance_estimator=None,
        precision_estimator=None,
    ):
        """
        Initializes ConditionalCrossCovariance with base estimators.


        """

        # Note: covariance_estimator should be initialized with assume_centered=True
        pass

    def fit(self, X, Z, Y):
        """

        Fits a conditional covariance matrix

        Parameters
        ----------
        X,Z,Y : ndarray
            Input data matrices. ``X``, ``Z`` and ``Y`` must have the same number of
            samples. That is, the shapes must be ``(n, r)``, ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples, `p` and `q` are the number of
            dimensions of ``Z`` and ``Y`` respectively.
        Returns
        -------
        self : object
        """

        # Step 1: Residualize with regression

        # TODO: Check regression type for supporting single or multi-output regression
        regfit_xy = self.regression_estimator.fit(X, Y)
        regfit_xz = self.regression_estimator.fit(X, Z)

        # Compute residualized Zs and Ys.
        self.residualized_Z_ = Z - regfit_xz.predict(X)
        self.residualized_Y_ = Y - regfit_xy.predict(X)

        # Step 2: Covariance estimation
        W = np.concatenate((self.residualized_Z_, self.residualized_Y_), axis=1)
        self.covfit_zy_ = self.covariance_estimator.fit(W)

        rows_Z, cols_Z = self.residualized_Z_.shape
        rows_Y, cols_Y = self.residualized_Y_.shape

        self.cov_zz_ = self.covfit_zy_.covariance_[0 : cols_Z - 1, 0 : cols_Z - 1]
        self.cov_zy_ = self.covfit_zy_.covariance_[0 : cols_Z - 1, cols_Z:]
        self.cov_yy_ = self.covfit_zy_.covariance_[cols_Z:, cols_Z:]

        # Estimate inverse of ZZ if dimensionality is small
        self.prec_zz_ = linalg.pinvh(self.cov_zz_, check_finite=False)

        self.residual_crosscovariance_ = np.matmul(
            np.matmul(self.cov_zy_, self.prec_zz_), np.transpose(self.cov_zy_)
        )

        return self
