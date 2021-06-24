"""
Classes for computations of conditional independence.
"""
import collections
import inspect
import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EmpiricalCovariance


class ConditionalCrossCovariance(object):
    """Conditional dependence testing between multivariate quantities. """

    def __init__(
        self,
        regression_estimator=None,
        covariance_estimator=None,
        precision_estimator=None,
    ):
        """
        Initializes ConditionalCrossCovariance with base estimators.

        Parameters
        ----------
        regression_estimator : sklearn estimator class or sequence.
            This class will be used to fit Y=f(X) and  X=f(X) and
            to generate residuals for covariance estimation.
            Default: :class:`sklearn.linear_model.LinearRegression`

        covariance_estimator : sklearn covariance estimator class.
            This class will be used to compute the covariance between
            the residuals the f(X) and the Y, Z.

        precision_estimator : sklearn covariance estimator class.
            Default: :class:`sklearn.covariance.EmpiricalCovariance`


        Notes
        -----
        .. [1] Wim Van der Elst, Ariel Abad Alonso, Helena Geys, Paul Meyvisch,
               Luc Bijnens, Rudradev Sengupta & Geert Molenberghs (2019)
               Univariate Versus Multivariate Surrogates in the Single-Trial
               Setting, Statistics in Biopharmaceutical Research, 11:3,
               301-310, DOI: 10.1080/19466315.2019.1575276
        """

        if regression_estimator is None:
            self.regression_estimator_xz = (
                self.regression_estimator_xy
            ) = LinearRegression()
        elif isinstance(regression_estimator, collections.Sequence):
            self.regression_estimator_xz = regression_estimator[0]
            self.regression_estimator_xy = regression_estimator[1]
        else:
            self.regression_estimator_xz = (
                self.regression_estimator_xy
            ) = regression_estimator

        if covariance_estimator is None:
            covariance_estimator = EmpiricalCovariance(assume_centered=True)

        if precision_estimator is None:
            precision_estimator = EmpiricalCovariance(assume_centered=True)

        self.covariance_estimator = covariance_estimator
        self.precision_estimator = precision_estimator

    def fit(self, X, Z, Y):
        """
        Fits a conditional covariance matrix.

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
        regfit_xy = self.regression_estimator_xy.fit(X, Y)
        regfit_xz = self.regression_estimator_xz.fit(X, Z)

        # Compute residualized Zs and Ys.
        self.residualized_Z_ = Z - regfit_xz.predict(X)
        self.residualized_Y_ = Y - regfit_xy.predict(X)

        # Step 2: Covariance estimation
        W = np.concatenate((self.residualized_Z_, self.residualized_Y_), axis=1)
        self.covfit_zy_ = self.covariance_estimator.fit(W)

        rows_Z, cols_Z = self.residualized_Z_.shape
        rows_Y, cols_Y = self.residualized_Y_.shape

        self.cov_zz_ = self.covfit_zy_.covariance_[0 : cols_Z - 1, 0 : cols_Z - 1]
        self.cov_yz_ = self.covfit_zy_.covariance_[cols_Z:, 0 : cols_Z - 1]
        self.cov_yy_ = self.covfit_zy_.covariance_[cols_Z:, cols_Z:]

        # Estimate inverse of ZZ if dimensionality is small:
        self.prec_zz_ = linalg.pinvh(self.cov_zz_, check_finite=False)

        self.residual_crosscovariance_ = np.matmul(
            np.matmul(self.cov_yz_, self.prec_zz_), np.transpose(self.cov_yz_)
        )

        return self
