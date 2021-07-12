"""Classes for computations of conditional independence."""
import collections
import numpy as np
import warnings

from sklearn.base import clone, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.covariance import (
    EmpiricalCovariance,
    GraphicalLasso,
    GraphicalLassoCV,
    LedoitWolf,
    MinCovDet,
    OAS,
    ShrunkCovariance,
)
from sklearn.utils import shuffle
from tqdm.auto import trange
from inspect import getmro


COV_ESTIMATORS = {
    "empirical": EmpiricalCovariance(),
    "graphical_lasso": GraphicalLasso(),
    "graphical_lasso_cv": GraphicalLassoCV(),
    "ledoit_wolf": LedoitWolf(),
    "min_cov_det": MinCovDet(),
    "oas": OAS(),
    "shrunk": ShrunkCovariance(),
}


class ConditionalCrossCovariance(object):
    """Conditional dependence testing between multivariate quantities."""

    def __init__(
        self,
        regression_estimator=None,
        covariance_estimator=None,
        precision_estimator=None,
        residualized=False,
        estimate_p_value=False,
        n_shuffle=1000,
        show_progress=True,
    ):
        """
        Initialize ConditionalCrossCovariance with base estimators.

        Parameters
        ----------
        regression_estimator : sklearn estimator class or sequence.
            This class will be used to fit Y=f(X) and  X=f(X) and to generate
            residuals for covariance estimation.
            Default: :class:`sklearn.linear_model.LinearRegression`

        covariance_estimator : sklearn covariance estimator class.
            This class will be used to compute the covariance between the
            residuals of f(X) and the Y, Z. This may also be a string, one of
            ["empirical", "graphical_lasso", "graphical_lasso_cv",
            "ledoit_wolf", "min_cov_det", "oas", "shrunk"], to select one of the
            covariance classes from sklearn.covariance.
            Default: :class:`sklearn.covariance.EmpiricalCovariance`

        precision_estimator : sklearn covariance estimator class.
            This class will be used to compute the precision of the residualized
            Y and Z. This may also be a string, one of ["empirical",
            "graphical_lasso", "graphical_lasso_cv", "ledoit_wolf",
            "min_cov_det", "oas", "shrunk"], to select one of the covariance
            classes from sklearn.covariance.
            Default: :class:`sklearn.covariance.EmpiricalCovariance`

        residualized: bool
            If True, assume that ``Y`` and ``Z`` have already been residualized
            on ``X``.
            Default: False

        estimate_p_value: bool
            If True, perform repeated permutation tests on ``Y`` to estimate the
            p-value for the residual_cross_covariance_ score.
            Default: False

        n_shuffle: int
            The number of permutation tests to perform to estimate the p-value.
            Default: 1000

        show_progress: bool
            If True, show a progress bar while estimating the p-value
            Default: True

        Notes
        -----
        .. [1] Wim Van der Elst, Ariel Abad Alonso, Helena Geys, Paul Meyvisch,
               Luc Bijnens, Rudradev Sengupta & Geert Molenberghs (2019)
               Univariate Versus Multivariate Surrogates in the Single-Trial
               Setting, Statistics in Biopharmaceutical Research, 11:3,
               301-310, DOI: 10.1080/19466315.2019.1575276
        """
        if regression_estimator is None:
            self.regression_estimator_xz = LinearRegression()
            self.regression_estimator_xy = LinearRegression()
        elif isinstance(regression_estimator, collections.Sequence):
            if not all(isinstance(r, RegressorMixin) for r in regression_estimator):
                mro = [getmro(type(r)) for r in regression_estimator]
                raise ValueError(
                    f"regression_estimator must contain estimator instances that "
                    f"inherit from sklearn.base.RegressorMixin. Got "
                    f"{regression_estimator} instead. These instances have the "
                    f"following method resolution order:\n{mro}"
                )
            self.regression_estimator_xz = regression_estimator[0]
            self.regression_estimator_xy = regression_estimator[1]
        else:
            if not isinstance(regression_estimator, RegressorMixin):
                raise ValueError(
                    f"regression_estimator must inherit from "
                    f"sklearn.base.RegressorMixin. Got {regression_estimator} "
                    f"instead, which has the following method resolution "
                    f"order:\n{getmro(type(regression_estimator))}"
                )
            self.regression_estimator_xz = clone(regression_estimator)
            self.regression_estimator_xy = clone(regression_estimator)

        if covariance_estimator is None:
            covariance_estimator = EmpiricalCovariance(assume_centered=True)

        if precision_estimator is None:
            precision_estimator = EmpiricalCovariance(assume_centered=True)

        if isinstance(covariance_estimator, str):
            if covariance_estimator not in COV_ESTIMATORS.keys():
                raise ValueError(
                    f"If covariance_estimator is a string, it must be one of "
                    f"{COV_ESTIMATORS.keys()}. Got {covariance_estimator} "
                    f"instead."
                )

            self.covariance_estimator = clone(COV_ESTIMATORS[covariance_estimator])
        else:
            self.covariance_estimator = covariance_estimator

        if isinstance(precision_estimator, str):
            if precision_estimator not in COV_ESTIMATORS.keys():
                raise ValueError(
                    f"If precision_estimator is a string, it must be one of "
                    f"{COV_ESTIMATORS.keys()}. Got {precision_estimator} "
                    f"instead."
                )

            self.precision_estimator = clone(COV_ESTIMATORS[precision_estimator])
        else:
            self.precision_estimator = precision_estimator

        self.residualized = residualized
        self.estimate_p_value = estimate_p_value
        self.n_shuffle = n_shuffle
        self.show_progress = show_progress

    def fit(self, Z, Y, X=None, estimate_p_value=False):
        """
        Fits a conditional covariance matrix.

        Parameters
        ----------
        Z, Y, X : ndarray
            Input data matrices. ``Z``, ``Y``, and ``X`` must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            and ``(n, r)``,  where `n` is the number of samples, `p` and
            `q` are the number of dimensions of ``Z`` and ``Y`` respectively.

        Returns
        -------
        self : object
        """
        # Step 1: Residualize with regression

        # TODO: Check regression type for supporting single or multi-output regression
        if not self.residualized:
            regfit_xz = self.regression_estimator_xz.fit(X, Z)
            regfit_xy = self.regression_estimator_xy.fit(X, Y)

            # Compute residualized Zs and Ys.
            self.residualized_Z_ = Z - regfit_xz.predict(X)
            self.residualized_Y_ = Y - regfit_xy.predict(X)
        else:
            self.residualized_Z_ = np.copy(Z)
            self.residualized_Y_ = np.copy(Y)

            if X is not None:
                warnings.warn(
                    "You supplied `X` to the fit method but specified "
                    "`residualized=True` on init. This method will not use the "
                    "`X` argument that you provided."
                )

        # Step 2: Covariance estimation
        # Step 2a: Estimate covariance of Y,Z
        W = np.concatenate((self.residualized_Y_, self.residualized_Z_), axis=1)
        self.covfit_zy_ = self.covariance_estimator.fit(W)
        cols_Y = self.residualized_Y_.shape[1]
        self.cov_zy_ = self.covfit_zy_.covariance_[:cols_Y, cols_Y:]

        # Step 2b: Estimate precision of Z
        self.covfit_zz_ = self.precision_estimator.fit(self.residualized_Z_)
        self.prec_zz_ = self.covfit_zz_.precision_

        # Step 2c: Estimate precision of Y
        self.covfit_yy_ = self.precision_estimator.fit(self.residualized_Y_)
        self.prec_yy_ = self.covfit_yy_.precision_

        # Step 2d: Calculate residual cross-covariance
        self.residual_crosscovariance_ = np.diag(
            ((self.cov_zy_ @ self.prec_zz_) @ self.cov_zy_.T) @ self.prec_yy_
        ).flatten()

        n, k = Z.shape
        self.residual_crosscovariance_wherry_corrected_ = 1 - (
            1 - self.residual_crosscovariance_
        ) * ((n - 1) / (n - k - 1))

        if self.estimate_p_value:
            rcc_shuffle = []

            shuffle_cov_est = clone(self.covariance_estimator)
            shuffle_prec_est = clone(self.precision_estimator)

            if self.show_progress:
                shuffle_range = trange(self.n_shuffle)
            else:
                shuffle_range = range(self.n_shuffle)

            for n in shuffle_range:
                shuffle_Y = shuffle(Y)

                shuffle_W = np.concatenate((shuffle_Y, self.residualized_Z_), axis=1)
                shuffle_covfit_zy_ = shuffle_cov_est.fit(shuffle_W)
                shuffle_cov_zy_ = shuffle_covfit_zy_.covariance_[:cols_Y, cols_Y:]

                shuffle_prec_yy_ = shuffle_prec_est.fit(shuffle_Y).precision_

                rcc_shuffle.append(
                    np.diag(
                        ((shuffle_cov_zy_ @ self.prec_zz_) @ shuffle_cov_zy_.T)
                        @ shuffle_prec_yy_
                    ).flatten()
                )
            
            self.null_distribution_ = rcc_shuffle

            self.rcc_p_value_ = (
                np.sum(rcc_shuffle >= self.residual_crosscovariance_, axis=0)
                / self.n_shuffle
            )

        return self
