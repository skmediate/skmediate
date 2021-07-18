"""Tests for functions in conditonal independence """

import numpy as np
import pytest
import warnings

from sklearn.linear_model import Ridge, LinearRegression
from skmediate.conditional_independence import ConditionalCrossCovariance


def test_conditional_independence():
    X = np.random.randn(10, 1)
    Z = np.random.randn(10, 5)
    Y = np.random.randn(10, 1)

    xcov_estimator = ConditionalCrossCovariance()
    xcov_estimator.fit(Z, Y, X)

    xcov_estimator = ConditionalCrossCovariance(
        regression_estimator=[Ridge(), LinearRegression()]
    )
    xcov_estimator.fit(Z, Y, X)

    xcov_estimator = ConditionalCrossCovariance(Ridge(), residualized=True)
    xcov_estimator.fit(Z, Y)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xcov_estimator.fit(Z, Y, X)
        # Verify warning
        assert len(w) == 1
        assert "residualized" in str(w[-1].message)

    xcov_estimator = ConditionalCrossCovariance(
        covariance_estimator="ledoit_wolf",
        precision_estimator="min_cov_det",
        estimate_p_value=True,
        show_progress=False,
    )
    xcov_estimator.fit(Z, Y, X)

    xcov_estimator = ConditionalCrossCovariance(
        covariance_estimator="ledoit_wolf",
        precision_estimator="min_cov_det",
        estimate_p_value=True,
        show_progress=True,
    )
    xcov_estimator.fit(Z, Y, X)

    with pytest.raises(ValueError):
        ConditionalCrossCovariance(regression_estimator=[22, "error"])

    with pytest.raises(ValueError):
        ConditionalCrossCovariance(regression_estimator=22)

    with pytest.raises(ValueError):
        ConditionalCrossCovariance(covariance_estimator="error")

    with pytest.raises(ValueError):
        ConditionalCrossCovariance(precision_estimator="error")
