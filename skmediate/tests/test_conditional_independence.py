"""Tests for functions in conditonal independence """

import numpy as np
from skmediate.conditional_independence import ConditionalCrossCovariance


def test_conditional_independence():

    X = np.random.randn(10, 1)
    Z = np.random.randn(10, 5)
    Y = np.random.randn(10, 1)

    xcov_estimator = ConditionalCrossCovariance()
    xcov_estimator.fit(X, Z, Y)
