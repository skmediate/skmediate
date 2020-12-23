# import numpy as np
# import pytest

from skmediate.datasets import make_null_mediation, make_mediation

# from sklearn.utils._testing import assert_array_almost_equal
# from sklearn.utils._testing import assert_raises


def test_make_null_mediation():
    # Smoke test
    answer = make_null_mediation()
    assert len(answer) == 6


def test_make_mediation():
    answer = make_mediation()
    assert len(answer) == 6
