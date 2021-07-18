# import numpy as np
import pytest

from skmediate.datasets import make_null_mediation, make_mediation

# from sklearn.utils._testing import assert_array_almost_equal
# from sklearn.utils._testing import assert_raises


@pytest.mark.parametrize("n_mediators", [1, 2, 3])
@pytest.mark.parametrize("dag_type", ["null-dag1", "null-dag2", "null-dag3"])
def test_make_null_mediation(n_mediators, dag_type):
    # Smoke test
    answer = make_null_mediation(n_mediators=n_mediators, dag_type=dag_type)
    assert len(answer) == 6


@pytest.mark.parametrize("n_mediators", [1, 2, 3])
def test_make_mediation(n_mediators):
    answer = make_mediation(n_mediators=n_mediators)
    assert len(answer) == 6

    answer = make_mediation(n_mediators=n_mediators, n_informative_mo=1)
    assert len(answer) == 6
