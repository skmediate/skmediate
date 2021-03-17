"""Generate simulated and pseudoreal data."""

import numpy as np

from sklearn.datasets import make_regression
from sklearn.utils import check_random_state


def make_null_mediation(
    n_samples=600, n_mediators=50, random_state=0, dag_type="null-dag1", y_noise=1.0
):
    """
    Simulate null-dags for the mediation CPDAG.

    Uses make_regression from sklearn to simulate X-Z relationship.
    Currently n_exposures and n_outcomes are fixed to be univariate.
    - 'null-dag1': confounding motif or common cause problem where X->Z,
    and X->Y
    - 'null-dag2': independent motif where  X->Y and Z->Y
    - 'null-dag3': noise motif where X->Y and Z is totally independent

    Parameters
    ----------
    n_samples : int, optional
        The number of samples in the problem. Default: 600

    n_mediators : int, optional
        The number of mediators in the problem. Default: 1

    random_state : int, optional
        Sets the random seed for the random number generator

    dag_type : string, optional
        Choose the type of null model to simulate

    y_noise : float, optional
        Set the random noise level added to outcomes. Default is 1.0

    Returns
    -------
    outcomes : ndarray of shape (n_samples, n_outcomes)

    exposures : ndarray of shape (n_samples, n_exposures)

    mediators : ndarray of shape (n_samples, n_mediators)

    true_alpha :
        Coefficient(s) of X in the outcome model

    true_beta :
        Coefficient(s) of M in the outcome model

    true_gam :
        Coefficient(s) of X in the mediator model

    Examples
    --------
    >>> from skmediate.datasets import make_null_mediation
    >>> answer = make_null_mediation()
    >>> assert len(answer) == 6

    """
    generator = check_random_state(random_state)

    # exposure-mediator simulation X->Z, single exposure

    if dag_type == "null-dag1":

        exposures, mediators, true_gam = make_regression(
            n_samples=n_samples,
            n_features=1,
            n_informative=1,
            n_targets=n_mediators,
            coef=True,
            shuffle=True,
            noise=0.25,
            random_state=generator,
        )

    else:

        exposures, mediators, true_gam = make_regression(
            n_samples=n_samples,
            n_features=1,
            n_informative=0,
            n_targets=n_mediators,
            coef=True,
            shuffle=True,
            noise=0.25,
            random_state=generator,
        )

    if len(mediators.shape) == 1:
        mediators = mediators.reshape((mediators.shape[0], 1))

    n_exposures = exposures.shape[1]
    n_mediators = mediators.shape[1]
    n_informative_eo = n_exposures  # informative exposure-outcome coefficients
    n_outcomes = 1

    true_alpha = np.zeros((n_exposures, n_outcomes))
    true_beta = np.zeros((n_mediators, n_outcomes))

    if dag_type == "null-dag1":

        true_alpha[:n_informative_eo, :] = 75 + 25 * generator.rand(
            n_informative_eo, n_outcomes
        )

        # Y = alpha X + noise
        outcomes = np.dot(exposures, true_alpha) + generator.normal(
            scale=y_noise, size=(n_samples, n_outcomes)
        )

    elif dag_type == "null-dag2":

        true_alpha[:n_informative_eo, :] = 75 + 25 * generator.rand(
            n_informative_eo, n_outcomes
        )

        true_beta = 25 + 75 * generator.rand(n_mediators, n_outcomes)

        # Y = alpha X + beta M + noise
        outcomes = (
            np.dot(exposures, true_alpha)
            + np.dot(mediators, true_beta)
            + generator.normal(scale=y_noise, size=(n_samples, n_outcomes))
        )
    else:

        true_alpha[:n_informative_eo, :] = 75 + 25 * generator.rand(
            n_informative_eo, n_outcomes
        )

        # Y = alpha X + noise
        outcomes = np.dot(exposures, true_alpha) + generator.normal(
            scale=y_noise, size=(n_samples, n_outcomes)
        )

    return outcomes, exposures, mediators, true_alpha, true_beta, true_gam


def make_mediation(n_samples=600, n_mediators=50, n_informative_mo=None, y_noise=1.0):
    """
    Simulate data according to a mediation motif.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples in the problem. Default: 600

    n_mediators : int, optional
        The number of mediators in the problem. Default: 50

    n_informative_mo : int, optional
        The number of true mediational paths
        out of the maximum number, n_mediators. Default: n_mediators

    y_noise : float,
        The noise levels of y. Default: 1.0

    random_state : int, optional
        Sets the random seed for the random number generator

    Returns
    -------
    outcomes : ndarray of shape (n_samples, n_outcomes)

    exposures : ndarray of shape (n_samples, n_exposures)

    mediators : ndarray of shape (n_samples, n_mediators)

    true_alpha :
        Coefficient(s) of X in the outcome model

    true_beta :
        Coefficient(s) of M in the outcome model

    true_gam :
        Coefficient(s) of X in the mediator model

    Examples
    --------
    >>> from skmediate.datasets import make_mediation
    >>> y, x, z, true_alpha, true_beta, true_gam = make_mediation()
    >>> assert y.shape == (600, 1)
    >>> assert x.shape == (600, 1)
    """
    random_state = 2
    generator = check_random_state(random_state)

    # exposure-mediator simulation
    exposures, mediators, true_gam = make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        n_targets=n_mediators,
        coef=True,
        shuffle=True,
        noise=0.25,
        random_state=generator,
    )

    if len(mediators.shape) == 1:
        mediators = mediators.reshape((mediators.shape[0], 1))

    n_exposures = exposures.shape[1]
    n_mediators = mediators.shape[1]
    n_informative_eo = n_exposures  # informative exposure-outcome coefficients
    if n_informative_mo is None:
        n_informative_mo = n_mediators
        # n_informative_mo = np.rint(np.floor(n_mediators/2))
    n_outcomes = 1

    true_alpha = np.zeros((n_exposures, n_outcomes))
    true_beta = np.zeros((n_mediators, n_outcomes))

    true_alpha[:n_informative_eo, :] = 75 + 25 * generator.rand(
        n_informative_eo, n_outcomes
    )

    true_beta = 75 + 10 * generator.rand(n_mediators, n_outcomes)
    true_beta[int(n_informative_mo) : n_mediators, :] = 0

    outcomes = (
        np.dot(exposures, true_alpha)
        + np.dot(mediators, true_beta)
        + generator.normal(scale=y_noise, size=(n_samples, n_outcomes))
    )

    return outcomes, exposures, mediators, true_alpha, true_beta, true_gam
