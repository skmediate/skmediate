import numpy as np

from sklearn.datasets import make_regression
from sklearn.utils import check_random_state


def make_null_mediation(n_samples=600, n_mediators=1, random_state=0):
    """
    Uses `make_regression` to simulate a naive common cause problem where X->M, and X->Y

    Parameters
    ----------
    n_samples : int, optional
        The number of samples in the problem. Default: 600

    n_mediators : int, optional
        The number of mediators in the problem. Default: 1

    random_state : int, optional
        Sets the random seed for the random number generator

    Returns
    -------
    outcomes :

    exposures :

    mediators :

    true_alpha :

    true_betas :

    true_gam :

    Examples
    --------
    >>> from skmediate.datasets import make_null_mediation
    >>> answer = make_null_mediation()
    >>> assert len(answer) == 6

    """
    generator = check_random_state(random_state)

    # exposure-mediator simulation X->M
    exposures, mediators, true_gam = make_regression(
        n_samples=n_samples,
        n_features=2,
        n_informative=1,
        n_targets=n_mediators,
        coef=True,
        shuffle=True,
        noise=1.0,
        random_state=generator,
    )

    if len(mediators.shape) == 1:
        mediators = mediators.reshape((mediators.shape[0], 1))

    # Y = alpha X + noise
    n_exposures = exposures.shape[1]
    n_mediators = mediators.shape[1]
    n_informative_eo = n_exposures  # informative exposure-outcome coefficients
    n_outcomes = 1

    true_alpha = np.zeros((n_exposures, n_outcomes))
    true_betas = np.zeros((n_mediators, n_outcomes))
    y_noise = 1.0

    true_alpha[:n_informative_eo, :] = 25 + 75 * generator.rand(
        n_informative_eo, n_outcomes
    )

    outcomes = np.dot(exposures, true_alpha) + generator.normal(
        scale=y_noise, size=(n_samples, n_outcomes)
    )

    return outcomes, exposures, mediators, true_alpha, true_betas, true_gam


def make_mediation(n_samples=600, n_mediators=1, y_noise=1.5):
    """
    Uses `make_regression` to simulate a naive multiple independent mediators
    problem.


    """

    random_state = 2
    generator = check_random_state(random_state)

    # exposure-mediator simulation
    exposures, mediators, true_gam = make_regression(
        n_samples=n_samples,
        n_features=2,
        n_informative=1,
        n_targets=n_mediators,
        coef=True,
        shuffle=True,
        noise=1.0,
        random_state=generator,
    )
    if len(mediators.shape) == 1:
        mediators = mediators.reshape((mediators.shape[0], 1))

    print(exposures.shape)
    print(mediators.shape)

    # Y = alpha X + beta_1 M_1 + beta_2 M_2 + noise

    n_exposures = exposures.shape[1]
    n_mediators = mediators.shape[1]
    n_informative_eo = n_exposures  # informative exposure-outcome coefficients
    n_outcomes = 1

    true_alpha = np.zeros((n_exposures, n_outcomes))
    true_betas = np.zeros((n_mediators, n_outcomes))

    true_alpha[:n_informative_eo, :] = 25 + 75 * generator.rand(
        n_informative_eo, n_outcomes
    )

    true_betas = 25 + 75 * generator.rand(n_mediators, n_outcomes)

    outcomes = (
        np.dot(exposures, true_alpha)
        + np.dot(mediators, true_betas)
        + generator.normal(scale=y_noise, size=(n_samples, n_outcomes))
    )

    return outcomes, exposures, mediators, true_alpha, true_betas, true_gam
