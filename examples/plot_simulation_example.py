"""
Simulation Example.

===============================
Shows the different combinations of parameters used to
simulate null and true data.

"""

import numpy as np
from skmediate.datasets import make_null_mediation, make_mediation

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_null_mediation(
    n_mediators=10, dag_type="null-dag1"
)
print("Parameter values when simulating null confounding motif:")
print("Alpha: {}".format(true_alpha))
print("Beta: {}".format(np.round(np.transpose(true_beta), 1)))
print("Gamma: {}".format(np.round(np.transpose(true_gam), 1)))
total_ie = np.multiply(np.transpose(true_beta), true_gam)
fraction_ie = np.round(np.divide(total_ie, total_ie + true_alpha), 2)
print("Indirect-to-Total Signal Strength: {}".format(fraction_ie))

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_null_mediation(
    n_mediators=10, dag_type="null-dag2"
)
print("Parameter values when simulating null independent variables motif:")
print("Alpha: {}".format(true_alpha))
print("Beta: {}".format(np.round(np.transpose(true_beta), 1)))
print("Gamma: {}".format(np.round(np.transpose(true_gam), 1)))
total_ie = np.multiply(np.transpose(true_beta), true_gam)
fraction_ie = np.round(np.divide(total_ie, total_ie + true_alpha), 2)
print("Indirect-to-Total Signal Strength: {}".format(fraction_ie))

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_null_mediation(
    n_mediators=10, dag_type="null-dag3"
)
print("Parameter values when simulating null noise motif:")
print("Alpha: {}".format(true_alpha))
print("Beta: {}".format(np.round(np.transpose(true_beta), 1)))
print("Gamma: {}".format(np.round(np.transpose(true_gam), 1)))
total_ie = np.multiply(np.transpose(true_beta), true_gam)
fraction_ie = np.round(np.divide(total_ie, total_ie + true_alpha), 2)
print("Indirect-to-Total Signal Strength: {}".format(fraction_ie))

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_mediation(
    n_mediators=10
)
print("Parameter values when simulating true mediation motif, no sparsity:")
print("Alpha: {}".format(true_alpha))
print("Beta: {}".format(np.round(np.transpose(true_beta), 1)))
print("Gamma: {}".format(np.round(np.transpose(true_gam), 1)))
total_ie = np.multiply(np.transpose(true_beta), true_gam)
fraction_ie = np.round(np.divide(total_ie, total_ie + true_alpha), 2)
print("Indirect-to-Total Signal Strength: {}".format(fraction_ie))

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_mediation(
    n_mediators=10, n_informative_mo=5
)
print("Parameter values when simulating true but partial mediation motif, 60% sparse :")
print("Alpha: {}".format(true_alpha))
print("Beta: {}".format(np.round(np.transpose(true_beta), 1)))
print("Gamma: {}".format(np.round(np.transpose(true_gam), 1)))
total_ie = np.multiply(np.transpose(true_beta), true_gam)
fraction_ie = np.round(np.divide(total_ie, total_ie + true_alpha), 2)
print("Indirect-to-Total Signal Strength: {}".format(fraction_ie))
