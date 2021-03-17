"""
Simulation Example.

===============================
Shows the different combinations of parameters used to
simulate null and true data.

"""

import numpy as np
from datasets import make_null_mediation, make_mediation

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_null_mediation(
    dag_type="null-dag1"
)
print(true_alpha)
print(np.round(np.transpose(true_beta), 1))
print(np.round(np.transpose(true_gam)), 1)

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_null_mediation(
    dag_type="null-dag2"
)
print(true_alpha)
print(np.round(np.transpose(true_beta), 1))
print(np.round(np.transpose(true_gam)), 1)

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_null_mediation(
    dag_type="null-dag3"
)
print(true_alpha)
print(np.round(np.transpose(true_beta), 1))
print(np.round(np.transpose(true_gam)), 1)

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_mediation()
print(true_alpha)
print(np.round(np.transpose(true_beta), 1))
print(np.round(np.transpose(true_gam)), 1)

outcomes, exposures, mediators, true_alpha, true_beta, true_gam = make_mediation(
    n_informative_mo=20
)
print(true_alpha)
print(np.round(np.transpose(true_beta), 1))
print(np.round(np.transpose(true_gam)), 1)
