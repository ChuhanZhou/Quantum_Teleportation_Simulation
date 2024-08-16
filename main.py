import numpy as np

import continuous_variables
import gates
import qubits
import circuits
import bell_state
from matplotlib import pyplot as plt
import datetime
import ipywidgets as widgets
from ipywidgets import interact
from multiprocessing import Manager,Process


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    N=10
    fock_0 = continuous_variables.get_fock_basic(0,N)
    fock_2 = continuous_variables.get_fock_basic(2,N)
    creation_operator = continuous_variables.get_creation_operator(N)
    annihilation_operator = continuous_variables.get_annihilation_operator(N)
    fock_1 = np.dot(creation_operator,fock_0)
    fock_3 = np.dot(creation_operator,fock_2)
    coherent_0_0 = continuous_variables.get_coherent_state(0.0, N)
    coherent_1_0 = continuous_variables.get_coherent_state(1.0, N)
    coherent_1_5 = continuous_variables.get_coherent_state(1.5, N)
    coherent_2_0 = continuous_variables.get_coherent_state(2.0,N)
    epr_mode_1, epr_mode_2 = continuous_variables.get_epr_state([1.0,2.0],1,N)

    plt.bar(range(N), np.abs(epr_mode_1[:,0]) ** 2, alpha=0.6, label='EPR Mode 1')
    plt.bar(range(N), np.abs(epr_mode_2[:,0]) ** 2, alpha=0.6, label='EPR Mode 2')
    plt.title("Initial")
    plt.xlabel("x")
    plt.ylabel("|psi(x)|^2")
    plt.show()
    a=1


