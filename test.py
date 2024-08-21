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
import noises
import math
import jupyter_function

if __name__ == '__main__':
    state_ab = "00"
    message = 1
    #qubit_c = np.array([[0.6],[0.8]])
    qubit_c = qubits.get_qubit_matrix([message])
    density_c = qubits.get_density_matrix(qubit_c)
    density_b, state_ca = bell_state.teleportation(density_c, state_ab,[0.5,1])
    density_b = bell_state.unitary_operation(density_b, 0, state_ca, state_ab)
    state_b = qubits.measurement(density_b, [0])[0]
    print(qubits.get_fidelity(density_c, density_b))
    print("qubit_c(sender):\n{}\nmeasurement_ca(send):{}\nqubit_b(receiver):\n{}".format(density_c, state_ca, density_b))