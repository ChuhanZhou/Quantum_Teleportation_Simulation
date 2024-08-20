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

if __name__ == '__main__':
    batch_n = 10 ** 4
    step = 0.1
    noise_intensity_list = np.arange(0, 1+step, step)

    message = 1
    state_ab="00"
    qubit_c = qubits.get_qubit_matrix([message])
    #qubit_c = np.array([[0.6], [0.8]])
    density_c = qubits.get_density_matrix(qubit_c)

    for noise_intensity in noise_intensity_list:
        state_b_list = []
        fidelity_list = []
        for i in range(batch_n):
            density_b, state_ca = bell_state.teleportation(density_c, state_ab, [noise_intensity, 1])
            density_b = bell_state.unitary_operation(density_b, 0, state_ca, state_ab)
            state_b = qubits.measurement(density_b, [0])[0]
            fidelity_list.append(qubits.get_fidelity(density_c,density_b))
            state_b_list.append(int(state_b))
        accuracy = (batch_n-len(np.nonzero(np.array(state_b_list)-np.ones((len(state_b_list)))*message)[0]))/batch_n
        fidelity = np.average(fidelity_list)
        print(noise_intensity,accuracy,fidelity)