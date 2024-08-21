import numpy as np

import continuous_variables
import gates
import qubits
import circuits
import bell_state
from matplotlib import pyplot as plt
from matplotlib import cm
import datetime
import ipywidgets as widgets
from ipywidgets import interact
from multiprocessing import Manager,Process
import noises

if __name__ == '__main__':
    step = 0.1
    noise_intensity_list = np.arange(0, 1+step, step)
    z_keep_list = np.arange(0, 1+step, step)
    state_ab="00"
    qubit_c = np.array([[0.6], [0.8]])
    density_c = qubits.get_density_matrix(qubit_c)

    fidelity_matrix = []
    for noise_intensity in noise_intensity_list:
        fidelity_list = []
        for z_keep in z_keep_list:
            density_b, state_ca = bell_state.teleportation(density_c, state_ab, [noise_intensity, z_keep])
            density_b = bell_state.unitary_operation(density_b, 0, state_ca, state_ab)
            state_b = qubits.measurement(density_b, [0])[0]
            fidelity = qubits.get_fidelity(density_c,density_b)
            fidelity_list.append(fidelity)
        fidelity_matrix.append(fidelity_list)
    fidelity_matrix = np.array(fidelity_matrix)

    X, Y = np.meshgrid(z_keep_list,noise_intensity_list)
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, fidelity_matrix, cmap=cm.gist_rainbow)

    #plt.show()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, fidelity_matrix, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(X, Y, fidelity_matrix, zdir='z', offset=0.5, cmap='coolwarm')
    ax.contourf(X, Y, fidelity_matrix, zdir='x', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, fidelity_matrix, zdir='y', offset=0, cmap='coolwarm')
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0.5, 1),xlabel='z-axis proportion', ylabel='noise intensity', zlabel='fidelity')
    ax.set_title('fidelities')
    plt.show()