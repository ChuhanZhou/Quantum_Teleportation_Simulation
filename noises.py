import numpy as np
import gates
import math
import circuits
import qubits


def dephasing_noise(qubit_matrix, i=0, gamma=0.0):
    gamma = max(0,min(1,gamma))
    qubit_num = int(math.log2(qubit_matrix.shape[0]))
    rho = qubits.get_density_matrix(qubit_matrix)
    Z = circuits.Circuit(qubit_num, [["Z", [i]]]).circuit_matrix

    out_dm = (1 - gamma) * rho + gamma * np.dot(Z, np.dot(rho, Z))
    return out_dm


def density_matrix_to_amplitudes(rho):
    eigenvalues, eigenvectors = np.linalg.eig(rho)

    idx = np.argmax(eigenvalues)
    state_vector = eigenvectors[:, idx:idx+1]
    state_vector = qubits.normalization(state_vector)

    return state_vector