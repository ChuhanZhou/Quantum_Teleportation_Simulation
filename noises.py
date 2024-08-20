import numpy as np
import gates
import math
import circuits
import qubits


def dephasing_noise(qubit_matrix, index_list=[], gamma=0.0):
    gamma = max(0,min(1,gamma))
    qubit_num = int(math.log2(qubit_matrix.shape[0]))
    rho = qubits.get_density_matrix(qubit_matrix)
    plan = []
    for i in index_list:
        plan.append(["Z",[i]])
    Z = circuits.Circuit(qubit_num, plan).circuit_matrix
    #gate[I]
    out_dm = ((1 + gamma) * rho + (1 - gamma) * Z @ rho @ Z)/2
    aa = (1 - gamma) * rho
    ab = gamma * np.dot(np.dot(Z, rho), Z)
    out = qubits.normalization(out_dm)
    return out


def density_matrix_to_amplitudes(rho):
    eigenvalues, eigenvectors = np.linalg.eig(rho)

    idx = np.argmax(eigenvalues)
    state_vector = eigenvectors[:, idx:idx+1]
    state_vector = qubits.normalization(state_vector)

    return state_vector