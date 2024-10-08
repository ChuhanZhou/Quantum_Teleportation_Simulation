import math

import circuits
import qubits


def dephasing_noise(density_matrix, gamma=0.0, z=0.0):
    '''
    gamma:  noise intensity
    z:  noise on z-axis of bloch sphere
    '''
    gamma = max(0, min(1, gamma))
    z = max(0, min(1, z))
    qubit_num = int(math.log2(density_matrix.shape[0]))
    rho = qubits.normalization(density_matrix)
    plan = []
    for i in range(qubit_num):
        plan.append(["I", [i]])
    I = circuits.Circuit(qubit_num, plan).circuit_matrix
    # All gates here are working as a matrix, a object, not calculated as a gate in circuit
    # [I] is a selector
    # ρ' = (1 - γ) * ρ + γ * [I] * (z*ρ+(1-z)*I)
    # ρ'z => zρ+(1-z)*I
    #   z = 0, ρ'z => I
    #   z = 1, ρ'z => 0
    # ρ'x => 0
    out = (1 - gamma) * rho + gamma * qubits.normalization(I * (z * rho + (1 - z) * I))
    out = qubits.normalization(out)
    return out
