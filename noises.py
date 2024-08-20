import numpy as np
import gates
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
    # ρ' = (1 - γ) * ρ + γ * I * (z*ρ+(1-z)*I)
    # ρ'z => zρ+(1-z)*I
    #   z = 0, ρ'z => I
    #   z = 1, ρ'z => 0
    # ρ'x => 0
    out = (1 - gamma) * rho + gamma * qubits.normalization(I * (z*rho+(1-z)*I))
    out = qubits.normalization(out)
    return out

def dephasing_noise_keep_z(density_matrix, gamma=0.0):
    gamma = max(0, min(1, gamma))
    qubit_num = int(math.log2(density_matrix.shape[0]))
    rho = density_matrix
    plan = []
    for i in range(qubit_num):
        plan.append(["I", [i]])
    I = circuits.Circuit(qubit_num, plan).circuit_matrix

    # ρ' = (1 - γ) * ρ + γ * I * ρ
    # ρ'z => ρ
    # ρ'x => 0
    out = (1 - gamma) * rho + gamma * I * rho
    out = qubits.normalization(out)
    return out

def dephasing_noise_0_z(density_matrix, gamma=0.0):
    gamma = max(0, min(1, gamma))
    qubit_num = int(math.log2(density_matrix.shape[0]))
    rho = density_matrix
    plan = []
    for i in range(qubit_num):
        plan.append(["I", [i]])
    I = circuits.Circuit(qubit_num, plan).circuit_matrix

    #ρ' = (1 - γ) * ρ + γ * I
    #ρ' => I
    out = (1-gamma) * rho +gamma * I
    out = qubits.normalization(out)
    return out