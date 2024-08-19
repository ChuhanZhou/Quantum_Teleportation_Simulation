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

# Pauli-Z 算符
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


# 创建 CNOT 门，作用于2个量子比特
def CNOT():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


# 创建 Hadamard 门，作用于单个量子比特
def H():
    return np.array([[1, 1],
                     [1, -1]], dtype=complex) / np.sqrt(2)


# 创建 Dephasing 噪声算符
def dephasing_noise(rho, p):
    # 对所有量子比特应用Dephasing噪声
    return (1 - p) * rho + p * np.dot(Z, np.dot(rho, Z))


# 密度矩阵形式的态转换
def ket_to_density(ket):
    return np.outer(ket, ket.conj())


# 保真度计算
def fidelity(rho, sigma):
    sqrt_rho = np.linalg.matrix_power(rho, 1 / 2)
    return np.real(np.trace(np.linalg.matrix_power(np.dot(sqrt_rho, np.dot(sigma, sqrt_rho)), 1 / 2)) ** 2)

if __name__ == '__main__':

    p = 1
    state_ab = "00"
    qubit_c = qubits.get_qubit_matrix([1])
    density_c = qubits.get_density_matrix(qubit_c)
    qubits_ab = qubits.get_qubit_matrix([s for s in state_ab])

    #density_cab = qubits.to_muti_qubit_matrix([density_c,density_ab])
    # create bell state
    density_bell_ab = bell_state.entangler(qubits.get_density_matrix(qubits_ab), 0, 1)
    density_bell_ab_noise = noises.dephasing_noise(noises.dephasing_noise(density_bell_ab,[0],0.0),[1],1)
    density_c_bell_ab = qubits.to_muti_qubit_matrix([density_c, density_bell_ab])

    density_c_bell_ab_noise =  qubits.to_muti_qubit_matrix([density_c, density_bell_ab_noise])

    density_bell_ca_b = bell_state.bell_measurement(density_c_bell_ab, 0, 1)
    density_bell_ca_b_noise = bell_state.bell_measurement(density_c_bell_ab, 0, 1)
    ideal_phi = qubits.get_density_matrix(qubit_c)
    measurement_ca, density_b = qubits.measurement(density_bell_ca_b,[0,1])
    density_b = bell_state.unitary_operation(density_b, 0, measurement_ca, state_ab)
    state_b = qubits.measurement(density_b, [0])[0]
    print(1,state_b)
    print(qubits.get_fidelity(density_b, density_c))

    # 分别施加不同强度的Dephasing噪声，并计算保真度
    for p in [0, 0.2, 0.5, 1.0]:
        rho_cab_noise = noises.dephasing_noise(noises.dephasing_noise(rho_ABC,1, p),2,p)
        a = qubits.slice_density_matrix(rho_cab_noise)
        rho_B = a[2]
        # 从密度矩阵中提取量子比特B的部分

        F = qubits.get_fidelity(rho_B, ideal_phi)
        print(f"噪声强度 p = {p} 时的保真度: {F:.4f}")