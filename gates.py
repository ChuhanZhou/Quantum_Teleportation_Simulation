import math

import numpy as np
import qubits

gate_list = [
    ["I","H","X","Y","Z","S","T"],
    ["CNOT","CZ"],
    ["CCNOT"]]

def get_gate_by_name(name="", is_inverse=False,inner_gates=[]):
    if name == "I" or name == "identity":
        return get_identity_gate()
    elif name == "H" or name == "hadamard":
        return get_hadamard_gate()
    elif name == "X" or name == "pauli_x":
        return get_pauli_x_gate()
    elif name == "Y" or name == "pauli_y":
        return get_pauli_y_gate()
    elif name == "Z" or name == "pauli_z":
        return get_pauli_z_gate()
    elif name == "S" or name == "P" or name == "phase":
        return get_phase_gate()
    elif name == "T":
        return get_t_gate()
    elif name == "CNOT" or name == "controlled_not":
        return get_cnot_gate(is_inverse, inner_gates)
    elif name == "CZ" or name == "controlled_z":
        return get_cz_gate(is_inverse,inner_gates)
    elif name == "CCNOT" or name == "toffoli":
        return get_ccnot_gate(is_inverse, inner_gates)
    return np.array([[]])


def get_identity_gate():
    print_str = ["┏━━━━━┓",
                 "┃  I  ┃",
                 "┗━━━━━┛"]
    identity_gate = np.array([
        [1, 0],
        [0, 1],
    ])
    return [identity_gate,print_str]


def get_hadamard_gate():
    print_str = ["┏━━━━━┓",
                 "┃  H  ┃",
                 "┗━━━━━┛"]
    hadamard_gate = np.array([
        [1, 1],
        [1, -1],
    ]) * pow(2, 1 / 2)
    return [hadamard_gate,print_str]


def get_pauli_x_gate():
    print_str = ["┏━━━━━┓",
                 "┃  X  ┃",
                 "┗━━━━━┛"]
    pauli_x_gate = np.array([
        [0, 1],
        [1, 0],
    ])
    return [pauli_x_gate,print_str]


def get_pauli_y_gate():
    print_str = ["┏━━━━━┓",
                 "┃  Y  ┃",
                 "┗━━━━━┛"]
    pauli_y_gate = np.array([
        [0, -1j],
        [1j, 0],
    ])
    return [pauli_y_gate,print_str]


def get_pauli_z_gate():
    print_str = ["┏━━━━━┓",
                 "┃  Z  ┃",
                 "┗━━━━━┛"]
    pauli_z_gate = np.array([
        [1, 0],
        [0, -1],
    ])
    return [pauli_z_gate,print_str]


def get_phase_gate():
    print_str = ["┏━━━━━┓",
                 "┃  P  ┃",
                 "┗━━━━━┛"]
    phase_gate = np.array([
        [1, 0],
        [0, 1j],
    ])
    return [phase_gate,print_str]


def get_t_gate():
    print_str = ["┏━━━━━┓",
                 "┃  T  ┃",
                 "┗━━━━━┛"]
    t_gate = np.array([
        [1, 0],
        [0, pow(math.e, 1j * math.pi / 4)],
    ])
    return [t_gate,print_str]

#inner_gates:[inner_gates_0:gates_matrix,inner_gates_1:gates_matrix]
def get_cnot_gate(is_inverse=False,inner_gates=[]):
    # cnot_gate = np.array([
    #    [1, 0, 0, 0],
    #    [0, 1, 0, 0],
    #    [0, 0, 0, 1],
    #    [0, 0, 1, 0],
    # ])
    print_str = []
    if is_inverse:
        print_str += ["       ",
                      "---⊕---",
                      "   ┃   "]
    else:
        print_str += ["       ",
                      "---●---",
                      "   ┃   "]
    qubit_0 = qubits.get_basic_qubit_0()
    qubit_1 = qubits.get_basic_qubit_1()
    gate_matrix_list = []
    inner_gates_num = 1

    for i in range(inner_gates_num):
        if len(inner_gates)>i:
            gate_matrix_list.append(inner_gates[i][0])
            print_str += inner_gates[i][1]
        else:
            gate_matrix_list.append(1)

    if is_inverse:
        print_str += ["   ┃   ",
                      "---●---",
                      "       "]
    else:
        print_str += ["   ┃   ",
                      "---⊕---",
                      "       "]

    if not is_inverse:
        cnot_gate = np.kron(np.kron(qubit_0, qubit_0.T), np.kron(gate_matrix_list[0],get_identity_gate()[0])) + \
                    np.kron(np.kron(qubit_1, qubit_1.T), np.kron(gate_matrix_list[0],get_pauli_x_gate()[0]))
    else:
        cnot_gate = np.kron(np.kron(get_identity_gate()[0],gate_matrix_list[0]), np.kron(qubit_0, qubit_0.T)) + \
                    np.kron(np.kron(get_pauli_x_gate()[0],gate_matrix_list[0]), np.kron(qubit_1, qubit_1.T))
    return [cnot_gate,print_str]


def get_cz_gate(is_inverse=False,inner_gates=[]):
    # cz_gate = np.array([
    #    [1, 0, 0, 0],
    #    [0, 1, 0, 0],
    #    [0, 0, 1, 0],
    #    [0, 0, 0,-1],
    # ])

    qubit_0 = qubits.get_basic_qubit_0()
    qubit_1 = qubits.get_basic_qubit_1()
    gate_matrix_list = []
    inner_gates_num = 1

    for i in range(inner_gates_num):
        if len(inner_gates) > i:
            gate_matrix_list.append(inner_gates[i])
        else:
            gate_matrix_list.append(1)

    if not is_inverse:
        cz_gate = np.kron(np.kron(qubit_0, qubit_0.T), np.kron(gate_matrix_list[0], get_identity_gate()[0])) + \
                  np.kron(np.kron(qubit_1, qubit_1.T), np.kron(gate_matrix_list[0], get_pauli_x_gate()[0]))
    else:
        cz_gate = np.kron(np.kron(get_identity_gate()[0], gate_matrix_list[0]), np.kron(qubit_0, qubit_0.T)) + \
                  np.kron(np.kron(get_pauli_x_gate()[0], gate_matrix_list[0]), np.kron(qubit_1, qubit_1.T))
    return [cz_gate,""]


def get_ccnot_gate(is_inverse=False,inner_gates=[]):
    # ccnot_gate = np.array([
    #   [1, 0, 0, 0, 0, 0, 0, 0],
    #   [0, 1, 0, 0, 0, 0, 0, 0],
    #   [0, 0, 1, 0, 0, 0, 0, 0],
    #   [0, 0, 0, 1, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 1, 0, 0, 0],
    #   [0, 0, 0, 0, 0, 1, 0, 0],
    #   [0, 0, 0, 0, 0, 0, 0, 1],
    #   [0, 0, 0, 0, 0, 0, 1, 0],
    # ])
    qubit_0 = qubits.get_basic_qubit_0()
    qubit_1 = qubits.get_basic_qubit_1()
    gate_matrix_list = []
    inner_gates_num = 2

    for i in range(inner_gates_num):
        if len(inner_gates) > i:
            gate_matrix_list.append(inner_gates[i])
        else:
            gate_matrix_list.append(1)

    if not is_inverse:
        ccnot_gate = np.kron(np.kron(np.kron(np.kron(qubit_0, qubit_0.T),gate_matrix_list[0]) , np.kron(np.kron(qubit_0, qubit_0.T),gate_matrix_list[1])), get_identity_gate()[0]) + \
                     np.kron(np.kron(np.kron(np.kron(qubit_0, qubit_0.T),gate_matrix_list[0]) , np.kron(np.kron(qubit_1, qubit_1.T),gate_matrix_list[1])), get_identity_gate()[0]) + \
                     np.kron(np.kron(np.kron(np.kron(qubit_1, qubit_1.T),gate_matrix_list[0]) , np.kron(np.kron(qubit_0, qubit_0.T),gate_matrix_list[1])), get_identity_gate()[0]) + \
                     np.kron(np.kron(np.kron(np.kron(qubit_1, qubit_1.T),gate_matrix_list[0]) , np.kron(np.kron(qubit_1, qubit_1.T),gate_matrix_list[1])), get_pauli_x_gate()[0])
    else:
        ccnot_gate = np.kron(np.kron(np.kron(get_identity_gate()[0], gate_matrix_list[0]),np.kron(np.kron(qubit_0, qubit_0.T), gate_matrix_list[1])), np.kron(qubit_0, qubit_0.T)) + \
                     np.kron(np.kron(np.kron(get_identity_gate()[0], gate_matrix_list[0]),np.kron(np.kron(qubit_1, qubit_1.T), gate_matrix_list[1])), np.kron(qubit_0, qubit_0.T)) + \
                     np.kron(np.kron(np.kron(get_identity_gate()[0], gate_matrix_list[0]),np.kron(np.kron(qubit_0, qubit_0.T), gate_matrix_list[1])), np.kron(qubit_1, qubit_1.T)) + \
                     np.kron(np.kron(np.kron(get_pauli_x_gate()[0], gate_matrix_list[0]),np.kron(np.kron(qubit_1, qubit_1.T), gate_matrix_list[1])), np.kron(qubit_1, qubit_1.T))
    return [ccnot_gate,""]
