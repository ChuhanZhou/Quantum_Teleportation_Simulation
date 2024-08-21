import math
import random
import numpy as np

import gates
import qubits
from scipy.linalg import sqrtm


def get_basic(type=0):
    if type==0:
        return get_basic_qubit_0()
    return get_basic_qubit_1()

def get_qubit_matrix(input_list=[]):
    """
    input_list : [q0,q1,q2,...,qn]
        ex : [0,1,1,0,1]
    """
    qubit_matrix = 1
    for q_type in input_list:
        qubit = get_basic(int(q_type))
        qubit_matrix = np.kron(qubit_matrix, qubit)
    return qubit_matrix

def to_muti_qubit_matrix(qubit_list=[]):
    muti_qubit_matrix = 1
    for q_matrix in qubit_list:
        muti_qubit_matrix = np.kron(muti_qubit_matrix, q_matrix)
    return muti_qubit_matrix

def get_basic_qubit_0():# ∣0⟩
    return np.array([[1.], [0.]])

def get_basic_qubit_1():# ∣1⟩
    return np.array([[0.], [1.]])

def get_fidelity(qubit_0,qubit_1):
    if qubit_0.shape[0] == qubit_0.shape[1] and qubit_0.shape[0] > 1:
        return get_density_matrix_fidelity(qubit_0,qubit_1)
    else:
        return get_state_vector_fidelity(qubit_0,qubit_1)

def get_state_vector_fidelity(qubit_0,qubit_1):
    f = abs(qubit_0.T @ qubit_1) ** 2
    return f[0, 0]

def get_density_matrix_fidelity(qubit_0,qubit_1):
    f = np.trace(sqrtm(sqrtm(qubit_0) @ qubit_1 @ sqrtm(qubit_0))) ** 2
    f = float(f.real)
    return f

def normalization(qubit_matrix):
    """
    normalization=> <ψ|ψ>=1
    """
    if qubit_matrix.shape[0] == qubit_matrix.shape[1] and qubit_matrix.shape[0] > 1:
        return normalize_density_matrix(qubit_matrix)
    else:
        return normalize_state_vector(qubit_matrix)

def normalize_state_vector(qubit_matrix):
    s = (np.abs(qubit_matrix) ** 2).sum()
    qubit_matrix_norm = qubit_matrix / np.sqrt(s)
    return qubit_matrix_norm

def normalize_density_matrix(qubit_matrix):
    qubit_matrix_norm = qubit_matrix / np.trace(qubit_matrix)
    return qubit_matrix_norm

def measurement(qubit_matrix,measurement_list=[]):
    '''
    measurement_list:[measurement_i_0,measurement_i_1,...]
    '''
    if qubit_matrix.shape[0] == qubit_matrix.shape[1] and qubit_matrix.shape[0] > 1:
        return measurement_density_matrix(qubit_matrix,measurement_list)
    else:
        return measurement_state_vector(qubit_matrix,measurement_list)

def measurement_state_vector(qubit_matrix,measurement_list=[]):
    qubit_num = int(math.log2(qubit_matrix.shape[0]))
    matrix_part_vector = {}
    options_probabilities = {}
    for o_i in range(qubit_matrix.shape[0]):
        decode_all = bin(o_i).split("0b")[-1].zfill(qubit_num)  # from 2,3 to 010,011
        decode_measure = ""
        decode_other = ""
        for i in range(0, qubit_num):
            if i not in measurement_list:
                decode_other += decode_all[i]
            else:
                decode_measure += decode_all[i]
        part_vector = qubit_matrix[o_i, 0]
        if decode_measure in options_probabilities.keys():
            matrix_part_vector.get(decode_measure).update({decode_other: part_vector})
            options_probabilities.update({decode_measure: options_probabilities.get(decode_measure) + abs(part_vector)**2})
        else:
            matrix_part_vector.update({decode_measure: {decode_other: part_vector}})
            options_probabilities.update({decode_measure:abs(part_vector)**2})
    options = list(options_probabilities.keys())
    probabilities = list(options_probabilities.values())
    result = random.choices(options, probabilities)[0]

    # create state vector for qubits which is not measured
    other_prob = matrix_part_vector.get(result)
    other_qubit_num = qubit_num - len(measurement_list)
    other_matrix = np.zeros((2 ** other_qubit_num, 1))
    for i in range(other_matrix.shape[0]):
        decode = bin(i).split("0b")[-1].zfill(other_qubit_num)
        other_matrix[i,0] = other_prob.get(decode)
    other_matrix = qubits.normalization(other_matrix)
    return result, other_matrix

def measurement_density_matrix(density_matrix,measurement_list=[]):
    qubit_num = int(math.log2(density_matrix.shape[0]))
    matrix_part_prob = {}
    options_probabilities = {}
    for o_i in range(density_matrix.shape[0]):
        decode_all = bin(o_i).split("0b")[-1].zfill(qubit_num)  # from 2,3 to 010,011
        decode_measure = ""
        decode_other = ""
        for i in range(0, qubit_num):
            if i not in measurement_list:
                decode_other += decode_all[i]
            else:
                decode_measure += decode_all[i]
        state_target = get_qubit_matrix([s for s in decode_all])

        part_prob = np.sum(state_target.T*density_matrix*state_target)#P(state)=<state|*ρ*|state>
        if decode_measure in options_probabilities.keys():
            matrix_part_prob.get(decode_measure).update({decode_all: part_prob})
            options_probabilities.update(
                {decode_measure: options_probabilities.get(decode_measure) + part_prob})
        else:
            matrix_part_prob.update({decode_measure: {decode_all: part_prob}})
            options_probabilities.update({decode_measure: part_prob})
    options = list(options_probabilities.keys())
    probabilities = list(options_probabilities.values())
    result = random.choices(options, probabilities)[0]

    # get density matrix for qubits which is not measured
    gate_i = gates.get_gate_by_name("I")
    qubit_m = 1
    m_i=0
    for i in range(0, qubit_num):
        if i not in measurement_list:
            qubit_m = np.kron(qubit_m,gate_i)
        else:
            qubit_m = np.kron(qubit_m, qubits.get_basic(int(result[m_i])))
            m_i+=1
    other_matrix = qubit_m.T @ density_matrix @ qubit_m
    other_matrix = qubits.normalization(other_matrix)
    return result, other_matrix

# rho:ρ
def get_density_matrix(qubit_matrix):
    rho = np.dot(qubit_matrix, qubit_matrix.T)
    return rho

def slice_density_matrix(density_matrix,keep=[]):
    qubit_num = int(math.log2(density_matrix.shape[0]))
    density_matrix = density_matrix.reshape((np.ones((qubit_num*2)) * 2).astype("int").tolist())

    part_density_matrix = density_matrix
    qubit_n = qubit_num
    for p_i in sorted(set(range(qubit_num)) - set(keep), reverse=True):
        part_density_matrix = np.trace(part_density_matrix, axis1=p_i, axis2=p_i + qubit_n)
    part_density_matrix = part_density_matrix.reshape((2**len(keep),2**len(keep)))
    part_density_matrix = normalization(part_density_matrix)
    return part_density_matrix

def decode_qubits(qubits = np.array([])):
    qubit_num = int(math.log2(qubits.shape[0]))
    qubit_list = []
    one_i_list = np.nonzero(qubits)[0]
    for i in one_i_list:
        product_group = []
        for qubit_i in range(qubit_num):
            q = np.zeros((2, 1))

            qubit_matrix_i = int(i%2**(qubit_num-qubit_i)>=2**(qubit_num-qubit_i)/2)
            q[qubit_matrix_i,0]=1
            product_group.append(q)
        qubit_list.append(product_group)
    return qubit_list