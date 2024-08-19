import math
import random
import numpy as np

import qubits


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
    return np.array([[1], [0]])

def get_basic_qubit_1():# ∣1⟩
    return np.array([[0], [1]])

def get_fidelity (qubit_0,qubit_1):
    a = np.dot(qubit_0.T, qubit_1)
    f = abs(np.dot(qubit_0.T, qubit_1)) ** 2
    return f[0, 0]

def normalization(qubit_matrix):
    """
    normalization=> <ψ|ψ>=1
    """
    #s = (np.abs(qubit_matrix) ** 2).sum()
    #qubit_matrix_norm = qubit_matrix / np.sqrt(s)

    qubit_matrix_norm = qubit_matrix/np.trace(qubit_matrix)
    return qubit_matrix_norm

def measurement(qubit_matrix,measurement_list=[]):
    '''
    measurement_list:[measurement_i_0,measurement_i_1,...]
    '''
    if qubit_matrix.shape[0] == qubit_matrix.shape[1]:
        return measurement_density_matrix(qubit_matrix,measurement_list)
    else:
        return measurement_state_vector(qubit_matrix,measurement_list)

def measurement_state_vector(qubit_matrix,measurement_list=[]):
    qubit_num = int(math.log2(qubit_matrix.shape[0]))
    matrix_part_prob = {}
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
        part_prob = qubit_matrix[o_i, 0]
        if decode_measure in options_probabilities.keys():
            matrix_part_prob.get(decode_measure).update({decode_other: part_prob})
            options_probabilities.update(
                {decode_measure: options_probabilities.get(decode_measure) + abs(part_prob) ** 2})
        else:
            matrix_part_prob.update({decode_measure: {decode_other: part_prob}})
            options_probabilities.update({decode_measure: abs(part_prob)})
    options = list(options_probabilities.keys())
    probabilities = list(options_probabilities.values())
    result = random.choices(options, probabilities)[0]

    # create matrix for qubits which is not measured
    other_prob = matrix_part_prob.get(result)
    other_qubit_num = qubit_num - len(measurement_list)
    other_matrix = np.zeros((2 ** other_qubit_num, 1))
    for i in range(other_matrix.shape[0]):
        decode = bin(i).split("0b")[-1].zfill(other_qubit_num)
        other_matrix[i, 0] = other_prob.get(decode)
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
        part_prob = np.sum(state_target.T*density_matrix*state_target)
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

    # create matrix for qubits which is not measured
    other_prob = matrix_part_prob.get(result)

    other_matrix = 0
    for state_i in other_prob.keys():
        state_qubit = get_qubit_matrix([s for s in state_i])
        state_prob = other_prob.get(state_i)
        other_matrix += state_prob*np.dot(np.dot(state_qubit,state_qubit.T),density_matrix)

    keep_list = []
    for i in range(0, qubit_num):
        if i not in measurement_list:
            keep_list.append(i)
    other_matrix = slice_density_matrix(other_matrix,keep_list)
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