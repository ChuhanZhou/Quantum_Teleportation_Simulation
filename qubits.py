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

def normalization(qubit_matrix):
    """
    normalization=> <ψ|ψ>=1
    """
    s = (np.abs(qubit_matrix) ** 2).sum()
    qubit_matrix_norm = qubit_matrix / np.sqrt(s)
    return qubit_matrix_norm

def measurement(qubit_matrix,measurement_list=[]):
    '''
    measurement_list:[measurement_i_0,measurement_i_1,...]
    '''
    qubit_num = int(math.log2(qubit_matrix.shape[0]))
    matrix_part_prob = {}
    options_probabilities = {}
    for o_i in range(qubit_matrix.shape[0]):
        decode_all = bin(o_i).split("0b")[-1].zfill(qubit_num)#from 2,3 to 010,011
        decode_measure = ""
        decode_other = ""
        for i in range(0,qubit_num):
            if i not in measurement_list:
                decode_other+=decode_all[i]
            else:
                decode_measure+=decode_all[i]
        part_prob = qubit_matrix[o_i,0]
        if decode_measure in options_probabilities.keys():
            matrix_part_prob.get(decode_measure).update({decode_other:part_prob})
            options_probabilities.update({decode_measure:options_probabilities.get(decode_measure)+abs(part_prob)})
        else:
            matrix_part_prob.update({decode_measure:{decode_other:part_prob}})
            options_probabilities.update({decode_measure:abs(part_prob)})
    options = list(options_probabilities.keys())
    probabilities = list(options_probabilities.values())
    result = random.choices(options,probabilities)[0]

    #create matrix for qubits which is not measured
    other_prob = matrix_part_prob.get(result)
    other_qubit_num = qubit_num-len(measurement_list)
    other_matrix = np.zeros((2**other_qubit_num,1))
    for i in range(other_matrix.shape[0]):
        decode = bin(i).split("0b")[-1].zfill(other_qubit_num)
        other_matrix[i,0] = other_prob.get(decode)
    other_matrix = qubits.normalization(other_matrix)
    return result,other_matrix

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