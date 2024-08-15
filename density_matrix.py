import numpy as np


# rho:ρ
def get_density_matrix(state_vector):
    rho = np.dot(state_vector, state_vector.T)
    return rho


def plus(matrix_list):
    out = np.zeros((1, 1))
    for matrix in matrix_list:
        result = np.zeros((max(out.shape[0],matrix.shape[0]),max(out.shape[1],matrix.shape[1])))
        result[0:out.shape[0], 0:out.shape[1]] += out[:, :]
        result[0:matrix.shape[0], 0:matrix.shape[1]] += matrix[:, :]
        out = result
    return out

def minus(basic,matrix_list):
    out = np.copy(basic)
    for matrix in matrix_list:
        result = np.zeros((max(out.shape[0], matrix.shape[0]), max(out.shape[1], matrix.shape[1])))
        result[0:out.shape[0], 0:out.shape[1]] += out[:, :]
        result[0:matrix.shape[0], 0:matrix.shape[1]] -= matrix[:, :]
        out = result
    return out