#from qutip import about, basis, destroy, mcsolve, mesolve
import math
from scipy.special import genlaguerre
import density_matrix as density_matrix

import numpy as np

N = 20

def get_creation_operator(N=N):
    """
    a†
    """
    a_dag = np.zeros((N, N))
    for n in range(1, N):
        a_dag[n ,n - 1] = np.sqrt(n)
    return a_dag


def get_annihilation_operator(N):
    """
    a
    """
    a = get_creation_operator(N).T
    return a

#def get_quantum_state(alpha=[0,0],density_matrix=[]):
#    alpha = alpha[0]+alpha[1]*1j

def get_fock_basic(n,N=N):
    basic_n = np.zeros((N,1))
    basic_n[n,0] = 1
    return basic_n

def get_coherent_state(alpha=0.0,N=N):
    # basic state
    vacuum_state = get_fock_basic(0,N)
    a_dag = get_creation_operator(N)
    a = get_annihilation_operator(N)

    #D(α) = exp(αa^† - α*a)
    D_alpha = np.exp(alpha * a_dag - np.conjugate(alpha) * a)

    coherent_state = np.dot(D_alpha, vacuum_state)
    return coherent_state

def get_squeezed_state(basic_state=get_fock_basic(0,N),r=0,N=N):
    a_dag = get_creation_operator(N)
    a = get_annihilation_operator(N)
    squeezing_operator = np.exp(r * (np.dot(a,a) - np.dot(a_dag,a_dag)) / 2)
    if r!=0:
        squeezed_state = np.dot(squeezing_operator,basic_state)
    else:
        squeezed_state = basic_state
    return squeezed_state

def get_epr_state(alpha_list=[0.0,0.0],r=0,N=N):
    psi_a = get_squeezed_state(get_coherent_state(alpha_list[0],N),r,N)
    psi_b = get_squeezed_state(get_coherent_state(alpha_list[1],N),r,N)
    epr_mode_1 = (psi_a + psi_b) / np.sqrt(2)
    epr_mode_2 = (psi_a - psi_b) / np.sqrt(2)
    return epr_mode_1, epr_mode_2


def bell_state_measurement(mode1, mode2, N=N):
    combined_state = np.kron(mode1, mode2)
    return combined_state
#def get_matrix_element(m,n,rho):
#    state_m = get_fock_basic(m)
#    state_n = get_fock_basic(n)
#    rho = density_matrix.plus([np.zeros((state_m.shape[0],state_n.shape[0])),rho])
#    rho_mn = np.dot(np.dot(state_m.T,rho),state_n)
#    return rho_mn
#
#def get_wigner_function(x,p,m,n,rho):
#    w_mn_xp = 1/np.pi*(np.e**(-x**2-p**2))*(-1)**n*(x-1j)**(m-n)*np.sqrt(2**(m-n)*math.factorial(n)/math.factorial(m))*genlaguerre(n, m-n)(2*x**2+2*p**2)
#    rho_mn = get_matrix_element(m,n,rho)
#    w_xp = float(np.sum(np.dot(rho_mn,w_mn_xp)))
#    return w_xp

