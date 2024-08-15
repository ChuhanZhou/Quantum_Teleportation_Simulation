#from qutip import about, basis, destroy, mcsolve, mesolve
import math
from scipy.special import genlaguerre
import density_matrix as density_matrix

import numpy as np

N = 20

def get_quantum_state(alpha=[0,0],density_matrix=[]):
    alpha = alpha[0]+alpha[1]*1j

def get_fock_basic(n):
    basic_n = np.zeros((N+1,1))
    basic_n[n,0] = 1
    return basic_n

def get_matrix_element(m,n,rho):
    state_m = get_fock_basic(m)
    state_n = get_fock_basic(n)
    rho = density_matrix.plus([np.zeros((state_m.shape[0],state_n.shape[0])),rho])
    rho_mn = np.dot(np.dot(state_m.T,rho),state_n)
    return rho_mn

def get_wigner_function(x,p,m,n,rho):
    w_mn_xp = 1/np.pi*(np.e**(-x**2-p**2))*(-1)**n*(x-1j)**(m-n)*np.sqrt(2**(m-n)*math.factorial(n)/math.factorial(m))*genlaguerre(n, m-n)(2*x**2+2*p**2)
    rho_mn = get_matrix_element(m,n,rho)
    w_xp = float(np.sum(np.dot(rho_mn,w_mn_xp)))
    return w_xp

