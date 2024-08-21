import numpy as np

import gates
import qubits
import circuits
import bell_state
from matplotlib import pyplot as plt
from matplotlib import cm
import datetime
import ipywidgets as widgets
from ipywidgets import interact
from multiprocessing import Manager,Process
import noises

#function for muti-thread in Jupyter Notebook
def multi_thread_measurement(index_test_n,message,i,qubit_b,decode_key,q):
    result = np.ones((index_test_n, 1)) * -1
    answer = np.ones((index_test_n, 1)) * message
    for t_n in range(index_test_n):
        batch_n = 10 ** i
        receive = {"0": 0, "1": 0}

        for b_n in range(batch_n):
            state_b = qubits.measurement(qubit_b, [0])[0]
            receive[state_b] = receive[state_b] + 1

        if receive["1"] is not 0:
            receive_message = round(decode_key / np.sqrt(receive["1"]) * np.sqrt(receive["0"]))
        else:
            receive_message = 1
        result[t_n, 0] = receive_message
    wrong_n = len(np.nonzero(result - answer)[0])
    q.put(wrong_n)

if __name__ == '__main__':
    # Alice and Bob know state_ab, Alice gets qubit_a, Bob has qubit_b
    state_ab = "00"

    # Alice set message in qubit_c state (0 or 1)
    qubit_c = qubits.get_qubit(0.6, 0.8)

    # Alice do bell measurement on qubit_c and qubit_a to build entangled relationship
    # and this operation change the state of qubit_b base on qubit_c at the same time
    # then Alice send measurement result in traditional communication method to Bob
    qubit_b, state_ca = bell_state.teleportation(qubit_c, state_ab)

    # Bob use gates on qubit_b base on the measurement result
    # how to choose gates:
    #    measurement result is base on entangled relationship between qubit_c and qubit_a which is same like the relationship between qubit_a and qubit_b
    #    which means that how the measurement result changed from state_ab to state_bc is same like the state that qubit_c changed to qubit_b
    #    so when Bob know the state_ab and state_ca, he could find the way to restore state_ab from state_ca
    #    and he could do the same operation on qubit_b to restore the state of orignal qubit_c and get the message
    qubit_b = bell_state.unitary_operation(qubit_b, 0, state_ca, state_ab)
    state_b = qubits.measurement(qubit_b, [0])[0]
    print("qubit_c(sender):\n{}\nmeasurement_ca(send):{}\nqubit_b(receiver):\n{}".format(qubit_c, state_ca, qubit_b))