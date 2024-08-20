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
import jupyter_function

def test_message_in_superposition_state(message=1,decode_key=1,index_test_n = 1000,index_list = range(0, 7),thread_number=8,state_ab="00"):
    qubit_c = qubits.normalization(np.array([[message],
                                             [decode_key]]))

    qubit_b, state_ca = bell_state.teleportation(qubit_c, state_ab)
    print(qubit_b)
    qubit_b = bell_state.unitary_operation(qubit_b, 0, state_ca, state_ab)
    print(qubit_b)
    print(qubit_c)

    accuracy_list = []
    for i in index_list:
        # multi_thread
        thread_n = thread_number
        task_manager = Manager()
        result_queue = task_manager.Queue(thread_n)
        submit_total = 0
        task_list = []
        for t_i in range(thread_n):
            if t_i is not thread_n - 1:
                submit_n = int(index_test_n / thread_n)
            else:
                submit_n = index_test_n - submit_total
            task = Process(target=jupyter_function.multi_thread_measurement,
                           args=[submit_n, message, i, qubit_b, decode_key, result_queue])
            task.start()
            task_list.append(task)
            submit_total += submit_n
        for task in task_list:
            task.join()
            task.close()

        wrong_n = 0
        for r_i in range(result_queue.qsize()):
            wrong_n += result_queue.get()
        accuracy = (index_test_n - wrong_n) / index_test_n
        accuracy_list.append(accuracy)
        print(i,accuracy)
    return accuracy_list

if __name__ == '__main__':
    thread_number = 16
    index = range(2, 4)
    accuracy_3_4 = test_message_in_superposition_state(message=9, decode_key=5, index_test_n=1000,
                                                       index_list=index, thread_number=thread_number)

    plt.plot(index, np.array(accuracy_3_4) * 100, label="send 3 to 4 binary bits")
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Number of qubit [10ⁿ]")
    plt.title("How many qubits need to be used when send a message base on the superposition state")
    plt.legend()
    plt.show()

    state_ab = "00"
    message = 1
    #qubit_c = np.array([[0.6],[0.8]])
    qubit_c = qubits.get_qubit_matrix([message])
    density_c = qubits.get_density_matrix(qubit_c)
    density_b, state_ca = bell_state.teleportation(density_c, state_ab,[0.5,1])
    density_b = bell_state.unitary_operation(density_b, 0, state_ca, state_ab)
    state_b = qubits.measurement(density_b, [0])[0]
    print(qubits.get_fidelity(density_c, density_b))
    print("qubit_c(sender):\n{}\nmeasurement_ca(send):{}\nqubit_b(receiver):\n{}".format(density_c, state_ca, density_b))