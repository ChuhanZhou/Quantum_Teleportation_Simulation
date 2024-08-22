import numpy as np

import qubits
import circuits
import bell_state

#function for muti-thread in Jupyter Notebook
def multi_thread_measurement(message,decode_key,digit_n, epoch_n,qubit_b,max_index,q):
    digit_diff_list = np.ones((digit_n, max_index, epoch_n))
    for e_n in range(epoch_n):
        batch_n = 10 ** max_index
        batch_result = []

        for b_n in range(batch_n):
            state_b = qubits.measurement(qubit_b, [0])[0]
            batch_result.append(int(state_b))

        #calculate difference between received message and true message for decimal digits
        for i in range(0, max_index):
            sample_n = 10 ** i
            result = {"0": 0, "1": 0}
            sample = np.array(batch_result[0:sample_n])
            result["1"] = len(np.nonzero(sample)[0])
            result["0"] = sample_n - result["1"]

            if result["1"] is not 0:
                receive_message = decode_key / np.sqrt(result["1"]) * np.sqrt(result["0"])
            else:
                receive_message = 0

            for d_i in range(digit_n):
                message_i = message * 10 ** d_i
                diff = round(receive_message * 10 ** d_i) - message_i
                digit_diff_list[d_i][i][e_n] = diff
    q.put(digit_diff_list)

if __name__ == '__main__':
    teleportation_circuit = circuits.Circuit(3, [["H", [1]],
                                                 ["CNOT", [1, 2]],
                                                 ["CNOT", [0, 1]],
                                                 ["H", [0]],
                                                 ["M", [0, 1]],
                                                 ["CX", [1, 2]],
                                                 ["CZ", [0, 2]]], ["c", "a", "b"])
    tele_print_str, tele_print_list = teleportation_circuit.get_print()
    print(tele_print_str)

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