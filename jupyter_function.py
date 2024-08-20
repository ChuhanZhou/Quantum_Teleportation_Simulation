import numpy as np
import qubits

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
            receive_message = round(decode_key / receive["1"] * receive["0"])
        else:
            receive_message = 1
        result[t_n, 0] = receive_message
        print(receive_message,decode_key,receive["1"],receive["0"])
    wrong_n = len(np.nonzero(result - answer)[0])
    q.put(wrong_n)
