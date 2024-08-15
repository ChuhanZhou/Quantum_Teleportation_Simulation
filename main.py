import numpy as np
import gates
import qubits
import circuits
import bell_state
from matplotlib import pyplot as plt
import datetime
import ipywidgets as widgets
from ipywidgets import interact

if __name__ == '__main__':
    message = bin(9).split("0b")[-1]
    print(message)
    message_encode_d = int(message, 2)
    print(message_encode_d)
    decode_key = 5
    state_ab = "00"
    qubit_c = qubits.normalization(np.array([[message_encode_d],
                                             [decode_key]]))
    receive = {"0":0,"1":0}

    qubit_b, state_ca = bell_state.teleportation(qubit_c, state_ab)
    qubit_b = bell_state.unitary_operation(qubit_b, 0, state_ca, state_ab)
    for i in range(10 ** 3):
        state_b = qubits.measurement(qubit_b, [0])[0]
        #print("qubit_c(sender):{} | measurement_ca(send):{} | qubit_b(receiver):{}".format("N/A", state_ca, state_b))
        receive[state_b] = receive[state_b]+1
    receive_message = round(decode_key/receive["1"]*receive["0"])
    print(receive)
    print(receive_message)

    state_ab = "00"
    message = "1"
    message_encode_d = int(format(int(bytes(message, 'utf-8').hex(), base=16), 'b'), 2)
    print(message_encode_d)
    decode_key = 50
    qubit_c = qubits.normalization(np.array([[message_encode_d],
                                             [decode_key]]))
    receive = {"0": 0, "1": 0}

    qubit_b, state_ca = bell_state.teleportation(qubit_c, state_ab)
    qubit_b = bell_state.unitary_operation(qubit_b, 0, state_ca, state_ab)
    for i in range(10 ** 4):
        state_b = qubits.measurement(qubit_b, [0])[0]
        receive[state_b] = receive[state_b] + 1
    print(receive)
    receive_d = round(decode_key / receive["1"] * receive["0"])
    print(receive_d)
    receive_message = bytes.fromhex(format(int(bin(receive_d).split("0b")[-1], base=2), 'x')).decode('utf-8')
    print(receive_message)

