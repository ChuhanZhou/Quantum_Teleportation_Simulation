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

if __name__ == '__main__':
    qubits_abc = qubits.get_qubit_matrix([1,1,1])
    density_abc = qubits.get_density_matrix(qubits_abc)
    slice_abc = qubits.slice_density_matrix(density_abc)
    a=1



    message = "Hello World!"
    print("[Send]:{}".format(message))
    message_encode = format(int(bytes(message, 'utf-8').hex(), base=16), 'b')
    print("[Encode]:{}".format(message_encode))

    receive_buffer = ""
    for m in message_encode:
        state_ab = "10"
        qubits_ab = qubits.get_qubit_matrix(state_ab)
        qubit_c = qubits.get_qubit_matrix([m])
        qubit_b, state_ca = bell_state.teleportation(qubit_c, state_ab, [0.1])

        qubit_b = bell_state.unitary_operation(qubit_b, 0, state_ca, state_ab)
        state_b = qubits.measurement(qubit_b, [0])[0]
        receive_buffer += state_b
        a = qubits.get_fidelity(qubit_c,qubit_b)
        print(a)
        #print("qubit_c(sender):{} | measurement_ca(send):{} | qubit_b(receiver):{}".format(m, state_ca, state_b))

    print("[Receive]:{}".format(receive_buffer))
    message_decode = bytes.fromhex(format(int(receive_buffer, base=2), 'x')).decode('utf-8')
    print("[Decode]:{} ({} binary bit)".format(message_decode, len(receive_buffer)))