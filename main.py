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
    message = "Hello World!"
    print("[Send]:{}".format(message))
    message_encode = format(int(bytes(message, 'utf-8').hex(), base=16), 'b')
    print("[Encode]:{}".format(message_encode))

    receive_buffer=""
    for m in message_encode:
        state_ab = [0,1]
        #qubits_ab = qubits.get_qubit_matrix([np.random.choice([0,1]),np.random.choice([0,1])])
        qubits_ab = qubits.get_qubit_matrix(state_ab)
        qubit_b,measurement_ca = bell_state.teleportation(int(m),state_ab)

        message_b = qubits.measurement(qubit_b,[0])[0]
        receive_buffer += message_b
        print("qubit_c(sender):{} | measurement_ca(send):{} | qubit_b(receiver):{}".format(m, measurement_ca, message_b))
    print("[Receive]:{}".format(receive_buffer))
    message_decode = bytes.fromhex(format(int(receive_buffer, base=2), 'x')).decode('utf-8')
    print("[Decode]:{}".format(message_decode))
