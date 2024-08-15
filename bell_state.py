import qubits
import circuits
import math

def entangler(qubit_matrix,qubit_a_i=0,qubit_b_i=1):
    qubit_n = int(math.log2(qubit_matrix.shape[0]))
    plan = [["H",[qubit_a_i]],["CNOT",[qubit_a_i,qubit_b_i]]]
    creator = circuits.Circuit(qubit_n, plan)
    return creator.run(qubit_matrix)

def measurement(qubit_matrix,qubit_a_i=0,qubit_b_i=1):
    qubit_n = int(math.log2(qubit_matrix.shape[0]))
    plan = [["CNOT", [qubit_a_i, qubit_b_i]], ["H", [qubit_a_i]]]
    decoder = circuits.Circuit(qubit_n, plan)
    return decoder.run(qubit_matrix)

def unitary_operation(qubit_matrix,qubit_a_i=0,state="00",bell_state_o="00"):
    """
    phi_plus    |Φ+>:00
    phi_minus   |Φ->:10
    psi_plus    |Ψ+>:01
    psi_minus   |Ψ->:11
    """
    encode = {"Φ+":"00","Φ-":"10","Ψ+":"01","Ψ-":"11"}
    qubit_n = int(math.log2(qubit_matrix.shape[0]))

    plan = [["I",[qubit_a_i]]]
    if state+bell_state_o in [encode["Φ+"]+encode["Φ-"],encode["Φ-"]+encode["Φ+"],encode["Ψ+"]+encode["Ψ-"],encode["Ψ-"]+encode["Ψ+"]]:
        plan = [["Z", [qubit_a_i]]]
    elif state+bell_state_o in [encode["Φ+"]+encode["Ψ+"],encode["Ψ+"]+encode["Φ+"],encode["Φ-"]+encode["Ψ-"],encode["Ψ-"]+encode["Φ-"]]:
        plan = [["X", [qubit_a_i]]]
    elif state+bell_state_o in [encode["Φ+"]+encode["Ψ-"],encode["Ψ-"]+encode["Φ+"],encode["Φ-"]+encode["Ψ+"],encode["Ψ+"]+encode["Φ-"]]:
        plan = [["Z", [qubit_a_i]],["X", [qubit_a_i]]]
    return circuits.Circuit(qubit_n, plan).run(qubit_matrix)

def teleportation(qubit_c=qubits.get_qubit_matrix([0]),state_ab="00"):
    qubits_ab = qubits.get_qubit_matrix([s for s in state_ab])
    # create bell state
    bell_ab = entangler(qubits_ab, 0, 1)
    qubits_c_bell_ab = qubits.to_muti_qubit_matrix([qubit_c, bell_ab])
    qubits_bell_ca_b = measurement(qubits_c_bell_ab, 0, 1)
    measurement_ca, qubit_b = qubits.measurement(qubits_bell_ca_b, [0, 1])
    return qubit_b,measurement_ca