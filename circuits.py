import numpy as np
from multiprocessing import Manager,Process
from matplotlib import pyplot as plt
import datetime

import gates
import qubits

class  Circuit():
    def __init__(self,input_n=0,gate_plan = [],id_list = list(range(0))):
        """
        gate_plan: [[gate_name,[gate_index_0,...]],...]
            ex: ["CNOT",[contral_qubit_i,target_qubit_i]],
                ["CCNOT",[contral_qubit_i,contral_qubit_i,target_qubit_i]],
                ["measurement",[start_qubit_i,end_qubit_i]]
        """
        self.id_list = id_list
        if len(self.id_list) < input_n:
            self.id_list+=list(range(input_n-len(self.id_list)))
        self.gate_plan = gate_plan
        self.blue_print, self.gate_list,gate_str_list = self.create_blue_print(input_n,gate_plan)
        self.circuit_matrix,self.circuit_str = self.create_circuit_matrix(self.blue_print, self.gate_list,gate_str_list)

    def run(self,qubit_matrix):
        if qubit_matrix.shape[0] == qubit_matrix.shape[1] and qubit_matrix.shape[0] > 1:
            return self.run_density_matrix(qubit_matrix)
        else:
            return self.run_state_vector(qubit_matrix)

    def run_state_vector(self,qubit_matrix):
        # v=U*v
        out = np.dot(self.circuit_matrix,qubit_matrix)
        #normalization => <ψ|ψ>=1
        out = qubits.normalization(out)
        return out

    def run_density_matrix(self,qubit_matrix):
        # ρ′=U*ρ*U.T
        out = np.dot(np.dot(self.circuit_matrix, qubit_matrix), self.circuit_matrix.T)
        # normalization => <ψ|ψ>=1
        out = qubits.normalization(out)
        return out

    def create_blue_print(self,input_n=0,gate_plan=[]):
        blue_print = np.ones((input_n, len(gate_plan))).astype(int) * -1

        gate_list = []
        gate_str_list = []
        step = 0
        for g_name, g_indexs in gate_plan:
            # single_qubit_gate
            if len(g_indexs) == 1:
                gate,print_str = gates.get_gate_by_name(g_name)
                gate_list.append(gate)
                gate_str_list.append(print_str)
                if blue_print[g_indexs[0], step] != -1:
                    step += 1
                blue_print[g_indexs[0], step] = len(gate_list) - 1
            # muti_qubit_gate
            elif len(g_indexs) > 1:
                if g_indexs[0] != g_indexs[-1]:
                    is_inverse = g_indexs[0] > g_indexs[-1]
                    g_indexs.sort()
                    inner_gates = []
                    for g_i in range(len(g_indexs) - 1):
                        i_gate_num = abs(g_indexs[g_i] - g_indexs[g_i + 1]) - 1
                        inner_gate = np.array([[1]])
                        inner_print = []
                        if i_gate_num > 0:
                            for i in range(i_gate_num):
                                print_str = ["   ┃   ",
                                             "---┃---",
                                             "   ┃   "]
                                inner_gate = np.kron(inner_gate,gates.get_identity_gate()[0])
                                inner_print+=print_str
                            inner_gates.append([inner_gate,inner_print])
                        else:
                            inner_gates.append([inner_gate,inner_print])
                    gate, print_str = gates.get_gate_by_name(g_name, is_inverse, inner_gates)
                    gate_list.append(gate)
                    gate_str_list.append(print_str)
                    if blue_print[min(g_indexs):max(g_indexs) + 1, step].max() != -1:
                        step += 1
                    blue_print[min(g_indexs):max(g_indexs) + 1, step] = len(gate_list) - 1
        blue_print = blue_print[:,0:step+1]
        return blue_print,gate_list,gate_str_list

    def create_circuit_matrix(self,blue_print=np.zeros((0, 0)), gate_list=[],gate_str_list=[]):
        circuit_matrix = 1
        circuit_str = []
        for i in range(blue_print.shape[0]*3):
            circuit_str.append("")

        for step_i in range(blue_print.shape[1]):
            step_matrix = 1
            gate_i_old = -1
            step_str = []
            for gate_i in blue_print[:,step_i:step_i+1].T[0]:
                gate = 1
                gate_str = []
                if gate_i >= 0:#has gate
                    if gate_i != gate_i_old:#is new gate
                        gate = gate_list[gate_i]
                        gate_str+=gate_str_list[gate_i]
                        gate_i_old = gate_i
                elif gate_i == -1:#no gate
                    gate = gates.get_identity_gate()[0]
                    gate_str+=["       ",
                               "-------",
                               "       "]
                step_matrix = np.kron(step_matrix,gate)
                step_str+=gate_str
            circuit_matrix = np.dot(step_matrix,circuit_matrix)

            for line_i in range(len(circuit_str)):
                step_to_step_str = "  "
                if line_i % 3 == 1:#middle of each line of circuit
                    step_to_step_str = "--"

                circuit_str[line_i] += step_to_step_str + step_str[line_i] + step_to_step_str
        return circuit_matrix,circuit_str

    def get_print(self):
        circuit_print_str = ""
        circuit_print_list = []
        input_str = []

        max_length = 0
        for i in range(len(self.id_list)):
            max_length=max(max_length,len(str(self.id_list[i]))+4)
        empty_str = ""
        for i in range(max_length):
            empty_str += " "

        for i in range(len(self.id_list)):
            input_str+=[empty_str,
                        " Q{}:{} ".format(str(self.id_list[i]),empty_str[0:max_length-4-len(str(self.id_list[i]))]),
                        empty_str]
        for i in range(len(input_str)):
            circuit_print_str += input_str[i]+self.circuit_str[i]+"\n"
            circuit_print_list.append(input_str[i]+self.circuit_str[i])
        return circuit_print_str,circuit_print_list

def get_random_circuit(input_n=3,gate_n=10):
    gate_plan = []
    g_list = gates.gate_list
    g_weighted_list = []
    for t_i in range(len(g_list)):
        if t_i>=input_n:
            break
        t_len = (t_i+1)*len(g_list[t_i])
        for t_len_i in range(t_len):
            g_weighted_list.append(t_i)
    g_list_weighting = np.array(g_weighted_list)
    for g_i in range(gate_n):
        g_type = np.random.choice(g_list_weighting)
        g_name = g_list[g_type][np.random.randint(0,len(g_list[g_type]))]
        g_indexs = []
        g_index_old = -1
        for index_i in range(g_type+1):
            g_index_old = np.random.randint(g_index_old+1,input_n-g_type+index_i)
            g_indexs.append(g_index_old)#(input_n-(g_type+1)+1)
        if bool(np.random.randint(0,2)):
            g_indexs.reverse()
        gate_plan.append([g_name,g_indexs])
    return Circuit(input_n,gate_plan)

#random_circuit_sampling
def rcs(input_n = 3,gate_n = 15,batch_n = 1000):
    basic_qubit_list = []
    for i in range(input_n):
        basic_qubit_list.append(0)

    f_list = []
    start_time = datetime.datetime.now()
    for f_i in range(batch_n):
        #circuit_0 = get_random_circuit(input_n, np.random.randint(1, gate_max_n + 1))
        #circuit_1 = get_random_circuit(input_n, np.random.randint(1, gate_max_n + 1))
        circuit_0 = get_random_circuit(input_n, gate_n)
        circuit_1 = get_random_circuit(input_n, gate_n)
        input_matrix = qubits.get_qubit_matrix(basic_qubit_list)
        result_0 = circuit_0.run(input_matrix)
        result_1 = circuit_1.run(input_matrix)
        # fidelity
        f = qubits.get_fidelity(result_0,result_1)
        f_list.append(f)
        if (f_i + 1) % (batch_n / 10) == 0:
            print("[{}] {}|{}".format(datetime.datetime.now(),batch_n, f_i + 1))
    end_time = datetime.datetime.now()
    print("(2 circuits per batch) avg_batch_running_time[hh:mm:ss]: {}".format((end_time-start_time)/batch_n))
    return f_list

def rcs_multi_thread(input_n = 3,gate_max_n = 15,batch_n = 1000,thread_n=8,safe_print=False):
    """
    safe_print : when using jupyter notebook
    """
    thread_n = min(thread_n,batch_n)
    basic_qubit_list = []
    for i in range(input_n):
        basic_qubit_list.append(0)
    f_list = []
    task_manager = Manager()
    f_queue = task_manager.Queue(batch_n)
    p_queue = task_manager.Queue(thread_n*5)

    start_time = datetime.datetime.now()
    print("[{}] thread_number: {}".format(start_time,thread_n))
    batch_submit_total = 0
    task_list = []
    for i in range(thread_n):
        if i is not thread_n-1:
            submit_n = int(batch_n/thread_n)
        else:
            submit_n = batch_n-batch_submit_total
        task = Process(target=get_fidelities, args=[submit_n,basic_qubit_list,batch_n,gate_max_n,f_queue,p_queue,safe_print])
        task.start()
        task_list.append(task)
        batch_submit_total += submit_n

    for task in task_list:
        task.join()
        task.close()

    if safe_print:
        for p_i in range(p_queue.qsize()):
            print(p_queue.get())

    for f_i in range(f_queue.qsize()):
        f_list.append(f_queue.get())
    end_time = datetime.datetime.now()
    print("[{}] (2 circuits per batch) avg_batch_running_time[hh:mm:ss]: {}".format(datetime.datetime.now(),(end_time-start_time)/batch_n))
    return f_list

#function for multi thread
def get_fidelities (loop_n,basic_qubit_list,batch_n,gate_n,f_q,p_q,safe_print=False):
    for f_i in range(loop_n):
        circuit_0 = get_random_circuit(len(basic_qubit_list), gate_n)
        circuit_1 = get_random_circuit(len(basic_qubit_list), gate_n)
        input_matrix = qubits.get_qubit_matrix(basic_qubit_list)
        result_0 = circuit_0.run(input_matrix)
        result_1 = circuit_1.run(input_matrix)
        # fidelity
        f = qubits.get_fidelity(result_0,result_1)
        f_q.put(f)
        finish_n = f_q.qsize()
        if finish_n % (batch_n / 10) == 0:
            info = "[{}] {}|{}".format(datetime.datetime.now(), batch_n, finish_n)
            p_q.put(info)
            if not safe_print:
                print(info)

def show_p_f(f_list=[],input_n = None,gate_n = None,batch_n = None):
    ax = plt.figure().add_subplot()
    ax.hist(f_list, rwidth=0.75, align="left", bins=np.arange(-0.1, 1.1, 0.05),label="P(F)")
    ax.set_xlabel("Fidelity")
    ax.set_ylabel("Batch number")
    ax.set_xticks(np.arange(-0.1, 1.1, 0.1))
    ax.set_title("qubit_num:{}|batch_num:{}|gate_num:{}".format(input_n, batch_n, gate_n))

    if input_n is not None and batch_n is not None:
        line = ax.patches[::len(ax.patches)]
        label = [p.get_label() for p in ax.patches[::len(ax.patches)]]
        show_p_haar_f(input_n,[ax,line,label])
    plt.show()

def show_p_haar_f(input_n = 1, main_info = None):
    f = np.linspace(0, 1, 1000)
    N = 2 ** input_n
    if main_info is None:
        plt.plot(f, (N - 1) * (1 - f) ** (N - 2),label="P_Haar(f)")
        plt.xticks(np.arange(-0.1, 1.15, 0.1))
        plt.title("P_Haar(f)|qubit_num:{}|Hilbert_space_dim:{}".format(input_n, N))
        plt.show()
    else:
        ax_0,line_0,label_0 = main_info
        ax_1 = ax_0.twinx()
        line_1 = ax_1.plot(f, (N - 1) * (1 - f) ** (N - 2), label="P_Haar(F)",color="tab:orange")
        ax_1.set_ylim(bottom=0)
        label_1 = [l.get_label() for l in line_1]
        lines = line_0+line_1
        labels = label_0 + label_1
        ax_0.legend(lines,labels)