"""
qcircuit.py: definition of quantum circuit simulation.

"""

import os

import numpy as np

from tools.qgates import Identity, QuantumGate


class QuantumCircuit:
    def __init__(self, size, name):
        self.size = size
        self.depth = 0
        self.gates = []
        self.name = name

    def check_ciruit(self):
        for j, gate in zip(range(len(self.gates)), self.gates):
            if gate.qubit1 is not None and gate.qubit2 is not None:
                if gate.qubit1 > self.size - 1:
                    print("Error: #{} gate:{} 1qubit is out of range".format(j, gate.name))
                    os._exit(0)
                elif gate.qubit2 > self.size - 1:
                    print("Error: #{} gate:{} 2qubit is out of range".format(j, gate.name))
                    os._exit(0)

    def get_mat_rep(self):
        matrix = Identity(self.size)
        for gate in self.gates:
            g = gate.matrix_representation(self.size, False)
            matrix = np.matmul(g, matrix)
        return np.asmatrix(matrix)

    def get_grad_mat_rep(self, index, signal="none", type="matrix_multiplication"):
        """
            matrix multipliction: explicit way to calculate the gradient using matrix multiplication
            shift_phase: generate two quantum circuit to calculate the gradient
            Evaluating analytic gradients on quantum hardware
            https://arxiv.org/pdf/1811.11184.pdf
        :param index:
        :param type: the type of calculate gradient
        :return:
        """
        if type == "shift_phase":
            matrix = Identity(self.size)
            for j, gate in zip(range(len(self.gates)), self.gates):
                if index == j:
                    g = gate.matrix_representation_shift_phase(self.size, True, signal)
                    matrix = np.matmul(g, matrix)
                else:
                    g = gate.matrix_representation_shift_phase(self.size, False, signal)
                    matrix = np.matmul(g, matrix)
            return np.asmatrix(matrix)

        if type == "matrix_multiplication":
            matrix = Identity(self.size)
            for j, gate in zip(range(len(self.gates)), self.gates):
                if index == j:
                    g = gate.matrix_representation(self.size, True)
                    matrix = np.matmul(g, matrix)
                else:
                    g = gate.matrix_representation(self.size, False)
                    matrix = np.matmul(g, matrix)
            return np.asmatrix(matrix)

        return None

    def get_grad_qc(self, indx, type="0"):
        qc_list = []
        for j, gate in zip(range(len(self.gates)), self.gates):
            tmp = QuantumGate(" ", qubit1=None, qubit2=None, angle=None)
            tmp.name = gate.name
            tmp.qubit1 = gate.qubit1
            tmp.qubit2 = gate.qubit2
            tmp.angle = gate.angle
            if j == indx:
                try:
                    if self.gates[j].name != "G" or self.gates[j].name != "CNOT":
                        if type == "+":
                            tmp.angle = gate.angle + gate.s
                        elif type == "-":
                            tmp.angle = gate.angle - gate.s
                except:
                    print("param value error")
                qc_list.append(tmp)
            else:
                qc_list.append(tmp)
        return qc_list

    def add_gate(self, quantum_gate):
        self.depth += 1
        self.gates.append(quantum_gate)
