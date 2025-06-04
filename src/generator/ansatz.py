# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ansatz module for constructing quantum circuits with specific gates"""

import numpy as np

from tools.qcircuit import QuantumCircuit
from tools.qgates import QuantumGate


def construct_qcircuit_XX_YY_ZZ_Z(qc: QuantumCircuit, size: int, layer: int) -> QuantumCircuit:
    """Construct a quantum circuit with the ansatz of XYZ and FieldZ

    Args:
        qc (QuantumCircuit): Quantum Circuit
        size (int): Size of the Quantum Circuit
        layer (int): Number of layers

    Returns:
        QuantumCircuit: Quantum Circuit with the ansatz of XYZ and FieldZ
    """

    entg_list = ["XX", "YY", "ZZ"]
    for j in range(layer):
        for i in range(size):
            if i < size - 1:
                for gate in entg_list:
                    qc.add_gate(QuantumGate(gate, i, i + 1, angle=0.5000 * np.pi))
                qc.add_gate(QuantumGate("Z", i, angle=0.5000 * np.pi))
        for gate in entg_list:
            qc.add_gate(QuantumGate(gate, 0, size - 1, angle=0.5000 * np.pi))
        qc.add_gate(QuantumGate("Z", size - 1, angle=0.5000 * np.pi))
        # qc.add_gate(Quantum_Gate("G", None, angle=0.5000 * np.pi))

    theta = np.random.randn(len(qc.gates))
    for i in range(len(qc.gates)):
        qc.gates[i].angle = theta[i]

    return qc


def construct_qcircuit_ZZ_X_Z(qc: QuantumCircuit, size: int, layer: int) -> QuantumCircuit:
    """Construct a quantum circuit with the ansatz of ZZ and XZ

    Args:
        qc (QuantumCircuit): Quantum Circuit
        size (int): Size of the Quantum Circuit
        layer (int): Number of layers

    Returns:
        QuantumCircuit: Quantum Circuit with the ansatz of ZZ and XZ
    """
    for j in range(layer):
        for i in range(size):
            qc.add_gate(QuantumGate("X", i, angle=0.5000 * np.pi))
            qc.add_gate(QuantumGate("Z", i, angle=0.5000 * np.pi))
        for i in range(size - 1):
            qc.add_gate(QuantumGate("ZZ", i, i + 1, angle=0.5000 * np.pi))

    theta = np.random.randn(len(qc.gates))
    for i in range(len(qc.gates)):
        qc.gates[i].angle = theta[i]

    return qc


def get_ansatz_func(type_of_ansatz: str) -> callable:
    """Construct the ansatz based on the type specified.

    Args:
        type_of_ansatz (str): Type of ansatz to construct, either 'XX_YY_ZZ_Z' or 'ZZ_X_Z'.

    Returns:
        callable: Function to construct the quantum circuit with the specified ansatz.
    """
    if type_of_ansatz == "XX_YY_ZZ_Z":
        return construct_qcircuit_XX_YY_ZZ_Z
    if type_of_ansatz == "ZZ_X_Z":
        return construct_qcircuit_ZZ_X_Z
    raise ValueError("Invalid type of ansatz specified.")
