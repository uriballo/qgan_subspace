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
"""Generator module implemented in PyTorch."""

import torch
import torch.nn as nn
import itertools
from config import CFG
from tools.qobjects.qcircuit import QuantumCircuit
from tools.qobjects.qgates import QuantumGate, Identity
import os

class Ansatz(nn.Module):
    """
    Ansatz class for constructing quantum circuits with specific gates.
    Implemented as a torch.nn.Module to be part of the Generator model.
    """
    def __init__(self, size: int, layers: int, ansatz_type: str, config=CFG):
        super().__init__()
        self.config = config
        self.qc = QuantumCircuit(size)
        
        # Build the circuit based on the specified ansatz type
        if ansatz_type == "XX_YY_ZZ_Z":
            self.construct_qcircuit_XX_YY_ZZ_Z(self.qc, size, layers)
        elif ansatz_type == "ZZ_X_Z":
            self.construct_qcircuit_ZZ_X_Z(self.qc, size, layers)
        else:
            raise ValueError("Invalid type of ansatz specified.")
            
        self.randomize_gates_in_qc(self.qc, size)

    def forward(self):
        return self.qc()

    def construct_qcircuit_XX_YY_ZZ_Z(self, qc: QuantumCircuit, size: int, layer: int):
        # Ancilla logic reduces the main system size by 1
        system_qubits = size - 1 if self.config.extra_ancilla else size
        
        entg_list = ["XX", "YY", "ZZ"]
        for _ in range(layer):
            for i in range(system_qubits):
                qc.add_gate(QuantumGate("Z", i, angle=0.0))
            if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
                qc.add_gate(QuantumGate("Z", system_qubits, angle=0.0))

            for i, gate in itertools.product(range(system_qubits - 1), entg_list):
                qc.add_gate(QuantumGate(gate, i, i + 1, angle=0.0))
                
            # Add ancilla coupling if specified
            if self.config.extra_ancilla:
                ancilla_q = system_qubits
                if self.config.ancilla_topology == "total":
                    for i, gate in itertools.product(range(system_qubits), entg_list):
                        qc.add_gate(QuantumGate(gate, i, ancilla_q, angle=0.0))
                elif self.config.ancilla_topology in ["bridge", "ansatz"]:
                    connect_to = self.config.ancilla_connect_to if self.config.ancilla_connect_to is not None else system_qubits - 1
                    for gate in entg_list:
                        qc.add_gate(QuantumGate(gate, connect_to, ancilla_q, angle=0.0))
                    if self.config.ancilla_topology == "bridge":
                         for gate in entg_list:
                            qc.add_gate(QuantumGate(gate, 0, ancilla_q, angle=0.0))

    def construct_qcircuit_ZZ_X_Z(self, qc: QuantumCircuit, size: int, layer: int):
        system_qubits = size - 1 if self.config.extra_ancilla else size

        for _ in range(layer):
            for i in range(system_qubits):
                qc.add_gate(QuantumGate("X", i, angle=0.0))
                qc.add_gate(QuantumGate("Z", i, angle=0.0))
            if self.config.extra_ancilla and self.config.do_ancilla_1q_gates:
                qc.add_gate(QuantumGate("X", system_qubits, angle=0.0))
                qc.add_gate(QuantumGate("Z", system_qubits, angle=0.0))
            
            for i in range(system_qubits - 1):
                qc.add_gate(QuantumGate("ZZ", i, i + 1, angle=0.0))

            if self.config.extra_ancilla:
                ancilla_q = system_qubits
                if self.config.ancilla_topology == "total":
                    for i in range(system_qubits):
                        qc.add_gate(QuantumGate("ZZ", i, ancilla_q, angle=0.0))
                elif self.config.ancilla_topology in ["bridge", "ansatz"]:
                    connect_to = self.config.ancilla_connect_to if self.config.ancilla_connect_to is not None else system_qubits - 1
                    qc.add_gate(QuantumGate("ZZ", connect_to, ancilla_q, angle=0.0))
                    if self.config.ancilla_topology == "bridge":
                        qc.add_gate(QuantumGate("ZZ", 0, ancilla_q, angle=0.0))

    def randomize_gates_in_qc(self, qc: QuantumCircuit, size: int):
        ancilla_q = size - 1 if self.config.extra_ancilla else -1
        with torch.no_grad():
            for gate in qc.gates:
                is_ancilla_gate = ancilla_q != -1 and (gate.qubit1 == ancilla_q or gate.qubit2 == ancilla_q)
                if not is_ancilla_gate or self.config.start_ancilla_gates_randomly:
                    gate.angle.uniform_(0, 2 * torch.pi)

class Generator(nn.Module):
    """Generator class for the Quantum GAN, implemented as a PyTorch Module."""

    def __init__(self, config=CFG, seed=None):
        super().__init__()
        self.config = config
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.size: int = self.config.system_size + (1 if self.config.extra_ancilla else 0)
        self.target_size: int = self.config.system_size
        
        # Store config for loading/saving compatibility checks
        self.ancilla: bool = self.config.extra_ancilla
        self.ancilla_topology: str = self.config.ancilla_topology
        self.ansatz_type: str = self.config.gen_ansatz
        self.layers: int = self.config.gen_layers
        self.target_hamiltonian: str = self.config.target_hamiltonian

        # The circuit is now a submodule of the Generator
        self.ansatz = Ansatz(self.size, self.layers, self.ansatz_type, self.config)

    def forward(self, total_input_state: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the generator. It applies the quantum circuit
        to the input state.
        """
        # Get the unitary matrix representation of the quantum circuit
        device = total_input_state.device
        generator_unitary = self.ansatz().to(device)

        # The full operator includes the identity on the untouched qubits
        # Identity is created on the correct device by the qgates module
        full_op = torch.kron(Identity(self.target_size), generator_unitary)
        
        # Apply the operator to the input state
        total_gen_state = torch.matmul(full_op, total_input_state)
        return total_gen_state

    def load_model_params(self, file_path: str):
        """Loads generator parameters from a saved state_dict."""
        try:
            # Load the entire saved dictionary, which includes config
            saved_data = torch.load(file_path)
            saved_config = saved_data.get('config', {})
            
            # --- Perform compatibility checks ---
            if saved_config.get('target_size') != self.target_size:
                raise ValueError("Incompatible target size.")
            if saved_config.get('target_hamiltonian') != self.target_hamiltonian:
                raise ValueError("Incompatible target Hamiltonian.")
            if saved_config.get('ansatz_type') != self.ansatz_type:
                raise ValueError("Incompatible ansatz type.")
            if saved_config.get('layers') != self.layers:
                raise ValueError("Incompatible number of layers.")
            
            # Load the state dictionary
            self.load_state_dict(saved_data['model_state_dict'])
            print(f"Generator parameters loaded successfully from {file_path}")

        except Exception as e:
            print(f"ERROR: Could not load generator model: {e}")

    def save_model_params(self, file_path: str):
        """Saves generator parameters and config to a file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save both the model's state_dict and its configuration
        save_data = {
            'model_state_dict': self.state_dict(),
            'config': {
                'target_size': self.target_size,
                'target_hamiltonian': self.target_hamiltonian,
                'ansatz_type': self.ansatz_type,
                'layers': self.layers,
            }
        }
        torch.save(save_data, file_path)
        print(f"Generator model saved to {file_path}")