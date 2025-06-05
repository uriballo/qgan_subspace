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
"""Generator module"""

import os
import pickle

import numpy as np
from scipy.linalg import expm

from ancilla.ancilla import get_final_fake_state_for_discriminator, get_final_real_state_for_discriminator
from config import CFG
from optimizer.momentum_optimizer import MomentumOptimizer
from tools.data_managers import print_and_train_log
from tools.qcircuit import Identity, QuantumCircuit


class Generator:
    def __init__(self, system_size):
        self.size = system_size
        self.qc = self.init_qcircuit()
        self.optimizer = MomentumOptimizer()

    def set_qcircuit(self, qc):
        self.qc = qc

    def init_qcircuit(self):
        qcircuit = QuantumCircuit(self.size, "generator")
        return qcircuit

    def get_Untouched_qubits_and_Gen(self):
        """Get the matrix representation of the circuit at the Generator step, including the untouched qubits in front."""
        return np.kron(Identity(CFG.system_size), self.qc.get_mat_rep())

    def _grad_theta(self, dis, total_real_state, total_input_state):
        Untouched_x_G = self.get_Untouched_qubits_and_Gen()

        phi = dis.getPhi()
        psi = dis.getPsi()

        total_output_state = np.matmul(Untouched_x_G, total_input_state)

        final_fake_state = get_final_fake_state_for_discriminator(total_output_state)
        final_real_state = get_final_real_state_for_discriminator(total_real_state)

        try:
            A = expm((-1 / CFG.lamb) * phi)
        except Exception:
            print("grad_gen -1/CFG.lamb:\n", (-1 / CFG.lamb))
            print("size of phi:\n", phi.shape)

        try:
            B = expm((1 / CFG.lamb) * psi)
        except Exception:
            print("grad_gen 1/CFG.lamb:\n", (1 / CFG.lamb))
            print("size of psi:\n", psi.shape)

        grad_g_psi, grad_g_phi, grad_g_reg = [], [], []

        for i in range(self.qc.depth):
            grad_i = np.kron(Identity(CFG.system_size), self.qc.get_grad_mat_rep(i))
            # for psi term
            grad_g_psi.append(0)

            # for phi term
            fake_grad = np.matmul(grad_i, total_input_state)
            final_fake_grad = np.matrix(get_final_fake_state_for_discriminator(fake_grad))
            tmp_grad = np.matmul(final_fake_grad.getH(), np.matmul(phi, final_fake_state)) + np.matmul(
                final_fake_state.getH(), np.matmul(phi, final_fake_grad)
            )

            grad_g_phi.append(np.ndarray.item(tmp_grad))
            # grad_g_phi.append(np.asscalar(tmp_grad))

            # for reg term

            term1 = np.matmul(final_fake_grad.getH(), np.matmul(A, final_fake_state)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_real_state)
            )
            term2 = np.matmul(final_fake_state.getH(), np.matmul(A, final_fake_grad)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_real_state)
            )

            term3 = np.matmul(final_fake_grad.getH(), np.matmul(B, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_fake_state)
            )
            term4 = np.matmul(final_fake_state.getH(), np.matmul(B, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_fake_grad)
            )

            term5 = np.matmul(final_fake_grad.getH(), np.matmul(A, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_fake_state)
            )
            term6 = np.matmul(final_fake_state.getH(), np.matmul(A, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_fake_grad)
            )

            term7 = np.matmul(final_fake_grad.getH(), np.matmul(B, final_fake_state)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_real_state)
            )
            term8 = np.matmul(final_fake_state.getH(), np.matmul(B, final_fake_grad)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_real_state)
            )

            tmp_reg_grad = (
                CFG.lamb
                / np.e
                * (
                    CFG.cst1 * (term1 + term2)
                    - CFG.cst2 * (term3 + term4)
                    - CFG.cst2 * (term5 + term6)
                    + CFG.cst3 * (term7 + term8)
                )
            )

            grad_g_reg.append(np.ndarray.item(tmp_reg_grad))
            # grad_g_reg.append(np.asscalar(tmp_reg_grad))

        g_psi = np.asarray(grad_g_psi)
        g_phi = np.asarray(grad_g_phi)
        g_reg = np.asarray(grad_g_reg)

        grad = np.real(g_psi - g_phi - g_reg)

        return grad

    def update_gen(self, dis, total_real_state, total_input_state):
        theta = []
        for gate in self.qc.gates:
            theta.append(gate.angle)

        grad = np.asarray(self._grad_theta(dis, total_real_state, total_input_state))
        theta = np.asarray(theta)
        new_angle = self.optimizer.compute_grad(theta, grad, "min")
        for i in range(self.qc.depth):
            self.qc.gates[i].angle = new_angle[i]

    def load_model_params(self, file_path):
        """
        Load generator parameters (angles) from a saved model, if compatible.
        Supports loading when adding or removing an ancilla (one qubit difference).
        WARNING: Only load trusted pickle files! Untrusted files may be insecure.
        """
        ##################################################################
        # Check if the file exists and is a valid pickle file
        ##################################################################
        if not os.path.exists(file_path):
            print_and_train_log("Generator model file not found", CFG.log_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved_gen = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_train_log(f"Could not load generator model: {e}", CFG.log_path)
            return False

        ##################################################################
        # Check for exact match
        ##################################################################
        if getattr(saved_gen, "size", None) == self.size and len(saved_gen.qc.gates) == len(self.qc.gates):
            for i, gate in enumerate(self.qc.gates):
                gate.angle = saved_gen.qc.gates[i].angle
            print_and_train_log("Generator parameters loaded", CFG.log_path)
            return True

        ##################################################################
        # When adding or removing an ancilla (one qubit difference)
        ###################################################################
        if abs(getattr(saved_gen, "size", None) - self.size) == 1:
            # Determine the minimum number of qubits (the overlap)
            min_size = min(saved_gen.size, self.size)
            for gate in self.qc.gates:
                q1 = getattr(gate, "qubit1", None)
                q2 = getattr(gate, "qubit2", None)
                # Only consider gates that act on the overlapping qubits
                if (q1 is not None and q1 >= min_size) or (q2 is not None and q2 >= min_size):
                    continue
                for saved_gate in saved_gen.qc.gates:
                    if (
                        isinstance(gate, type(saved_gate))
                        and getattr(saved_gate, "qubit1", None) == q1
                        and getattr(saved_gate, "qubit2", None) == q2
                    ):
                        gate.angle = saved_gate.angle
                        break
            print_and_train_log("Generator parameters partially loaded (excluding ancilla difference)", CFG.log_path)
            return True

        print_and_train_log(
            "Saved generator model is incompatible (size or depth mismatch). Skipping load.", CFG.log_path
        )
        return False
