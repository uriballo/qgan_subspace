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
"""Discriminator module"""

import os
import pickle

import numpy as np
from scipy.linalg import expm

from ancilla.ancilla import get_final_fake_state_for_discriminator, get_final_real_state_for_discriminator
from config import CFG
from optimizer.momentum_optimizer import MomentumOptimizer
from tools.data_managers import print_and_train_log

cst1, cst2, cst3, lamb = CFG.cst1, CFG.cst2, CFG.cst3, CFG.lamb


class Discriminator:
    def __init__(self, herm, system_size):
        self.size = system_size
        self.herm = herm
        self.alpha = np.zeros((self.size, len(self.herm)))
        self.beta = np.zeros((self.size, len(self.herm)))
        self._init_params()
        self.optimizer_psi = MomentumOptimizer()
        self.optimizer_phi = MomentumOptimizer()

    def _init_params(self):
        # Discriminator Parameters

        for i in range(self.size):
            self.alpha[i] = -1 + 2 * np.random.random(len(self.herm))
            self.beta[i] = -1 + 2 * np.random.random(len(self.herm))

    def getPsi(self) -> np.ndarray:
        """Get matrix representation of real part of discriminator

        Parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        Returns:
            np.ndarray: Psi, matrix representation of real part of discriminator
        """
        psi = 1
        for i in range(self.size):
            psi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                psi_i += self.alpha[i][j] * self.herm[j]
            psi = np.kron(psi, psi_i)
        return psi

    def getPhi(self) -> np.ndarray:
        """Get matrix representation of fake part of discriminator

        Parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        Returns:
            np.ndarray: Phi, matrix representation of fake part of discriminator
        """
        phi = 1
        for i in range(self.size):
            phi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                phi_i += self.beta[i][j] * self.herm[j]
            phi = np.kron(phi, phi_i)
        return phi

    # Psi gradients
    def _grad_psi(self, type):
        grad_psi = []
        for i in range(self.size):
            grad_psiI = 1
            for j in range(self.size):
                if i == j:
                    grad_psii = self.herm[type]
                else:
                    grad_psii = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        grad_psii += self.alpha[j][k] * self.herm[k]
                grad_psiI = np.kron(grad_psiI, grad_psii)
            grad_psi.append(grad_psiI)
        return grad_psi

    def _grad_alpha(self, gen, total_real_state, total_input_state):
        Untouched_x_G = gen.get_Untouched_qubits_and_Gen()

        psi = self.getPsi()
        phi = self.getPhi()

        total_output_state = np.matmul(Untouched_x_G, total_input_state)

        final_fake_state = get_final_fake_state_for_discriminator(total_output_state)
        final_real_state = get_final_real_state_for_discriminator(total_real_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print("grad_alpha -1/lamb:\n", (-1 / lamb))
            print("size of phi:\n", phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print("grad_alpha 1/lamb:\n", (1 / lamb))
            print("size of psi:\n", psi.shape)

        cs = 1 / lamb

        grad_psi_term = np.zeros_like(self.alpha, dtype=complex)
        grad_phi_term = np.zeros_like(self.alpha, dtype=complex)
        grad_reg_term = np.zeros_like(self.alpha, dtype=complex)

        for type in range(len(self.herm)):
            gradpsi = self._grad_psi(type)

            gradpsi_list, gradphi_list, gradreg_list = [], [], []

            for grad_psi in gradpsi:
                gradpsi_list.append(
                    np.ndarray.item(np.matmul(final_real_state.getH(), np.matmul(grad_psi, final_real_state)))
                )
                # gradpsi_list.append(np.asscalar(np.matmul(real_state.getH(), np.matmul(grad_psi, real_state))))

                gradphi_list.append(0)

                term1 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(A, final_fake_state))
                    * np.matmul(final_real_state.getH(), np.matmul(grad_psi, np.matmul(B, final_real_state)))
                )
                term2 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(grad_psi, np.matmul(B, final_real_state)))
                    * np.matmul(final_real_state.getH(), np.matmul(A, final_fake_state))
                )
                term3 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(A, final_real_state))
                    * np.matmul(final_real_state.getH(), np.matmul(grad_psi, np.matmul(B, final_fake_state)))
                )
                term4 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(grad_psi, np.matmul(B, final_fake_state)))
                    * np.matmul(final_real_state.getH(), np.matmul(A, final_real_state))
                )

                gradreg_list.append(
                    np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4))
                )
                # gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)

        return grad

    # Phi gradients
    def _grad_phi(self, type):
        grad_phi = []
        for i in range(self.size):
            grad_phiI = 1
            for j in range(self.size):
                if i == j:
                    grad_phii = self.herm[type]
                else:
                    grad_phii = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        grad_phii += self.beta[j][k] * self.herm[k]
                grad_phiI = np.kron(grad_phiI, grad_phii)
            grad_phi.append(grad_phiI)
        return grad_phi

    def _grad_beta(self, gen, total_real_state, total_input_state):
        Untouched_x_G = gen.get_Untouched_qubits_and_Gen()

        psi = self.getPsi()
        phi = self.getPhi()

        total_output_state = np.matmul(Untouched_x_G, total_input_state)

        final_fake_state = get_final_fake_state_for_discriminator(total_output_state)
        final_real_state = get_final_real_state_for_discriminator(total_real_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print("grad_beta -1/lamb:\n", (-1 / lamb))
            print("size of phi:\n", phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print("grad_beta 1/lamb:\n", (1 / lamb))
            print("size of psi:\n", psi.shape)

        cs = -1 / lamb

        grad_psi_term = np.zeros_like(self.beta, dtype=complex)
        grad_phi_term = np.zeros_like(self.beta, dtype=complex)
        grad_reg_term = np.zeros_like(self.beta, dtype=complex)

        for type in range(len(self.herm)):
            gradphi = self._grad_phi(type)

            gradpsi_list, gradphi_list, gradreg_list = [], [], []

            for grad_phi in gradphi:
                gradpsi_list.append(0)

                gradphi_list.append(
                    np.ndarray.item(np.matmul(final_fake_state.getH(), np.matmul(grad_phi, final_fake_state)))
                )
                # gradphi_list.append(np.asscalar(np.matmul(total_fake_state.getH(), np.matmul(grad_phi, total_fake_state))))

                term1 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(grad_phi, np.matmul(A, final_fake_state)))
                    * np.matmul(final_real_state.getH(), np.matmul(B, final_real_state))
                )
                term2 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(B, final_real_state))
                    * np.matmul(final_real_state.getH(), np.matmul(grad_phi, np.matmul(A, final_fake_state)))
                )
                term3 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(grad_phi, np.matmul(A, final_real_state)))
                    * np.matmul(final_real_state.getH(), np.matmul(B, final_fake_state))
                )
                term4 = (
                    cs
                    * np.matmul(final_fake_state.getH(), np.matmul(B, final_fake_state))
                    * np.matmul(final_real_state.getH(), np.matmul(grad_phi, np.matmul(A, final_real_state)))
                )

                gradreg_list.append(
                    np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4))
                )
                # gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)

        return grad

    def update_dis(self, gen, total_real_state, total_input_state):
        grad_alpha = self._grad_alpha(gen, total_real_state, total_input_state)
        # update alpha
        new_alpha = self.optimizer_psi.compute_grad(self.alpha, grad_alpha, "max")
        # new_alpha = self.alpha + eta * self._grad_alpha(gen)

        grad_beta = self._grad_beta(gen, total_real_state, total_input_state)
        # update beta
        new_beta = self.optimizer_phi.compute_grad(self.beta, grad_beta, "max")
        # new_beta = self.beta + eta * self._grad_beta(gen)

        self.alpha = new_alpha
        self.beta = new_beta

    def load_model_params(self, file_path):
        """
        Load discriminator parameters (alpha, beta) from a saved model, if compatible.
        If the saved model has one less qubit (no ancilla), load only the matching parameters.
        WARNING: Only load trusted pickle files! Untrusted files may be insecure.
        """
        ######################################################################
        # Check if the file exists and is a valid pickle file
        ########################################################################
        if not os.path.exists(file_path):
            print_and_train_log("Discriminator model file not found", file_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved_dis = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_train_log(f"Could not load discriminator model: {e}", file_path)
            return False

        ########################################################################
        # Check for exact match
        ########################################################################
        if getattr(saved_dis, "size", None) == self.size and saved_dis.alpha.shape == self.alpha.shape:
            self.alpha = saved_dis.alpha.copy()
            self.beta = saved_dis.beta.copy()
            print_and_train_log("Discriminator parameters loaded", file_path)
            return True

        # For the discriminator the ancilla qubit doesn't pass through the model, so we can load the model normally always

        print_and_train_log("Saved discriminator model is incompatible (size or shape mismatch). Skipping load.")
        return False
