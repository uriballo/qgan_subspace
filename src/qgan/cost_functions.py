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
"""Cost and Fidelity Functions"""

import numpy as np

from config import CFG
from qgan.ancilla import get_final_gen_state_for_discriminator, get_final_target_state_for_discriminator

np.random.seed()


def get_final_comp_states_for_dis(total_target_state: np.ndarray, total_gen_state: np.ndarray) -> tuple:
    """Get the final target and gen states for comparison in the discriminator.

    Args:
        total_target_state (np.ndarray): the target state, which is the target state.
        total_gen_state (np.ndarray): the gen state, which is the total gen state.


    Returns:
        tuple[np.ndarray]: the final gen state and target state for the discriminator.
    """
    final_gen_state: np.ndarray = get_final_gen_state_for_discriminator(total_gen_state)
    final_target_state: np.ndarray = get_final_target_state_for_discriminator(total_target_state)
    return final_target_state, final_gen_state


def compute_cost(dis, final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the cost function. Which is basically equivalent to the Wasserstein distance.

    Args:
        dis (Discriminator): the discriminator.
        total_target_state (np.ndarray): the target state.
        total_gen_state (np.ndarray): the gen state.

    Returns:
        float: the cost function.
    """
    A, B, psi, phi = dis.get_dis_matrices_rep()

    # fmt: off
    term1 = np.matmul(final_gen_state.getH(), np.matmul(A, final_gen_state))
    term2 = np.matmul(final_target_state.getH(), np.matmul(B, final_target_state))

    term3 = np.matmul(final_gen_state.getH(), np.matmul(B, final_target_state))
    term4 = np.matmul(final_target_state.getH(), np.matmul(A, final_gen_state))

    term5 = np.matmul(final_gen_state.getH(), np.matmul(A, final_target_state))
    term6 = np.matmul(final_target_state.getH(), np.matmul(B, final_gen_state))

    term7 = np.matmul(final_gen_state.getH(), np.matmul(B, final_gen_state))
    term8 = np.matmul(final_target_state.getH(), np.matmul(A, final_target_state))

    psiterm = np.trace(np.matmul(np.matmul(final_target_state, final_target_state.getH()), psi))
    phiterm = np.trace(np.matmul(np.matmul(final_gen_state, final_gen_state.getH()), phi))

    regterm = np.ndarray.item(CFG.lamb / np.e * (CFG.cst1 * term1 * term2 - CFG.cst2 * term3 * term4 - CFG.cst2 * term5 * term6 + CFG.cst3 * term7 * term8))
    # regterm = np.asscalar(CFG.lamb / np.e * (CFG.cst1 * term1 * term2 - CFG.cst2 * term3 * term4 - CFG.cst2 * term5 * term6 + CFG.cst3 * term7 * term8))
    # fmt: on

    loss = np.real(psiterm - phiterm - regterm)

    return loss


def compute_fidelity(final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the fidelity between target state and gen state

    Args:
        final_target_state (np.ndarray): The final target state of the system.
        final_gen_state (np.ndarray): The final gen state of the system.

    Returns:
        float: the fidelity between the target state and the gen state.
    """
    braket = np.matmul(final_target_state.getH(), final_gen_state)
    return np.abs(np.ndarray.item(braket)) ** 2
    # return np.abs(np.asscalar(np.matmul(target_state.getH(), total_final_state))) ** 2


def compute_fidelity_and_cost(dis, total_target_state: np.ndarray, total_gen_state: np.ndarray) -> tuple[float, float]:
    """Calculate the fidelity and cost function

    Args:
        dis (Discriminator): the discriminator.
        total_target_state (np.ndarray): the target state.
        total_gen_state (np.ndarray): the gen state.

    Returns:
        tuple[float, float]: the fidelity and cost function.
    """
    final_target_state, final_gen_state = get_final_comp_states_for_dis(total_target_state, total_gen_state)

    fidelity = compute_fidelity(final_target_state, final_gen_state)
    cost = compute_cost(dis, final_target_state, final_gen_state)

    return fidelity, cost
