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
"""Target State module"""

import numpy as np

from config import CFG
from qgan.target.target_hamiltonian import get_target_unitary
from tools.qobjects.qgates import Identity


def get_zero_state(size: int) -> np.ndarray:
    """Get the zero quantum state |0,...0>

    Args:
        size (int): the size of the system.

    Returns:
        np.ndarray: the zero quantum
    """
    zero_state = np.zeros(2**size)
    zero_state[0] = 1
    zero_state = np.asmatrix(zero_state).T
    return zero_state


def get_maximally_entangled_state(size: int) -> np.ndarray:
    """Get the maximally entangled state for the system size (With Ancilla if needed).

    Args:
        size (int): the size of the system.

    Returns:
        np.ndarray: the maximally entangled state.
    """
    # Generate the maximally entangled state for the system size
    state = np.zeros(2 ** (2 * size), dtype=complex)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(dim_register)
    # Add ancilla qubit at the end, if needed
    if CFG.extra_ancilla:
        state = np.kron(state, np.array([1, 0], dtype=complex))  # Ancilla in |0>
    return np.asmatrix(state).T


def initialize_target_state(total_input_state: np.ndarray) -> np.ndarray:
    """Initialize the target state. Applying the target unitary to the maximally entangled state.

    Args:
        total_input_state (np.ndarray): the input state, which is the maximally entangled state.

    Returns:
        np.ndarray: the target state after applying the target unitary.
    """
    target_unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)

    target_op = np.kron(Identity(CFG.system_size), target_unitary)
    if CFG.extra_ancilla:
        target_op = np.kron(target_op, Identity(1))
    return np.matmul(target_op, total_input_state)
