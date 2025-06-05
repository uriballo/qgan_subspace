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
from tools.qgates import Identity


def get_zero_state(size: int):
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


def get_maximally_entangled_state(size: int):
    """Get the maximally entangled state for the system size (With Ancilla if needed).

    Args:
        size (int): the size of the system.

    Returns:
        np.ndarray: the maximally entangled state.
    """
    state = np.zeros(2 ** (2 * size + (1 if CFG.extra_ancilla else 0)), dtype=complex)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(dim_register)
    return np.asmatrix(state).T


def initialize_target_state(target_unitary: np.ndarray, total_input_state: np.ndarray) -> np.ndarray:
    """Initialize the target state."""
    target_op = np.kron(Identity(CFG.system_size), target_unitary)
    if CFG.extra_ancilla:
        target_op = np.kron(target_op, Identity(1))
    return np.matmul(target_op, total_input_state)
