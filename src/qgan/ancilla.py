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
"""Ancilla post-processing tools."""

import math

import numpy as np

from config import CFG


def get_max_entangled_state_with_ancilla_if_needed(size: int) -> np.ndarray:
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


def project_ancilla_zero(state: np.ndarray, renormalize: bool = True) -> tuple[np.ndarray, float]:
    """Project the last qubit onto |0> and renormalize. Assumes state is a column vector.

    Args:
        state (np.ndarray): The quantum state vector to project.
        renormalize (bool): Whether to renormalize the projected state.

    Returns:
        np.ndarray: The projected state vector, normalized, with the ancilla qubit removed.
        float: The probability of the ancilla being in state |0>.
    """
    state = np.asarray(state).flatten()

    # Remove the ancilla qubit: keep only even indices:
    projected = state[::2]

    # Compute the norm of the projected state:
    norm = np.linalg.norm(projected)

    if math.isnan(norm):
        raise ValueError("Norm is NaN, check the input state.")

    if norm == 0:  # Return the system part (without ancilla) as zeros
        return np.zeros((2 ** (CFG.system_size * 2), 1)), 0.0

    # Renormalize if needed:
    if renormalize:
        if CFG.ancilla_project_norm in ["remove", "penalize"]:
            projected = projected / norm
        elif CFG.ancilla_project_norm == "pass":
            pass
        else:
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")

    return np.asmatrix(projected.reshape(-1, 1)), norm**2


# TODO: Think better what to do with this function... (how to use it)
# Right now, maybe its a cool way to codify if passing the "norm" or not.
def trace_out_ancilla(state: np.ndarray) -> np.ndarray:
    """Trace out the last qubit and return a sampled pure state from the reduced density matrix.

    Args:
        state (np.ndarray): The quantum state vector to trace out the ancilla.

    Returns:
        np.ndarray: The sampled pure state after tracing out the ancilla.
    """
    # state: (2**num_qubits, 1)
    state = np.asarray(state).flatten()
    # Reshape to (2**(n-1), 2) for last qubit
    state = state.reshape(-1, 2)
    # Compute reduced density matrix by tracing out last qubit
    rho_reduced = np.zeros((state.shape[0], state.shape[0]), dtype=complex)
    for i in range(2):
        rho_reduced += np.dot(state[:, i : i + 1], state[:, i : i + 1].conj().T)
    # Sample a pure state from the reduced density matrix
    eigvals, eigvecs = np.linalg.eigh(rho_reduced)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals / np.sum(eigvals)
    idx = np.random.choice(len(eigvals), p=eigvals)
    sampled_state = eigvecs[:, idx]
    return np.asmatrix(sampled_state.reshape(-1, 1))


def get_final_gen_state_for_discriminator(total_output_state: np.ndarray) -> np.ndarray:
    """Modifies the gen state to be passed to the discriminator, according to ancilla_mode.

    Args:
        total_output_state (np.ndarray): The output state from the generator.

    Returns:
        np.ndarray: The final state to be passed to the discriminator.
    """
    norm = None
    total_final_state = total_output_state
    if CFG.extra_ancilla:
        if CFG.ancilla_mode == "pass":
            # Pass ancilla to discriminator (current behavior)
            return total_final_state
        if CFG.ancilla_mode == "project":
            projected, norm = project_ancilla_zero(total_final_state)
            return projected
        if CFG.ancilla_mode == "trace":
            return trace_out_ancilla(total_final_state)
        raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")
    return total_final_state, norm


def get_final_target_state_for_discriminator(total_output_state: np.ndarray) -> np.ndarray:
    """Modifies the target state to be passed to the discriminator, according to ancilla_mode.

    Args:
        total_output_state (np.ndarray): The output state from the target.

    Returns:
        np.ndarray: The final state to be passed to the discriminator.
    """
    total_final_state = total_output_state
    if CFG.extra_ancilla:
        if CFG.ancilla_mode == "pass":
            # Pass ancilla to discriminator (current behavior)
            return total_final_state
        if CFG.ancilla_mode in ["project", "trace"]:
            state, prob = project_ancilla_zero(total_final_state, renormalize=False)
            return state
            # Return only the system part (project ancilla to zero)
            # No need to renorm or trace, as Ancilla is not used in Target, and total state should be T x |0>.
        raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")
    return total_final_state
