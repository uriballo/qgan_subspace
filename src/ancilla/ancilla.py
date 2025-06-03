# --- Ancilla post-processing tools ---
import numpy as np

import config as cf

# TODO: Check the next two functions for correctness and efficiency.


def project_ancilla_zero(state, num_qubits):
    """Project the last qubit onto |0> and renormalize. Assumes state is a column vector.
    Args:
        state (np.ndarray): The quantum state vector to project.
        num_qubits (int): The total number of qubits in the system, including the ancilla.

    Returns:
        np.ndarray: The projected state vector, normalized.
        float: The probability of the ancilla being in state |0>.
    """
    # state: (2**num_qubits, 1)
    state = np.asarray(state).flatten()
    dim = 2**num_qubits
    # Projector onto |0> for last qubit
    proj = np.zeros(dim)
    for i in range(dim):
        if (i % 2) == 0:
            proj[i] = 1
    projected = state * proj
    norm = np.linalg.norm(projected)
    if norm == 0:
        return projected.reshape(-1, 1), 0.0
    return (projected / norm).reshape(-1, 1), norm**2


def trace_out_ancilla(state, num_qubits):
    """Trace out the last qubit and return a sampled pure state from the reduced density matrix.

    Args:
        state (np.ndarray): The quantum state vector to trace out the ancilla.
        num_qubits (int): The total number of qubits in the system, including the ancilla.

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
    return sampled_state.reshape(-1, 1)


def get_fake_state_for_discriminator(output_state):
    """Return the fake state to be passed to the discriminator, according to ancilla_mode."""
    fake_state = output_state
    if cf.extra_ancilla:
        n = cf.system_size * 2 + 1  # total qubits (system + ancilla)
        if cf.ancilla_mode == "pass":
            # Pass ancilla to discriminator (current behavior)
            return fake_state
        elif cf.ancilla_mode == "project":
            projected, prob = project_ancilla_zero(fake_state, n)
            return projected
        elif cf.ancilla_mode == "tracing_out":
            traced = trace_out_ancilla(fake_state, n)
            return traced
        else:
            raise ValueError(f"Unknown ancilla_mode: {cf.ancilla_mode}")
    else:
        return fake_state
