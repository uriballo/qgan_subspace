##### target state


import numpy as np

import config as cf


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
    """Get the maximally entangled state for the system size.

    Args:
        size (int): the size of the system.

    Returns:
        np.ndarray: the maximally entangled state.
    """
    state = np.zeros(2 ** (2 * size), dtype=complex)
    for i in range(2**size):
        state_i = np.zeros(2**size)
        state_i[i] = 1
        state += np.kron(state_i, state_i)
    state = state / np.sqrt(2**size)
    state = np.asmatrix(state).T

    if cf.extra_ancilla:
        # If using an ancilla, append a zero state for the ancilla qubit
        ancilla_state = get_zero_state(1)
        state = np.kron(state, ancilla_state)
    return state
