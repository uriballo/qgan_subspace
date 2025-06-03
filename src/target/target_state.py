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
    """Get the maximally entangled state for the system size (With Ancilla if needed).

    Args:
        size (int): the size of the system.

    Returns:
        np.ndarray: the maximally entangled state.
    """
    state = np.zeros(2 ** (2 * size + (1 if cf.extra_ancilla else 0)), dtype=complex)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(dim_register)
    return np.asmatrix(state).T
