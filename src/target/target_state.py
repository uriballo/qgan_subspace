##### target state


import numpy as np


def get_zero_state(size):
    """
        get the zero quantum state |0,...0>
    :param size:
    :return:
    """
    zero_state = np.zeros(2**size)
    zero_state[0] = 1
    zero_state = np.asmatrix(zero_state).T
    return zero_state


def get_maximally_entangled_state(size):
    state = np.zeros(2 ** (2 * size), dtype=complex)
    for i in range(2**size):
        state_i = np.zeros(2**size)
        state_i[i] = 1
        state += np.kron(state_i, state_i)
    state = state / np.sqrt(2**size)
    state = np.asmatrix(state).T
    return state


def get_maximally_entangled_state_in_subspace(size):
    state = np.zeros(2 ** (2 * size + 2), dtype=complex)

    # add one additional qubit to each party
    upspin = np.zeros(2)
    upspin[0] = 1

    for i in range(2**size):
        state_i = np.zeros(2**size)
        state_i[i] = 1
        state_i = np.kron(state_i, upspin)
        state += np.kron(state_i, state_i)
    state = state / np.sqrt(2**size)

    state = np.asmatrix(state).T

    return state


def getreal_denmat(cf, prob_real, input_state):
    real_state_denmat = np.asmatrix(np.zeros((2**cf.system_size, 2**cf.system_size), dtype=complex))
    # linear combination of pure state
    for i in range(cf.num_to_mix):
        real_state_denmat += prob_real[i] * (np.matmul(input_state[i], input_state[i].getH()))
    return real_state_denmat
