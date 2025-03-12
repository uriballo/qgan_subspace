#### Cost and Fidelities file


import numpy as np
from scipy.linalg import expm

from config import cst1, cst2, cst3, lamb

np.random.seed()


def compute_cost(gen, dis, real_state, input_state):
    G = gen.getGen()
    psi = dis.getPsi()
    phi = dis.getPhi()

    fake_state = np.matmul(G, input_state)

    try:
        A = expm(float(-1 / lamb) * phi)
    except Exception:
        print("cost function -1/lamb:\n", (-1 / lamb))
        print("size of phi:\n", phi.shape)

    try:
        B = expm(float(1 / lamb) * psi)
    except Exception:
        print("cost function 1/lamb:\n", (1 / lamb))
        print("size of psi:\n", psi.shape)

    term1 = np.matmul(fake_state.getH(), np.matmul(A, fake_state))
    term2 = np.matmul(real_state.getH(), np.matmul(B, real_state))

    term3 = np.matmul(fake_state.getH(), np.matmul(B, real_state))
    term4 = np.matmul(real_state.getH(), np.matmul(A, fake_state))

    term5 = np.matmul(fake_state.getH(), np.matmul(A, real_state))
    term6 = np.matmul(real_state.getH(), np.matmul(B, fake_state))

    term7 = np.matmul(fake_state.getH(), np.matmul(B, fake_state))
    term8 = np.matmul(real_state.getH(), np.matmul(A, real_state))

    psiterm = np.trace(np.matmul(np.matmul(real_state, real_state.getH()), psi))
    phiterm = np.trace(np.matmul(np.matmul(fake_state, fake_state.getH()), phi))

    regterm = np.ndarray.item(
        lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8)
    )
    # regterm = np.asscalar(
    #     lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8))

    loss = np.real(psiterm - phiterm - regterm)

    return loss


def compute_fidelity(gen, input_state, real_state, type="training"):
    """
        calculate the fidelity between target state and fake state
    :param gen: generator(Generator)
    :param state: vector(array), input state
    :return:
    """
    # if type == 'test':
    #     G = gen.qc.get_mat_rep()
    # else:
    #     G = gen.getGen()

    G = gen.getGen()
    fake_state = np.matmul(G, input_state)

    return np.abs(np.ndarray.item(np.matmul(real_state.getH(), fake_state))) ** 2
    # return np.abs(np.asscalar(np.matmul(real_state.getH(), fake_state))) ** 2
