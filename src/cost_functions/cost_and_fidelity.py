#### Cost and Fidelities file


import numpy as np
from scipy.linalg import expm

from config import cst1, cst2, cst3, lamb
from discriminator.discriminator import Discriminator
from generator.generator import Generator

np.random.seed()


def compute_cost(gen: Generator, dis: Discriminator, real_state: np.ndarray, input_state: np.ndarray) -> float:
    """Calculate the cost function

    Args:
        gen (Generator): the generator.
        dis (Discriminator): the discriminator.
        real_state (np.ndarray): the real state.
        input_state (np.ndarray): the input state.

    Returns:
        float: the cost function.
    """
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


def compute_fidelity(gen: Generator, real_state: np.ndarray, input_state: np.ndarray, type: str = "training") -> float:
    """Calculate the fidelity between target state and fake state

    Args:
        gen (Generator): the generator.
        real_state (np.ndarray): the real state.
        input_state (np.ndarray): the input state.
        type (str): the type of the state. Default is 'training'.

    Returns:
        float: the fidelity between the target state and the fake state.
    """
    # if type == 'test':
    #     G = gen.qc.get_mat_rep()
    # else:
    #     G = gen.getGen()

    G = gen.getGen()
    fake_state = np.matmul(G, input_state)

    return np.abs(np.ndarray.item(np.matmul(real_state.getH(), fake_state))) ** 2
    # return np.abs(np.asscalar(np.matmul(real_state.getH(), fake_state))) ** 2
