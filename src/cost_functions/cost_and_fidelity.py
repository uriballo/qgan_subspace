#### Cost and Fidelities file


import numpy as np
from scipy.linalg import expm

from ancilla.ancilla import get_final_state_for_discriminator
from config import cst1, cst2, cst3, lamb
from discriminator.discriminator import Discriminator
from generator.generator import Generator

np.random.seed()


def compute_cost(
    gen: Generator, dis: Discriminator, total_real_state: np.ndarray, total_input_state: np.ndarray
) -> float:
    """Calculate the cost function

    Args:
        gen (Generator): the generator.
        dis (Discriminator): the discriminator.
        total_real_state (np.ndarray): the real state.
        total_input_state (np.ndarray): the input state.

    Returns:
        float: the cost function.
    """
    Untouched_x_G = gen.get_Untouched_qubits_and_Gen()
    psi = dis.getPsi()
    phi = dis.getPhi()

    total_output_state = np.matmul(Untouched_x_G, total_input_state)

    total_final_state = get_final_state_for_discriminator(total_output_state)

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

    term1 = np.matmul(total_final_state.getH(), np.matmul(A, total_final_state))
    term2 = np.matmul(total_real_state.getH(), np.matmul(B, total_real_state))

    term3 = np.matmul(total_final_state.getH(), np.matmul(B, total_real_state))
    term4 = np.matmul(total_real_state.getH(), np.matmul(A, total_final_state))

    term5 = np.matmul(total_final_state.getH(), np.matmul(A, total_real_state))
    term6 = np.matmul(total_real_state.getH(), np.matmul(B, total_final_state))

    term7 = np.matmul(total_final_state.getH(), np.matmul(B, total_final_state))
    term8 = np.matmul(total_real_state.getH(), np.matmul(A, total_real_state))

    psiterm = np.trace(np.matmul(np.matmul(total_real_state, total_real_state.getH()), psi))
    phiterm = np.trace(np.matmul(np.matmul(total_final_state, total_final_state.getH()), phi))

    regterm = np.ndarray.item(
        lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8)
    )
    # regterm = np.asscalar(
    #     lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8))

    loss = np.real(psiterm - phiterm - regterm)

    return loss


def compute_fidelity(gen: Generator, total_real_state: np.ndarray, total_input_state: np.ndarray) -> float:
    """Calculate the fidelity between target state and fake state

    Args:
        gen (Generator): the generator.
        total_real_state (np.ndarray): the real state.
        total_input_state (np.ndarray): the input state.

    Returns:
        float: the fidelity between the target state and the fake state.
    """
    Untouched_x_G = gen.get_Untouched_qubits_and_Gen()
    total_output_state = np.matmul(Untouched_x_G, total_input_state)

    total_final_state = get_final_state_for_discriminator(total_output_state)

    return np.abs(np.ndarray.item(np.matmul(total_real_state.getH(), total_final_state))) ** 2
    # return np.abs(np.asscalar(np.matmul(real_state.getH(), total_final_state))) ** 2
