##### target hamiltonian


import sys

import numpy as np
from scipy import linalg

from tools.qgates import I, X, Y, Z


def term_XXXX(size: int, qubit1: int, qubit2: int, qubit3: int, qubit4: int):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZZ(size: int, qubit1: int, qubit2: int, qubit3: int, qubit4: int):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZ(size: int, qubit1: int, qubit2: int, qubit3: int):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XZX(size: int, qubit1: int, qubit2: int, qubit3: int):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit3 == i):
            matrix = np.kron(matrix, X)
        elif qubit2 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XX(size: int, qubit1: int, qubit2: int):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZ(size: int, qubit1: int, qubit2: int):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_Z(size: int, qubit1: int):
    matrix = 1
    for i in range(size):
        if qubit1 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def construct_target(size: int, Z: bool = False, ZZ: bool = False, ZZZ: bool = False, I: bool = False) -> np.ndarray:
    """Construct target Hamiltonian. Specify the terms to include, setting them to true.

    Args:
        size (int): the size of the system.
        Z (bool): whether to include Z terms.
        ZZ (bool): whether to include ZZ terms.
        ZZZ (bool): whether to include ZZZ terms.
        I (bool): whether to include I terms.

    Returns:
        np.ndarray: the target Hamiltonian.
    """
    H = np.zeros([2**size, 2**size])
    if Z:
        H += sum(term_Z(size, i) for i in range(size))
    if ZZ:
        H += sum(term_ZZ(size, i, i + 1) for i in range(size - 1))
    if ZZZ:
        H += sum(term_ZZZ(size, i, i + 1, i + 2) for i in range(size - 2))
    if I:
        H = np.identity(2**size)
    return linalg.expm(-1j * H)


def construct_clusterH(size: int) -> np.ndarray:
    """Construct cluster Hamiltonian.

    Args:
        size (int): the size of the system.

    Returns:
        np.ndarray: the cluster Hamiltonian.
    """
    H = np.zeros([2**size, 2**size])
    for i in range(size - 2):
        H += term_XZX(size, i, i + 1, i + 2)
        H += term_Z(size, i)
    H += term_Z(size, size - 2)
    H += term_Z(size, size - 1)
    return linalg.expm(-1j * H)


def construct_RotatedSurfaceCode(size: int) -> np.ndarray:
    """Construct rotated surface code Hamiltonian.

    Args:
        size (int): the size of the system.

    Returns:
        np.ndarray: the rotated surface code Hamiltonian.
    """
    H = np.zeros([2**size, 2**size])

    if size == 4:
        H += -term_XXXX(size, 0, 1, 2, 3)
        H += -term_ZZ(size, 0, 1)
        H += -term_ZZ(size, 2, 3)
    elif size == 9:
        H += -term_XXXX(size, 0, 1, 3, 4)
        H += -term_XXXX(size, 4, 5, 7, 8)
        H += -term_XX(size, 2, 5)
        H += -term_XX(size, 3, 6)
        H += -term_ZZZZ(size, 1, 2, 4, 5)
        H += -term_ZZZZ(size, 3, 4, 6, 7)
        H += -term_ZZ(size, 0, 1)
        H += -term_ZZ(size, 7, 8)
    else:
        sys.exit("system size is not 2*2 or 3*3 either")

    return linalg.expm(-1j * H)
