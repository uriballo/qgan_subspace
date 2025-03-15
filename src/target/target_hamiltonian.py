##### target hamiltonian


import sys

import numpy as np
from scipy import linalg

from tools.qgates import I, X, Y, Z


def term_XXXX(size, qubit1, qubit2, qubit3, qubit4):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZZ(size, qubit1, qubit2, qubit3, qubit4):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZ(size, qubit1, qubit2, qubit3):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XZX(size, qubit1, qubit2, qubit3):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit3 == i):
            matrix = np.kron(matrix, X)
        elif qubit2 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XX(size, qubit1, qubit2):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZ(size, qubit1, qubit2):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_Z(size, qubit1):
    matrix = 1
    for i in range(size):
        if qubit1 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def construct_target(size, Z: bool = False, ZZ: bool = False, ZZZ: bool = False, I: bool = False):
    H = np.zeros([2**size, 2**size])

    if Z:
        for i in range(size):
            H += term_Z(size, i)

    if ZZ:
        for i in range(size - 1):
            H += term_ZZ(size, i, i + 1)

    if ZZZ:
        for i in range(size - 2):
            H += term_ZZZ(size, i, i + 1, i + 2)

    if I:
        H = np.identity(2**size)

    return linalg.expm(-1j * H)


def construct_clusterH(size):
    H = np.zeros([2**size, 2**size])
    for i in range(size - 2):
        H += term_XZX(size, i, i + 1, i + 2)
        H += term_Z(size, i)
    H += term_Z(size, size - 2)
    H += term_Z(size, size - 1)
    return linalg.expm(-1j * H)


def construct_RotatedSurfaceCode(size):
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
