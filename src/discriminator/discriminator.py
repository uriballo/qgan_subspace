#### Discriminator file


import numpy as np
from scipy.linalg import expm

from config import cst1, cst2, cst3, lamb
from optimizer.momentum_optimizer import MomentumOptimizer


class Discriminator:
    def __init__(self, herm, system_size):
        self.size = system_size
        self.herm = herm
        self.alpha = np.zeros((self.size, len(self.herm)))
        self.beta = np.zeros((self.size, len(self.herm)))
        self._init_params()
        self.optimizer_psi = MomentumOptimizer()
        self.optimizer_phi = MomentumOptimizer()

    def _init_params(self):
        # Discriminator Parameters

        for i in range(self.size):
            self.alpha[i] = -1 + 2 * np.random.random(len(self.herm))
            self.beta[i] = -1 + 2 * np.random.random(len(self.herm))

    def getPsi(self) -> np.ndarray:
        """Get matrix representation of real part of discriminator

        Parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        Returns:
            np.ndarray: Psi, matrix representation of real part of discriminator
        """
        psi = 1
        for i in range(self.size):
            psi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                psi_i += self.alpha[i][j] * self.herm[j]
            psi = np.kron(psi, psi_i)
        return psi

    def getPhi(self) -> np.ndarray:
        """Get matrix representation of fake part of discriminator

        Parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        Returns:
            np.ndarray: Phi, matrix representation of fake part of discriminator
        """
        phi = 1
        for i in range(self.size):
            phi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                phi_i += self.beta[i][j] * self.herm[j]
            phi = np.kron(phi, phi_i)
        return phi

    # Psi gradients
    def _grad_psi(self, type):
        grad_psi = []
        for i in range(self.size):
            grad_psiI = 1
            for j in range(self.size):
                if i == j:
                    grad_psii = self.herm[type]
                else:
                    grad_psii = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        grad_psii += self.alpha[j][k] * self.herm[k]
                grad_psiI = np.kron(grad_psiI, grad_psii)
            grad_psi.append(grad_psiI)
        return grad_psi

    def _grad_alpha(self, gen, real_state, input_state):
        G = gen.getGen()
        psi = self.getPsi()
        phi = self.getPhi()

        fake_state = np.matmul(G, input_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print("grad_alpha -1/lamb:\n", (-1 / lamb))
            print("size of phi:\n", phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print("grad_alpha 1/lamb:\n", (1 / lamb))
            print("size of psi:\n", psi.shape)

        cs = 1 / lamb

        grad_psi_term = np.zeros_like(self.alpha, dtype=complex)
        grad_phi_term = np.zeros_like(self.alpha, dtype=complex)
        grad_reg_term = np.zeros_like(self.alpha, dtype=complex)

        for type in range(len(self.herm)):
            gradpsi = self._grad_psi(type)

            gradpsi_list, gradphi_list, gradreg_list = [], [], []

            for grad_psi in gradpsi:
                gradpsi_list.append(np.ndarray.item(np.matmul(real_state.getH(), np.matmul(grad_psi, real_state))))
                # gradpsi_list.append(np.asscalar(np.matmul(real_state.getH(), np.matmul(grad_psi, real_state))))

                gradphi_list.append(0)

                term1 = cs * np.matmul(fake_state.getH(), np.matmul(A, fake_state)) * np.matmul(real_state.getH(), np.matmul(grad_psi, np.matmul(B, real_state)))
                term2 = cs * np.matmul(fake_state.getH(), np.matmul(grad_psi, np.matmul(B, real_state))) * np.matmul(real_state.getH(), np.matmul(A, fake_state))
                term3 = cs * np.matmul(fake_state.getH(), np.matmul(A, real_state)) * np.matmul(real_state.getH(), np.matmul(grad_psi, np.matmul(B, fake_state)))
                term4 = cs * np.matmul(fake_state.getH(), np.matmul(grad_psi, np.matmul(B, fake_state))) * np.matmul(real_state.getH(), np.matmul(A, real_state))

                gradreg_list.append(
                    np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4))
                )
                # gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)

        return grad

    # Phi gradients
    def _grad_phi(self, type):
        grad_phi = []
        for i in range(self.size):
            grad_phiI = 1
            for j in range(self.size):
                if i == j:
                    grad_phii = self.herm[type]
                else:
                    grad_phii = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        grad_phii += self.beta[j][k] * self.herm[k]
                grad_phiI = np.kron(grad_phiI, grad_phii)
            grad_phi.append(grad_phiI)
        return grad_phi

    def _grad_beta(self, gen, real_state, input_state):
        G = gen.getGen()
        psi = self.getPsi()
        phi = self.getPhi()

        fake_state = np.matmul(G, input_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print("grad_beta -1/lamb:\n", (-1 / lamb))
            print("size of phi:\n", phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print("grad_beta 1/lamb:\n", (1 / lamb))
            print("size of psi:\n", psi.shape)

        cs = -1 / lamb

        grad_psi_term = np.zeros_like(self.beta, dtype=complex)
        grad_phi_term = np.zeros_like(self.beta, dtype=complex)
        grad_reg_term = np.zeros_like(self.beta, dtype=complex)

        for type in range(len(self.herm)):
            gradphi = self._grad_phi(type)

            gradpsi_list, gradphi_list, gradreg_list = [], [], []

            for grad_phi in gradphi:
                gradpsi_list.append(0)

                gradphi_list.append(np.ndarray.item(np.matmul(fake_state.getH(), np.matmul(grad_phi, fake_state))))
                # gradphi_list.append(np.asscalar(np.matmul(fake_state.getH(), np.matmul(grad_phi, fake_state))))

                term1 = cs * np.matmul(fake_state.getH(), np.matmul(grad_phi, np.matmul(A, fake_state))) * np.matmul(real_state.getH(), np.matmul(B, real_state))
                term2 = cs * np.matmul(fake_state.getH(), np.matmul(B, real_state)) * np.matmul(real_state.getH(), np.matmul(grad_phi, np.matmul(A, fake_state)))
                term3 = cs * np.matmul(fake_state.getH(), np.matmul(grad_phi, np.matmul(A, real_state))) * np.matmul(real_state.getH(), np.matmul(B, fake_state))
                term4 = cs * np.matmul(fake_state.getH(), np.matmul(B, fake_state)) * np.matmul(real_state.getH(), np.matmul(grad_phi, np.matmul(A, real_state)))

                gradreg_list.append(np.ndarray.item(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))
                # gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)

        return grad

    def update_dis(self, gen, real_state, input_state):
        grad_alpha = self._grad_alpha(gen, real_state, input_state)
        # update alpha
        new_alpha = self.optimizer_psi.compute_grad(self.alpha, grad_alpha, "max")
        # new_alpha = self.alpha + eta * self._grad_alpha(gen)

        grad_beta = self._grad_beta(gen, real_state, input_state)
        # update beta
        new_beta = self.optimizer_phi.compute_grad(self.beta, grad_beta, "max")
        # new_beta = self.beta + eta * self._grad_beta(gen)

        self.alpha = new_alpha
        self.beta = new_beta
