#### Generator file


import numpy as np
from scipy.linalg import expm

import config as cf
from ancilla.ancilla import get_final_fake_state_for_discriminator, get_final_real_state_for_discriminator
from config import cst1, cst2, cst3, lamb, system_size
from optimizer.momentum_optimizer import MomentumOptimizer
from tools.qcircuit import Identity, QuantumCircuit


class Generator:
    def __init__(self, system_size):
        self.size = system_size
        self.qc = self.init_qcircuit()
        self.optimizer = MomentumOptimizer()

    def set_qcircuit(self, qc):
        self.qc = qc

    def init_qcircuit(self):
        qcircuit = QuantumCircuit(self.size, "generator")
        return qcircuit

    def get_Untouched_qubits_and_Gen(self):
        """Get the matrix representation of the circuit at the Generator step, including the untouched qubits in front."""
        return np.kron(Identity(cf.system_size), self.qc.get_mat_rep())

    def _grad_theta(self, dis, total_real_state, total_input_state):
        Untouched_x_G = self.get_Untouched_qubits_and_Gen()

        phi = dis.getPhi()
        psi = dis.getPsi()

        total_output_state = np.matmul(Untouched_x_G, total_input_state)

        final_fake_state = get_final_fake_state_for_discriminator(total_output_state)
        final_real_state = get_final_real_state_for_discriminator(total_real_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print("grad_gen -1/lamb:\n", (-1 / lamb))
            print("size of phi:\n", phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print("grad_gen 1/lamb:\n", (1 / lamb))
            print("size of psi:\n", psi.shape)

        grad_g_psi, grad_g_phi, grad_g_reg = [], [], []

        for i in range(self.qc.depth):
            grad_i = np.kron(Identity(system_size), self.qc.get_grad_mat_rep(i))
            # for psi term
            grad_g_psi.append(0)

            # for phi term
            fake_grad = np.matmul(grad_i, total_input_state)
            final_fake_grad = np.matrix(get_final_fake_state_for_discriminator(fake_grad))
            tmp_grad = np.matmul(final_fake_grad.getH(), np.matmul(phi, final_fake_state)) + np.matmul(
                final_fake_state.getH(), np.matmul(phi, final_fake_grad)
            )

            grad_g_phi.append(np.ndarray.item(tmp_grad))
            # grad_g_phi.append(np.asscalar(tmp_grad))

            # for reg term

            term1 = np.matmul(final_fake_grad.getH(), np.matmul(A, final_fake_state)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_real_state)
            )
            term2 = np.matmul(final_fake_state.getH(), np.matmul(A, final_fake_grad)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_real_state)
            )

            term3 = np.matmul(final_fake_grad.getH(), np.matmul(B, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_fake_state)
            )
            term4 = np.matmul(final_fake_state.getH(), np.matmul(B, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_fake_grad)
            )

            term5 = np.matmul(final_fake_grad.getH(), np.matmul(A, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_fake_state)
            )
            term6 = np.matmul(final_fake_state.getH(), np.matmul(A, final_real_state)) * np.matmul(
                final_real_state.getH(), np.matmul(B, final_fake_grad)
            )

            term7 = np.matmul(final_fake_grad.getH(), np.matmul(B, final_fake_state)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_real_state)
            )
            term8 = np.matmul(final_fake_state.getH(), np.matmul(B, final_fake_grad)) * np.matmul(
                final_real_state.getH(), np.matmul(A, final_real_state)
            )

            tmp_reg_grad = (
                lamb
                / np.e
                * (cst1 * (term1 + term2) - cst2 * (term3 + term4) - cst2 * (term5 + term6) + cst3 * (term7 + term8))
            )

            grad_g_reg.append(np.ndarray.item(tmp_reg_grad))
            # grad_g_reg.append(np.asscalar(tmp_reg_grad))

        g_psi = np.asarray(grad_g_psi)
        g_phi = np.asarray(grad_g_phi)
        g_reg = np.asarray(grad_g_reg)

        grad = np.real(g_psi - g_phi - g_reg)

        return grad

    def update_gen(self, dis, total_real_state, total_input_state):
        theta = []
        for gate in self.qc.gates:
            theta.append(gate.angle)

        grad = np.asarray(self._grad_theta(dis, total_real_state, total_input_state))
        theta = np.asarray(theta)
        new_angle = self.optimizer.compute_grad(theta, grad, "min")
        for i in range(self.qc.depth):
            self.qc.gates[i].angle = new_angle[i]
