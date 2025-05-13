import time
from datetime import datetime

import numpy as np
import scipy.io as scio

import config as cf
from cost_functions.cost_and_fidelity import compute_cost, compute_fidelity
from discriminator.discriminator import Discriminator
from generator.ansatz import construct_qcircuit_XX_YY_ZZ_Z, construct_qcircuit_ZZ_X_Z
from generator.generator import Generator
from target.target_hamiltonian import construct_clusterH, construct_RotatedSurfaceCode, construct_target
from target.target_state import get_maximally_entangled_state, get_maximally_entangled_state_in_subspace
from tools.data_managers import (
    save_fidelity_loss,
    save_model,
    save_theta,
    train_log,
)
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qgates import I, Identity, X, Y, Z

np.random.seed()


class Training:
    def __init__(self):
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""

        ################################################################
        # START OF FUNCTIONS TO CHANGE:
        ################################################################

        self.input_state = get_maximally_entangled_state(cf.system_size)
        # self.input_state = get_maximally_entangled_state_in_subspace(cf.system_size)
        """Preparation of max. entgl. state. First option is for full system, Second for subspace."""

        # self.target_unitary = scio.loadmat('./exp_ideal_{}_qubit.mat'.format(cf.system_size))['exp_ideal']
        # self.target_unitary = construct_target(cf.system_size, ZZZ=True)
        self.target_unitary = construct_clusterH(cf.system_size)
        # self.target_unitary = construct_RotatedSurfaceCode(cf.system_size)
        """Define target gates. First option is to specify the Z, ZZ, ZZZ and/or I terms, second and third is for the respective hardcoded Hamiltonians."""

        self.gen = Generator(cf.system_size)
        self.gen.set_qcircuit(construct_qcircuit_XX_YY_ZZ_Z(self.gen.qc, cf.system_size, cf.layer))
        # self.gen.set_qcircuit(construct_qcircuit_ZZ_XZ(self.gen.qc, cf.system_size, cf.layer))
        """Defines the Generator. First option is for XYZ and Z, second option is for ZZ and XZ."""

        ################################################################
        # END OF FUNCTIONS TO CHANGE:
        ################################################################

        self.real_state = self.initialize_target_state()
        """Define the size of target state (with ancilla or not, depending on value of `config.extra_ancilla`)."""

        self.dis = Discriminator([I, X, Y, Z], (cf.system_size + (1 if cf.extra_ancilla else 0)) * 2)
        """Defines the size of Discriminator (with ancilla or not, depending on value of `config.extra_ancilla`)."""

    def initialize_target_state(self):
        """Initialize the target state."""
        if cf.extra_ancilla:
            return np.matmul(
                np.kron(np.kron(np.kron(self.target_unitary, Identity(1)), Identity(cf.system_size)), Identity(1)),
                self.input_state,
            )

        return np.matmul(np.kron(self.target_unitary, Identity(cf.system_size)), self.input_state)

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        # Compute fidelity at initial
        f = compute_fidelity(self.gen, self.input_state, self.real_state)

        # Data storing
        fidelities, losses = np.zeros(cf.iterations_epoch), np.zeros(cf.iterations_epoch)
        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs = 0

        # Training
        while f < 0.99:
            # while (f < 0.95):
            fidelities[:] = 0.0
            losses[:] = 0.0
            num_epochs += 1
            for iter in range(cf.iterations_epoch):
                print("==================================================")
                print("Epoch {}, Iteration {}, Step_size {}".format(num_epochs, iter + 1, cf.eta))

                # Generator gradient descent
                self.gen.update_gen(self.dis, self.real_state, self.input_state)
                # print("Loss after generator step: {}".format(compute_cost(gen, dis, real_state, input_state)))

                # Discriminator gradient ascent
                self.dis.update_dis(self.gen, self.real_state, self.input_state)
                # print("Loss after discriminator step: {}".format(compute_cost(gen, dis, real_state, input_state)))

                fidelities[iter] = compute_fidelity(self.gen, self.input_state, self.real_state)
                losses[iter] = compute_cost(self.gen, self.dis, self.real_state, self.input_state)

                print("Fidelity between real and fake state: {}".format(fidelities[iter]))
                print("==================================================")

                if iter % 10 == 0:
                    # save log
                    endtime = datetime.now()
                    training_duration = (endtime - starttime).seconds / float(3600)
                    param = "epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n".format(
                        iter,
                        round(fidelities[iter], 6),
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        round(training_duration, 2),
                    )
                    train_log(param, cf.log_path)

            f = fidelities[-1]
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            plt_fidelity_vs_iter(fidelities_history, losses_history, cf, num_epochs)

            if num_epochs >= cf.epochs:
                print(f"The number of epochs exceeds {cf.epochs}.")
                break

        # Save data of fidelity and loss
        save_fidelity_loss(fidelities_history, losses_history, cf.fid_loss_path)

        # Save data of the generator and the discriminator
        save_model(self.gen, cf.model_gen_path)
        save_model(self.dis, cf.model_dis_path)

        # Output the parameters of the generator
        save_theta(self.gen, cf.theta_path)

        endtime = datetime.now()
        print("{} seconds".format((endtime - starttime).seconds))
        print("end")


##### Run training:

t = Training()
t.run()
