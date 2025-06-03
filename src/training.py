import time
from datetime import datetime

import numpy as np

import config as cf
from cost_functions.cost_and_fidelity import compute_cost, compute_fidelity
from discriminator.discriminator import Discriminator
from generator.generator import Generator
from target.target_state import get_maximally_entangled_state
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

        self.total_input_state = get_maximally_entangled_state(cf.system_size)
        """Preparation of max. entgl. state with ancilla qubit if needed."""

        self.target_unitary = cf.target_unitary
        """Define target gates. First option is to specify the Z, ZZ, ZZZ and/or I terms, second and third is for the respective hardcoded Hamiltonians."""

        # Define the sistem size for the generator and discriminator (the target unitary doesn't have ancilla, it's added later on).
        self.gen_system_size = cf.system_size + (1 if cf.extra_ancilla else 0)
        self.gen = Generator(self.gen_system_size)

        self.gen.set_qcircuit(cf.gen_ansatz(self.gen.qc, self.gen_system_size, cf.layer))
        """Defines the Generator. First option is for XYZ and Z, second option is for ZZ and XZ."""

        self.total_target_state = self.initialize_target_state()
        """Define the size of target state (with ancilla or not, depending on value of `config.extra_ancilla`)."""

        self.dis_system_size = cf.system_size * 2 + (1 if cf.extra_ancilla and cf.ancilla_mode == "pass" else 0)
        self.dis = Discriminator([I, X, Y, Z], self.dis_system_size)
        """Defines the size of Discriminator (with ancilla or not, depending on value of `config.extra_ancilla`)."""

    def initialize_target_state(self):
        """Initialize the target state."""
        target_op = np.kron(Identity(cf.system_size), self.target_unitary)
        if cf.extra_ancilla:
            target_op = np.kron(target_op, Identity(1))
        return np.matmul(target_op, self.total_input_state)

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        # Compute fidelity at initial
        f = compute_fidelity(self.gen, self.total_target_state, self.total_input_state)

        # Data storing
        fidelities, losses = np.zeros(cf.iterations_epoch), np.zeros(cf.iterations_epoch)
        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs = 0

        # Training
        while f < cf.max_fidelity:
            # while (f < 0.95):
            fidelities[:] = 0.0
            losses[:] = 0.0
            num_epochs += 1
            for iter in range(cf.iterations_epoch):
                print("==================================================")
                print("Epoch {}, Iteration {}, Step_size {}".format(num_epochs, iter + 1, cf.eta))

                # Generator gradient descent
                self.gen.update_gen(self.dis, self.total_target_state, self.total_input_state)
                # Discriminator gradient ascent
                for _ in range(cf.ratio_step_disc_to_gen):
                    self.dis.update_dis(self.gen, self.total_target_state, self.total_input_state)

                fidelities[iter] = compute_fidelity(self.gen, self.total_target_state, self.total_input_state)
                losses[iter] = compute_cost(self.gen, self.dis, self.total_target_state, self.total_input_state)

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


if __name__ == "__main__":
    # Run the training process
    # This will execute the training logic defined in the Training class
    # and save the results, models, and logs as specified in the configuration.
    t = Training()
    t.run()
