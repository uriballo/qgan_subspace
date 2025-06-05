# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training module for the Quantum GAN"""

from datetime import datetime

import numpy as np

from config import CFG
from cost_functions.cost_and_fidelity import compute_cost, compute_fidelity
from discriminator.discriminator import Discriminator
from generator.ansatz import get_ansatz_func
from generator.generator import Generator
from target.target_hamiltonian import get_target_unitary
from target.target_state import get_maximally_entangled_state, initialize_target_state
from tools.data_managers import (
    print_and_train_log,
    save_fidelity_loss,
    save_gen_final_params,
    save_model,
)

# from tools.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qgates import I, X, Y, Z

np.random.seed()


class Training:
    def __init__(self):
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""

        self.total_input_state = get_maximally_entangled_state(CFG.system_size)
        """Preparation of max. entgl. state with ancilla qubit if needed."""

        self.target_unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
        """Define target gates. First option is to specify the Z, ZZ, ZZZ and/or I terms, second and third is for the respective hardcoded Hamiltonians."""

        self.gen_system_size = CFG.system_size + (1 if CFG.extra_ancilla else 0)
        self.gen = Generator(self.gen_system_size)
        self.gen.set_qcircuit(get_ansatz_func(CFG.gen_ansatz)(self.gen.qc, self.gen_system_size, CFG.gen_layers))
        """Defines the Generator. First option is for XYZ and Z, second option is for ZZ and XZ."""

        self.total_target_state = initialize_target_state(self.target_unitary, self.total_input_state)
        """Define the size of target state (with ancilla or not, depending on value of `config.extra_ancilla`)."""

        self.dis_system_size = CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        self.dis = Discriminator([I, X, Y, Z], self.dis_system_size)
        """Defines the size of Discriminator (with ancilla or not, depending on value of `config.extra_ancilla`)."""

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        ###########################################################
        # Initialize training
        ###########################################################
        print_and_train_log(CFG.show_data(), CFG.log_path)

        # TODO: Make loading models compatible with adding ancilla & parameters changed
        # Load models if specified (only the params)
        # load_models_if_specified(self, CFG)

        # Data storing
        fidelities, losses = np.zeros(CFG.iterations_epoch), np.zeros(CFG.iterations_epoch)
        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs = 0

        ###########################################################
        # Main Training Block
        ###########################################################
        while fidelities[-1] < CFG.max_fidelity:  # TODO: Maybe change this cond, to use max(fidelities)?
            # while (f < 0.95):
            fidelities[:] = 0.0
            losses[:] = 0.0
            num_epochs += 1
            for iter in range(CFG.iterations_epoch):
                ###########################################################
                # Generator and Discriminator gradient descent
                ###########################################################
                # 1 step for generator
                self.gen.update_gen(self.dis, self.total_target_state, self.total_input_state)
                # Ratio of steps for discriminator
                for _ in range(CFG.ratio_step_disc_to_gen):
                    self.dis.update_dis(self.gen, self.total_target_state, self.total_input_state)

                ###########################################################
                # Compute fidelity and loss
                ###########################################################
                fidelities[iter] = compute_fidelity(self.gen, self.total_target_state, self.total_input_state)
                losses[iter] = compute_cost(self.gen, self.dis, self.total_target_state, self.total_input_state)

                ###########################################################
                # Every X iterations, log data
                ###########################################################
                if iter % CFG.log_every_x_iter == 0:
                    # save log
                    param = "epoch:{:4d} | iters:{:4d} | fidelity:{:8f} | loss:{:8f}\n".format(
                        num_epochs,
                        iter + 1,
                        round(fidelities[iter], 6),
                        round(losses[iter], 6),
                    )
                    print_and_train_log(param, CFG.log_path)

            ###########################################################
            # End of epoch, store data and plot
            ###########################################################
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            plt_fidelity_vs_iter(fidelities_history, losses_history, CFG, num_epochs)

            if num_epochs >= CFG.epochs:
                print_and_train_log(f"The number of epochs exceeds {CFG.epochs}.", CFG.log_path)
                break

        ###########################################################
        # End training, save all data into files
        ###########################################################
        # Save data of fidelity and loss
        save_fidelity_loss(fidelities_history, losses_history, CFG.fid_loss_path)

        # Save data of the generator and the discriminator
        save_model(self.gen, CFG.model_gen_path)
        save_model(self.dis, CFG.model_dis_path)

        # Output the parameters of the generator
        save_gen_final_params(self.gen, CFG.gen_final_params_path)

        endtime = datetime.now()
        print_and_train_log("{} seconds".format((endtime - starttime).seconds), CFG.log_path)
        print_and_train_log("end", CFG.log_path)
