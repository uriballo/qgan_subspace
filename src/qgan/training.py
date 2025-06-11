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
from qgan.cost_functions import compute_fidelity_and_cost
from qgan.discriminator import Discriminator
from qgan.generator import Generator
from qgan.target import get_maximally_entangled_state, initialize_target_state
from tools.data.data_managers import (
    print_and_train_log,
    save_fidelity_loss,
    save_gen_final_params,
    save_model,
)
from tools.data.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()


class Training:
    def __init__(self):
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""

        self.total_input_state: np.matrix = get_maximally_entangled_state(CFG.system_size)
        """Preparation of max. entgl. state with ancilla qubit if needed."""

        self.total_target_state: np.matrix = initialize_target_state(self.total_input_state)
        """Prepare the target state, with the size and Target unitary defined in config."""

        self.gen: Generator = Generator()
        """Prepares the Generator with the size, ansatz, layers and ancilla, defined in config."""

        self.dis: Discriminator = Discriminator()
        """Prepares the Discriminatos, with the size, and ancilla defined in config."""

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        ###########################################################
        # Initialize training
        ###########################################################
        print_and_train_log("\n" + CFG.show_data(), CFG.log_path)

        # Load models if specified (only the params, and only if compatible)
        load_models_if_specified(self)

        # Data storing
        fidelities, losses = np.zeros(CFG.iterations_epoch), np.zeros(CFG.iterations_epoch)
        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs: int = 0

        ###########################################################
        # Main Training Block
        ###########################################################
        while True:
            # while (f < 0.95):
            fidelities[:] = 0.0
            losses[:] = 0.0
            num_epochs += 1
            for epoch_iter in range(CFG.iterations_epoch):
                ###########################################################
                # Generator and Discriminator gradient descent
                ###########################################################
                # 1 step for generator
                self.gen.update_gen(self.dis, self.total_target_state, self.total_input_state)
                # Ratio of steps for discriminator
                for _ in range(CFG.ratio_step_dis_to_gen):
                    self.dis.update_dis(self.gen, self.total_target_state, self.total_input_state)

                ###########################################################
                # Compute fidelity and loss
                ###########################################################
                fidelities[epoch_iter], losses[epoch_iter] = compute_fidelity_and_cost(
                    self.gen, self.dis, self.total_target_state, self.total_input_state
                )

                ###########################################################
                # Every X iterations, log data
                ###########################################################
                if epoch_iter % CFG.log_every_x_iter == 0:
                    info = "\nepoch:{:4d} | iters:{:4d} | fidelity:{:8f} | loss:{:8f}".format(
                        num_epochs, epoch_iter + 1, round(fidelities[epoch_iter], 6), round(losses[epoch_iter], 6)
                    )
                    print_and_train_log(info, CFG.log_path)

            ###########################################################
            # End of epoch, store data and plot
            ###########################################################
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            plt_fidelity_vs_iter(fidelities_history, losses_history, CFG, num_epochs)

            #############################################################
            # Stopping conditions
            #############################################################
            if num_epochs >= CFG.epochs:
                print_and_train_log("\n==================================================\n", CFG.log_path)
                print_and_train_log(f"\nThe number of epochs exceeds {CFG.epochs}.", CFG.log_path)
                break

            if fidelities[-1] > CFG.max_fidelity:  # TODO: Maybe change this cond, to use max(fidelities)?
                print_and_train_log("\n==================================================\n", CFG.log_path)
                print_and_train_log(
                    f"\nThe fidelity {fidelities[-1]} exceeds the maximum {CFG.max_fidelity}.",
                    CFG.log_path,
                )
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
        print_and_train_log("\nRun took: {} time.".format((endtime - starttime)), CFG.log_path)
