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
from qgan.ancilla import (
    get_final_gen_state_for_discriminator,
    get_max_entangled_state_with_ancilla_if_needed,
)
from qgan.cost_functions import compute_fidelity_and_cost
from qgan.discriminator import Discriminator
from qgan.generator import Generator
from qgan.target import get_final_target_state
from tools.data.data_managers import (
    print_and_log,
    save_fidelity_loss,
    save_gen_final_params,
    save_model,
)
from tools.data.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()


class Training:
    def __init__(self, config=CFG, verbose=True, seed=None):
        self.config = config
        self.seed = seed
        self.verbose = verbose
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""

        initial_state_total, initial_state_final = get_max_entangled_state_with_ancilla_if_needed(self.config.system_size, self.config)
        """Preparation of max. entgl. state with ancilla qubit if needed, to generate state."""

        self.final_target_state: np.matrix = get_final_target_state(initial_state_final, self.config)
        """Prepare the target state to compare in the Dis, with the size and Target unitary defined in config."""

        self.gen: Generator = Generator(initial_state_total, config=self.config, seed=self.seed)
        """Prepares the Generator with the size, ansatz, layers and ancilla, defined in config."""

        self.dis: Discriminator = Discriminator(config=self.config, seed=self.seed)
        """Prepares the Discriminatos, with the size, and ancilla defined in config."""

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        ###########################################################
        # Initialize training
        ###########################################################
        if self.verbose:
            print_and_log("\n" + self.config.show_data(), self.config.log_path)

        # Load models if specified (only the params, and only if compatible)
        load_models_if_specified(self, verbose=self.verbose)

        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs: int = 0

        ###########################################################
        # Main Training Block
        ###########################################################
        while True:
            # while (f < 0.95):
            fidelities = []
            losses = []
            num_epochs += 1
            for epoch_iter in range(self.config.iterations_epoch):
                ###########################################################
                # Gen and Dis gradient descent
                ###########################################################
                for _ in range(self.config.steps_gen):
                    self.gen.update_gen(self.dis, self.final_target_state)

                # Remove ancilla if needed, with ancilla mode, before discriminator:
                final_gen_state = get_final_gen_state_for_discriminator(self.gen.total_gen_state)

                for _ in range(self.config.steps_dis):
                    self.dis.update_dis(self.final_target_state, final_gen_state)

                ###########################################################
                # Every X iterations: compute and save fidelity & loss
                ###########################################################
                if epoch_iter % self.config.save_fid_and_loss_every_x_iter == 0:
                    fid, loss = compute_fidelity_and_cost(self.dis, self.final_target_state, final_gen_state)
                    fidelities.append(fid), losses.append(loss)

                ############################################################
                # Every X iterations: Print and log fidelity and loss
                ############################################################
                if epoch_iter % self.config.log_every_x_iter == 0:
                    info = "\nepoch:{:4d} | iters:{:4d} | fidelity:{:8f} | loss:{:8f}".format(
                        num_epochs, epoch_iter + 1, round(fid, 6), round(loss, 6)
                    )
                    if self.verbose:
                        print_and_log(info, self.config.log_path)

            ###########################################################
            # End of epoch, store data and plot
            ###########################################################
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            #plt_fidelity_vs_iter(fidelities_history, losses_history, CFG, num_epochs)

            #############################################################
            # Stopping conditions
            #############################################################
            if num_epochs >= self.config.epochs:
                if self.verbose:
                    print_and_log("\n==================================================\n", self.config.log_path)
                    print_and_log(f"\nThe number of epochs exceeds {self.config.epochs}.", self.config.log_path)
                break

            if fidelities[-1] > self.config.max_fidelity:  # TODO: Maybe change this cond, to use max(fidelities)?
                if self.verbose:
                    print_and_log("\n==================================================\n", self.config.log_path)
                    print_and_log(
                        f"\nThe fidelity {fidelities[-1]} exceeds the maximum {self.config.max_fidelity}.",
                        self.config.log_path,
                    )
                break

        ###########################################################
        # End training, save all data into files
        ###########################################################
        # Save data of fidelity and loss
        save_fidelity_loss(fidelities_history, losses_history, self.config.fid_loss_path)

        # Save data of the generator and the discriminator
        save_model(self.gen, self.config.model_gen_path)
        save_model(self.dis, self.config.model_dis_path)

        # Output the parameters of the generator
        save_gen_final_params(self.gen, self.config.gen_final_params_path)

        endtime = datetime.now()
        if self.verbose:
            print_and_log(f"\nRun took: {endtime - starttime} time.", self.config.log_path)

        return losses_history, fidelities_history