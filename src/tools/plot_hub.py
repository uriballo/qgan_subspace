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

"""The plot tool"""

import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def plt_fidelity_vs_iter(fidelities, losses, config, indx=0):
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Epoch")
    axs1.set_ylabel("Fidelity between real and fake states")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Epoch")
    axs2.set_ylabel("Wasserstein Loss")
    plt.tight_layout()

    # Save the figure
    fig_path = "{}/{}qubit_{}_{}.png".format(config.figure_path, config.system_size, config.gen_layers, indx)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)


def plt_fidelity_vs_iter_projection(fidelities, losses, probability_up, config, indx=0):
    # fig, (axs1, axs2) = plt.subplots(1, 2)
    fig = plt.figure()
    axs1 = plt.subplot(121)
    axs2 = plt.subplot(222)
    axs3 = plt.subplot(224)

    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Epoch")
    axs1.set_ylabel("Fidelity between real and fake states")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Epoch")
    axs2.set_ylabel("Wasserstein Loss")
    axs3.plot(range(len(probability_up)), probability_up)
    axs3.set_xlabel("Epoch")
    axs3.set_ylabel("Probability of additonal qubit being up")
    plt.tight_layout()

    # Save the figure
    fig_path = "{}/{}qubit_{}_{}.png".format(config.figure_path, config.system_size, config.gen_layers, indx)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
