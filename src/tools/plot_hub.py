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
import numpy as np

from tools.data.data_managers import print_and_log

mpl.use("Agg")
import matplotlib.pyplot as plt


def plt_fidelity_vs_iter(fidelities, losses, config, indx=0):
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Iteration")
    axs1.set_ylabel("Fidelity")
    axs1.set_title("Fidelity <target|gen> vs Iterations")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Iteration")
    axs2.set_ylabel("Loss")
    axs2.set_title("Wasserstein Loss vs Iterations")
    plt.tight_layout()

    # Save the figure
    fig_path = f"{config.figure_path}/{config.system_size}qubit_{config.gen_layers}_{indx}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)


def get_max_fidelity_from_file(fid_loss_path):
    if not os.path.exists(fid_loss_path):
        return None
    try:
        data = np.loadtxt(fid_loss_path)
        if data.ndim == 1:
            fidelities = data
        else:
            fidelities = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
        return np.max(fidelities)
    except Exception:
        return None


def collect_max_fidelities(base_path, pattern):
    max_fids = []
    for root, dirs, files in os.walk(base_path):
        if pattern in root and "log_fidelity_loss.txt" in files:
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                max_fids.append(max_fid)
    return max_fids


def plot_recurrence_vs_fidelity(base_path, save_path=None):
    control_fids = collect_max_fidelities(base_path, "repeated_control_")
    changed_fids = collect_max_fidelities(base_path, "repeated_changed_")

    bins = np.linspace(0, 1, 21)
    control_hist, _ = np.histogram(control_fids, bins=bins)
    changed_hist, _ = np.histogram(changed_fids, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(8, 6))
    width = (bins[1] - bins[0]) * 0.4
    plt.bar(bin_centers - width / 2, control_hist, width=width, label="Control (no change)", alpha=0.7, color="C0")
    plt.bar(
        bin_centers + width / 2, changed_hist, width=width, label="Changed (with config change)", alpha=0.7, color="C1"
    )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Recurrence (Count)")
    plt.title("Recurrence vs Maximum Fidelity")
    plt.legend()
    plt.grid(True)
    if save_path is None:
        save_path = os.path.join(base_path, "recurrence_vs_fidelity.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}")
