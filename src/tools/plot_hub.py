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
import re

import matplotlib as mpl
import numpy as np

from tools.data.data_managers import print_and_log

mpl.use("Agg")
import matplotlib.pyplot as plt


########################################################################
# MAIN PLOTTING FUNCTION
########################################################################
def generate_all_plots(base_path, log_path, n_runs, max_fidelity, folder_mode):
    # Plot for each run
    for run_idx in range(1, n_runs + 1):
        plot_recurrence_vs_fid(base_path, log_path, run_idx=run_idx, max_fidelity=max_fidelity, folder_mode=folder_mode)

    # Plot all runs together (overwrites each time)
    plot_comparison_all_runs(base_path, log_path, n_runs=n_runs, max_fidelity=max_fidelity, folder_mode=folder_mode)

    # Plot average best fidelity per run
    plot_avg_best_fid_per_run(base_path, log_path, n_runs=n_runs, max_fidelity=max_fidelity, folder_mode=folder_mode)

    # Plot percent of runs above max_fidelity per run
    plot_success_percent_per_run(base_path, log_path, n_runs=n_runs, max_fidelity=max_fidelity, folder_mode=folder_mode)


########################################################################
# REAL TIME RUN PLOTTING FUNCTION
########################################################################
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
    plt.close()


#########################################################################
# PLOT INDIVIDUAL RUNS HISTOGRAMS
#########################################################################
def plot_recurrence_vs_fid(base_path, log_path, run_idx=None, max_fidelity=0.99, folder_mode="initial"):
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_controls", r"\d+") if folder_mode == "initial" else []
    )
    changed_fids = collect_latest_changed_fidelities_nested(base_path, folder_mode, run_idx)
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    control_hist, _ = np.histogram(control_fids, bins=bins) if control_fids else (np.zeros(len(bins) - 1), bins)
    changed_hist, _ = np.histogram(changed_fids, bins=bins)
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(8, 6))
    width = (bins[1] - bins[0]) * 0.4
    bars = []
    if folder_mode == "initial" and np.any(control_hist):
        bars.append(
            plt.bar(
                bin_centers - width / 2, control_hist, width=width, label="Control (no change)", alpha=0.7, color="C0"
            )
        )
    if np.any(changed_hist):
        bars.append(
            plt.bar(
                bin_centers + width / 2,
                changed_hist,
                width=width,
                label=f"Run {run_idx}" if run_idx else "Experiment Runs",
                alpha=0.7,
                color="C1",
            )
        )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Recurrence (Count)")
    title = "Recurrence vs Maximum Fidelity"
    if run_idx:
        title += f" (run {run_idx})"
    elif folder_mode == "experiment":
        title += " (Experiment Mode)"
    plt.title(title)
    if bars:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(
        base_path, f"comparison_recurrence_vs_fidelity_run{run_idx}.png" if run_idx else "recurrence_vs_fidelity.png"
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


###########################################################################
# PLOT COMPARISON OF ALL HISTOGRAMS TOGETHER
###########################################################################
def plot_comparison_all_runs(base_path, log_path, n_runs, max_fidelity, folder_mode="initial"):
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_controls", r"\d+") if folder_mode == "initial" else []
    )
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    control_hist, _ = np.histogram(control_fids, bins=bins) if control_fids else (np.zeros(len(bins) - 1), bins)
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(10, 7))
    width = (bins[1] - bins[0]) * 0.7 / (n_runs + (1 if folder_mode == "initial" and np.any(control_hist) else 0))
    bars = []
    if folder_mode == "initial" and np.any(control_hist):
        bars.append(
            plt.bar(
                bin_centers - width * (n_runs // 2), control_hist, width=width, label="Control (no change)", alpha=0.7
            )
        )
    colors = plt.cm.tab10.colors
    for run_idx in range(1, n_runs + 1):
        if folder_mode == "initial":
            changed_fids = collect_latest_changed_fidelities_nested_run(base_path, run_idx)
        else:
            changed_fids = collect_latest_changed_fidelities_nested(base_path, folder_mode, run_idx)
        changed_hist, _ = np.histogram(changed_fids, bins=bins)
        if np.any(changed_hist):
            bars.append(
                plt.bar(
                    bin_centers
                    + width
                    * (run_idx - (n_runs + (1 if folder_mode == "initial" and np.any(control_hist) else 0)) // 2),
                    changed_hist,
                    width=width,
                    label=f"Run {run_idx}",
                    alpha=0.7,
                    color=colors[(run_idx - 1) % len(colors)],
                )
            )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Recurrence (Count)")
    title = "Comparison: Recurrence vs Maximum Fidelity (All Runs)"
    if folder_mode == "experiment":
        title += " (Experiment Mode)"
    plt.title(title)
    if bars:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(base_path, "comparison_recurrence_vs_fidelity_all.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


##########################################################################
# PLOT AVERAGE BEST FIDELITY PER RUN
##########################################################################
def plot_avg_best_fid_per_run(base_path, log_path, n_runs, max_fidelity=0.99, folder_mode="initial"):
    import matplotlib.ticker as mticker

    avgs = []
    for run_idx in range(1, n_runs + 1):
        if folder_mode == "initial":
            changed_fids = collect_latest_changed_fidelities_nested_run(base_path, run_idx)
        else:
            changed_fids = collect_latest_changed_fidelities_nested(base_path, folder_mode, run_idx)
        if changed_fids:
            avgs.append(np.nanmean(changed_fids))
        else:
            avgs.append(0)
    plt.figure(figsize=(8, 5))
    x = np.arange(1, n_runs + 1)
    plt.plot(x, avgs, "o", color="green", label="Avg Best Fidelity", markersize=6)
    plt.axhline(max_fidelity, color="C0", linestyle="--", label=f"threshold_fidelity={max_fidelity}")
    plt.xlabel("Run index")
    plt.ylabel("Average of Best Fidelity Achieved")
    plt.title("Average Best Fidelity per Run")
    plt.ylim(0, 1.05)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    save_path = os.path.join(base_path, "avg_best_fidelity_per_run.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


##########################################################################
# PLOT SUCCESS PERCENTAGE PER RUN (> threshold fidelity)
##########################################################################
def plot_success_percent_per_run(base_path, log_path, n_runs, max_fidelity=0.99, folder_mode="initial"):
    import matplotlib.ticker as mticker

    percents = []
    for run_idx in range(1, n_runs + 1):
        changed_fids = collect_latest_changed_fidelities_nested(base_path, folder_mode, run_idx)
        perc = 100 * np.sum(np.array(changed_fids) >= max_fidelity) / len(changed_fids) if changed_fids else 0
        percents.append(perc)
    plt.figure(figsize=(8, 5))
    x = np.arange(1, n_runs + 1)
    plt.plot(x, percents, "o", color="red", label="Success %", markersize=6)
    plt.xlabel("Run index")
    plt.ylabel(f"% of Runs with Fidelity ≥ {max_fidelity}")
    plt.title("Success Rate per Run")
    plt.ylim(0, 105)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_path = os.path.join(base_path, "success_percent_per_run.png")
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


##########################################################################
# HELPER FUNCTIONS TO COLLECT MAX FIDELITIES
##########################################################################
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


def collect_max_fidelities_nested(base_path, outer_pattern, inner_pattern):
    """
    Collect max fidelities from all outer_pattern/inner_pattern/fidelities/log_fidelity_loss.txt
    """
    max_fids = []
    for root, dirs, files in os.walk(base_path):
        if (
            re.search(outer_pattern, root)
            and re.search(inner_pattern, root)
            and os.path.basename(root) == "fidelities"
            and "log_fidelity_loss.txt" in files
        ):
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested(base_path, folder_mode="initial", run_idx=None):
    """
    Collect max fidelities for changed runs, supporting both folder structures.
    folder_mode: 'initial' (default) or 'experiment'
    If run_idx is not None, only collect for that run.
    """
    run_dirs = {}
    if folder_mode == "initial":
        if run_idx is not None:
            # Only two groups: exp_j and x_num
            pattern = rf"initial_exp_(\d+)/repeated_changed_run{run_idx}/(\d+)/fidelities$"
        else:
            # Three groups: exp_j, run_y, x_num
            pattern = r"initial_exp_(\d+)/repeated_changed_run(\d+)/(\d+)/fidelities$"
    else:
        if run_idx is not None:
            pattern = rf"experiment{run_idx}/(\d+)/fidelities$"
        else:
            pattern = r"experiment(\d+)/(\d+)/fidelities$"
    for root, dirs, files in os.walk(base_path):
        m = re.search(pattern, root)
        if m and "log_fidelity_loss.txt" in files:
            if folder_mode == "initial":
                if run_idx is not None:
                    exp_j = int(m[1])
                    run_y = run_idx
                    x_num = int(m[2])
                else:
                    exp_j = int(m[1])
                    run_y = int(m[2])
                    x_num = int(m[3])
                key = (exp_j, x_num)
            else:
                if run_idx is not None:
                    run_y = run_idx
                    x_num = int(m[1])
                else:
                    run_y = int(m[1])
                    x_num = int(m[2])
                key = (run_y, x_num)
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested_run(base_path, run_idx):
    run_dirs = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(
            r"initial_exp_(\d+)[/\\]repeated_changed_run" + str(run_idx) + r"[/\\](\d+)[/\\]fidelities$", root
        )
        if m and "log_fidelity_loss.txt" in files:
            exp_j = int(m[1])
            x_num = int(m[2])
            key = (exp_j, x_num)
            run_y = run_idx
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids
