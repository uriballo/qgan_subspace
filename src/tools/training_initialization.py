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
"""Training initialization module for running training instances"""

import traceback

from config import CFG
from tools.data_managers import print_and_train_log
from training import Training

# Assuming Training class is in src.training (adjust if different)
# from ..training import Training # This relative import might need adjustment based on how you run the script
# For now, let's assume Training will be passed or imported directly in main


def run_single_training():
    """
    Runs a single training instance (the default case when testing=False).
    """
    try:
        ##############################################################
        # Run single training instance with specified configuration
        ##############################################################
        training_instance = Training()
        training_instance.run()
        success_msg = "\nDefault configuration run COMPLETED SUCCESSFULLY.\n"
        print_and_train_log(success_msg, CFG.log_path)  # Log to file

    except Exception as e:  # noqa: BLE001
        ##############################################################
        # Handle exceptions during the training run
        ##############################################################
        tb_str = traceback.format_exc()
        error_msg = (
            f"\n{'-' * 60}\n"
            f"Default configuration run FAILED!\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {e!s}\n"
            f"Traceback:\n{tb_str}"
            f"{'=' * 60}\n"
        )
        print_and_train_log(error_msg, CFG.log_path)  # Log to file


def run_test_configurations(test_configurations):
    """
    Runs a suite of test configurations.
    """

    all_passed = True
    for i, config_params in enumerate(test_configurations):
        test_header_msg = f"\n{'=' * 60}\nRunning Test Configuration {i + 1}/{len(test_configurations)}: {config_params['label_suffix']}\n{'-' * 60}\n"
        print_and_train_log(test_header_msg, CFG.log_path)  # Also log to file

        ##############################################################
        # Set config for the current test run
        ##############################################################
        CFG.system_size = config_params["system_size"]
        CFG.gen_layers = config_params["gen_layers"]  # Corrected key
        CFG.extra_ancilla = config_params["extra_ancilla"]
        CFG.iterations_epoch = config_params["iterations_epoch"]
        CFG.epochs = config_params["epochs"]
        setattr(CFG, "plot_every_epochs", config_params["epochs"])  # Plot at the end of this short run
        current_run_label = f"run_default_label_{config_params['label_suffix']}"
        setattr(CFG, "current_run_label", current_run_label)
        # Apply other test-specific config overrides from config_params
        if "target_hamiltonian" in config_params:
            CFG.target_hamiltonian = config_params["target_hamiltonian"]
        if "gen_ansatz" in config_params:
            CFG.ansatz_gen = config_params["gen_ansatz"]

        try:
            ##############################################################
            # Run the training instance with the current configuration
            ##############################################################
            training_instance = Training()
            training_instance.run()
            success_msg = f"\n{'-' * 60}\nTest Configuration {i + 1} ({config_params['label_suffix']}) COMPLETED SUCCESSFULLY.\n{'=' * 60}\n"
            print_and_train_log(success_msg, CFG.log_path)

        except Exception as e:  # noqa: BLE001
            ##############################################################
            # Handle exceptions for the current test configuration
            ##############################################################
            all_passed = False
            tb_str = traceback.format_exc()
            error_msg = (
                f"\n{'-' * 60}\n"
                f"Test Configuration {i + 1} ({config_params['label_suffix']}) FAILED!\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {e!s}\n"
                f"Traceback:\n{tb_str}"
                f"{'=' * 60}\n"
            )
            print_and_train_log(error_msg, CFG.log_path)
            # Continue with other test configurations

    ##############################################################
    # Final summary of test configurations
    ##############################################################
    final_summary_msg = ""
    if all_passed:
        final_summary_msg = "\nAll test configurations ran successfully! No errors detected during these runs.\n"
    else:
        final_summary_msg = "\nSome test configurations failed. Please review the logs above and the log file.\n"
    print_and_train_log(final_summary_msg, CFG.log_path)
