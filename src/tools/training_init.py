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

from config import CFG, test_configurations
from qgan.training import Training
from tools.data.data_managers import print_and_train_log

# Assuming Training class is in src.training (adjust if different)
# from ..training import Training # This relative import might need adjustment based on how you run the script
# For now, let's assume Training will be passed or imported directly in main


##################################################################
# SINGLE RUN mode
##################################################################
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
            f"FAILED: Default configuration run!\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {e!s}\n"
            f"Traceback:\n{tb_str}"
            f"{'=' * 60}\n"
        )
        print_and_train_log(error_msg, CFG.log_path)  # Log to file


###################################################################
# TESTING mode
###################################################################
def run_test_configurations():
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
        valid_keys = dir(CFG)  # Get a list of valid attributes of CFG
        for key, value in config_params.items():
            if key in valid_keys:
                setattr(CFG, key, value)
            else:
                error_msg = f"Invalid configuration key: {key}. This key does not exist in CFG."
                print_and_train_log(error_msg, CFG.log_path)  # Log the error
                raise AttributeError(error_msg)

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
                f"FAILED: Test Configuration {i + 1} ({config_params['label_suffix']})!\n"
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
        final_summary_msg = "\nAll test configurations ran SUCCESSFULLY! No errors detected during these runs.\n"
    else:
        final_summary_msg = "\nSome test configurations FAILED. Please review the logs above and the log file.\n"
    print_and_train_log(final_summary_msg, CFG.log_path)
