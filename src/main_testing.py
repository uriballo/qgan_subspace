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
"""Main module for running the quantum GAN training and testing."""

from config import CFG
from tools.data.data_managers import print_and_train_log
from tools.training_init import run_test_configurations


##############################################################
# TESTING mode
##############################################################
def main():
    # Change results directory to TESTING:
    CFG.base_data_path = f"./generated_data/TESTING/{CFG.run_timestamp}"
    CFG.set_results_paths()

    # Run the test configurations:
    print_and_train_log("Running in TESTING mode.\n", CFG.log_path)
    run_test_configurations()


if __name__ == "__main__":
    main()
