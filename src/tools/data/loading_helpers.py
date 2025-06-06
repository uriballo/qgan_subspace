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
import os

from config import CFG
from tools.data.data_managers import print_and_train_log


def load_models_if_specified(training_instance):
    """
    Loads generator and discriminator parameters if a load_timestamp is provided.
    Modifies training_instance.gen and training_instance.dis by calling their load_model_params methods.
    """

    ################################################################
    # Conditions to skip loading models
    ################################################################
    if CFG.testing:
        print_and_train_log("\nSkipping model loading in testing mode. \n", CFG.log_path)
        print_and_train_log("==================================================\n", CFG.log_path)
        return

    if not CFG.load_timestamp:
        print_and_train_log("\nStarting training from scratch (no timestamp specified).\n", CFG.log_path)
        print_and_train_log("==================================================\n", CFG.log_path)

        return

    ################################################################
    # If we reach here, we have a load_timestamp
    ################################################################
    print_and_train_log(f"\nAttempting to load model parameters [{CFG.load_timestamp}].\n", CFG.log_path)

    # Ensure the load_timestamp is valid
    gen_model_filename = os.path.basename(CFG.model_gen_path)
    dis_model_filename = os.path.basename(CFG.model_dis_path)
    # Path structure: "generated_data/<timestamp>/saved_model/<model_filename>"
    load_gen_path = os.path.join("generated_data", CFG.load_timestamp, "saved_model", gen_model_filename)
    load_dis_path = os.path.join("generated_data", CFG.load_timestamp, "saved_model", dis_model_filename)

    ################################################################
    # Attempt to load generator:
    ################################################################
    print_and_train_log(
        f"Attempting to load Generator parameters from: {load_gen_path}\n",
        CFG.log_path,
    )
    gen_loaded = training_instance.gen.load_model_params(load_gen_path)

    ################################################################
    # Attempt to load discriminator:
    ################################################################
    print_and_train_log(
        f"Attempting to load Discriminator parameters from: {load_dis_path}\n",
        CFG.log_path,
    )
    dis_loaded = training_instance.dis.load_model_params(load_dis_path)

    ################################################################
    # Final check: if both models are loaded, continue training
    ################################################################
    if gen_loaded and dis_loaded:
        print_and_train_log("Model parameter loading complete. Continuing training.\n", CFG.log_path)
        print_and_train_log("==================================================\n", CFG.log_path)
    else:
        raise ValueError("Incompatible or missing model parameters. Check the load paths or model compatibility.")
