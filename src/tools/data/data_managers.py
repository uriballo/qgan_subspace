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
import pickle

import numpy as np


def train_log(param, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        file.write(param)


def print_and_train_log(param, file_path):
    print(param)  # Console feedback
    train_log(param, file_path)  # Logging to file


# Not used, changed by specific gen and dis load_model_params methods:
# def load_model(file_path):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     with open(file_path, "rb") as qc:
#         model = pickle.load(qc)
#     return model


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb+") as file:
        pickle.dump(model, file)


def save_fidelity_loss(fidelities_history, losses_history, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        np.savetxt(f, fidelities_history)
        np.savetxt(f, losses_history)


def save_gen_final_params(gen, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    array_angle = np.zeros(len(gen.qc.gates))
    for i, gate_i in enumerate(gen.qc.gates):
        array_angle[i] = gate_i.angle
    np.savetxt(file_path, array_angle)
