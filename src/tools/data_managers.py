import os
import pickle

import numpy as np


def train_log(param, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        file.write(param)


def load_model(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "rb") as qc:
        model = pickle.load(qc)
    return model


def save_model(gen, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb+") as file:
        pickle.dump(gen, file)


def save_fidelity_loss(fidelities_history, losses_history, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        np.save(f, fidelities_history)
        np.save(f, losses_history)


def save_theta(gen, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    array_angle = np.zeros(len(gen.qc.gates))
    for i in range(len(gen.qc.gates)):
        array_angle[i] = gen.qc.gates[i].angle
    np.savetxt(file_path, array_angle)
