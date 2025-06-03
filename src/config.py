"""
config.py: the configuration for hamiltonian simulation task

"""

from datetime import datetime

import numpy as np
import scipy.io as scio

from generator.ansatz import construct_qcircuit_XX_YY_ZZ_Z, construct_qcircuit_ZZ_X_Z
from target.target_hamiltonian import construct_clusterH, construct_RotatedSurfaceCode, construct_target

################################################################
# START OF PARAMETERS TO CHANGE:
################################################################

# Training parameters
epochs = 10  # Number of epochs
iterations_epoch = 100  # Number of iterations per epoch
max_fidelity = 0.99  # Maximum fidelity to reach (stopping criterion)
eta = 0.01  # Initial learning rate
ratio_step_disc_to_gen = 1  # (int) Ratio of step size for discriminator to generator (Dis > Gen)

# System setting
system_size = 3  # Number of qubits (without choi or ancilla): #3 #4 #5 ...
layer = 4  # Number of layers in the Generator: #20 #15 #10 #4 #3 #2 ...

# Ancilla parameters
extra_ancilla = True  # If to include an extra ancilla: #True # False
ancilla_mode = "tracing_out"  # Options for the extra ancilla: "pass", "project", "tracing_out"
# TODO: Make handling of ancilla_mode more efficient, by never creating ancilla in Target.
# TODO: ancilla_topology = "ansatz"  # Options for the ancilla topology: "ansatz", "total"

# GEN ANSATZ (Callable):
gen_ansatz = construct_qcircuit_XX_YY_ZZ_Z  # Callable: construct_qcircuit_XX_YY_ZZ_Z or construct_qcircuit_ZZ_X_Z

# TARGET CIRCUIT (Called):
# target_unitary = scio.loadmat("./exp_ideal_{}_qubit.mat".format(system_size))["exp_ideal"]
target_unitary = construct_target(system_size, ZZZ=True)  # Remember you can chose Z, ZZ and ZZZ
# target_unitary = construct_clusterH(system_size)
# target_unitary = construct_RotatedSurfaceCode(system_size)

# Parameter for costs functions and gradients
lamb = float(10)

################################################################
# END OF PARAMETERS TO CHANGE:
################################################################

# Costs and gradients
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = (s / 2 + 1) ** 2
cst2 = (s / 2) * (s / 2 + 1)
cst3 = (s / 2) ** 2

# Datetime
curr_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

# File settings
figure_path = f"./generated_data/{curr_datetime}/figure"
model_gen_path = f"./generated_data/{curr_datetime}/saved_model/{system_size}qubit_model-gen(hs).mdl"
model_dis_path = f"./generated_data/{curr_datetime}/saved_model/{system_size}qubit_model-dis(hs).mdl"
log_path = f"./generated_data/{curr_datetime}/logs/{system_size}qubit_log.txt"
fid_loss_path = f"./generated_data/{curr_datetime}/fidelities/{system_size}qubit_log_fidelity_loss.npy"
theta_path = f"./generated_data/{curr_datetime}/theta/{system_size}qubit_theta_gen.txt"
