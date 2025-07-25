from config import CFG
from tools.training_init import run_single_training
from time import perf_counter as tpc
import copy

config = copy.deepcopy(CFG)

config.run_multiple_experiments = False
config.system_size = 2
config.extra_ancilla = False
config.epochs = 1
config.iterations_epoch = 100
config.gen_layers = 1
config.gen_ansatz = "XX_YY_ZZ_Z"
config.target_hamiltonian = "custom_h"
config.custom_hamiltonian_terms = ["ZZZ", "ZZ", "XX", "XZ"]
config.label_suffix = "c1_2q_1l_noanc_XXYYZZZ_CustomH_ZZZ"

run_single_training(config=config, verbose=False, seed=42)

#C:\Users\f52ga\CVC\Codes\qgan_subspace\src>scalene --profile-only=qgan profile_qgan.py

#python -m cProfile -o profile.prof profile_qgan.py
#snakeviz profile.prof