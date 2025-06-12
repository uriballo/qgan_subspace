import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


import numpy as np
from qgan.cost_functions import compute_cost, compute_fidelity, get_final_comp_states_for_dis
from qgan.generator import Generator
from qgan.discriminator import Discriminator
from qgan.ancilla import get_max_entangled_state_with_ancilla_if_needed
from qgan.target import initialize_target_state
from config import CFG
class TestCostFunctions():
    
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.gen = Generator()
        self.total_input_state: np.matrix = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        self.total_target_state: np.matrix = initialize_target_state(self.total_input_state)
        
        self.final_target_state, self.final_gen_state = get_final_comp_states_for_dis(self.gen, self.total_target_state, self.total_input_state)

    
    def test_compute_fidelity(self):
        result = compute_fidelity(self.final_target_state, self.final_gen_state)
        assert isinstance(result, float)

    def test_compute_cost(self):
        dis = Discriminator()
        result = compute_cost(dis, self.final_target_state, self.final_gen_state)
        assert isinstance(result, float)

    def test_compute_cost_and_fidelity(self):
        dis = Discriminator()
        cost = compute_cost(dis, self.final_target_state, self.final_gen_state)
        fidelity = compute_fidelity(self.final_target_state, self.final_gen_state)
        assert isinstance(cost, float)
        assert isinstance(fidelity, float)
        assert 0.0 <= fidelity <= 1.0
