import sys
import os

from qgan.target.target_state import get_maximally_entangled_state, initialize_target_state
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import unittest
import numpy as np
from qgan.cost_functions import compute_cost, compute_fidelity, get_final_comp_states_for_dis
from qgan.generator.generator import Generator
from qgan.discriminator import Discriminator
from config import CFG
class TestCostFunctions(unittest.TestCase):
    
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.gen = Generator()
        self.total_input_state: np.matrix = get_maximally_entangled_state(CFG.system_size)
        self.total_target_state: np.matrix = initialize_target_state(self.total_input_state)
        
        self.final_target_state, self.final_gen_state = get_final_comp_states_for_dis(self.gen, self.total_target_state, self.total_input_state)

    
    def test_compute_fidelity(self):
        result = compute_fidelity(self.final_target_state, self.final_gen_state)
        self.assertIsInstance(result, float)

    def test_compute_cost(self):
        dis = Discriminator()
        result = compute_cost(dis, self.final_target_state, self.final_gen_state)
        self.assertIsInstance(result, float)
