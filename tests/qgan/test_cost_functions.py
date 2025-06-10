import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import unittest
import numpy as np
from qgan.cost_functions import compute_cost, compute_fidelity
from qgan.generator.generator import Generator
from qgan.discriminator import Discriminator
from config import CFG
class TestCostFunctions(unittest.TestCase):
    def test_compute_fidelity(self):
        gen = Generator()
        total_target_state = np.ones((2**CFG.system_size, 1))
        total_input_state = np.ones((2**CFG.system_size, 1))
        result = compute_fidelity(gen, total_target_state, total_input_state)
        self.assertIsInstance(result, float)

    def test_compute_cost(self):
        gen = Generator()
        dis = Discriminator()
        total_target_state = np.ones((2**CFG.system_size, 1))
        total_input_state = np.ones((2**CFG.system_size, 1))
        result = compute_cost(gen, dis, total_target_state, total_input_state)
        self.assertIsInstance(result, float)

if __name__ == "__main__":
    unittest.main()
