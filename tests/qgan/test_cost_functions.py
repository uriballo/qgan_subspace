import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pytest
import numpy as np
from qgan.cost_functions import compute_cost, compute_fidelity, get_final_comp_states_for_dis
from qgan.generator import Generator
from qgan.discriminator import Discriminator
from qgan.ancilla import get_max_entangled_state_with_ancilla_if_needed
from qgan.target import initialize_target_state
from config import CFG

@pytest.fixture
def final_states_for_discriminator():
    """Get the final target state for the discriminator, considering ancilla mode."""
    gen = Generator()
    total_input_state: np.matrix = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
    total_target_state: np.matrix = initialize_target_state(total_input_state)
    
    final_target_state, final_gen_state = get_final_comp_states_for_dis(gen, total_target_state, total_input_state)
    
    return final_target_state, final_gen_state
        
class TestCostFunctions():
    
    def test_compute_fidelity(self, final_states_for_discriminator):
        result = compute_fidelity(*final_states_for_discriminator)
        assert isinstance(result, float)

    def test_compute_cost(self, final_states_for_discriminator):
        dis = Discriminator()
        result = compute_cost(dis, *final_states_for_discriminator)
        assert isinstance(result, float)

    def test_compute_cost_and_fidelity(self, final_states_for_discriminator):
        dis = Discriminator()
        cost = compute_cost(dis, *final_states_for_discriminator)
        fidelity = compute_fidelity(*final_states_for_discriminator)
        assert isinstance(cost, float)
        assert isinstance(fidelity, float)
        assert 0.0 <= fidelity <= 1.0
