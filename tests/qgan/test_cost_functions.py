import sys
import os

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pytest
import numpy as np
from qgan.cost_functions import compute_cost, compute_fidelity, get_final_comp_states_for_dis
from qgan.generator import Generator
from qgan.discriminator import Discriminator
from qgan.ancilla import get_max_entangled_state_with_ancilla_if_needed
from qgan.target import get_total_target_state
from config import CFG

@pytest.fixture
def final_states_for_discriminator():
    """Get the final target state for the discriminator, considering ancilla mode."""
    final_target_and_gen_states = []
    # Run multiple times to ensure stability (randomness in Gen thetas)
    for ancilla in [False, True]:
        CFG.extra_ancilla = ancilla
        for ancilla_mode in ["pass", "project", "trace"]:
            CFG.ancilla_mode = ancilla_mode
            # Compute input and target states:
            total_input_state: np.matrix = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            total_target_state: np.matrix = get_total_target_state(total_input_state)
            
            # Also try different ansatz and number of layers for gen:
            for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
                CFG.gen_ansatz = ansatz
                for num_layers in [1, 3]:
                    CFG.gen_layers = num_layers

                    gen = Generator()
                    total_gen_state: np.matrix = gen.get_total_gen_state(total_input_state)
                    final_target_and_gen_states.append((ancilla, ancilla_mode, get_final_comp_states_for_dis(total_target_state, total_gen_state)))
            
    return final_target_and_gen_states

@pytest.fixture
def gen_and_total_states_for_discriminator():
    """Get the final target state for the discriminator, considering ancilla mode."""
    gen_and_total_states_for_discriminator = []
    # Run multiple times to ensure stability (randomness in Gen thetas)
    for ancilla in [False, True]:
        CFG.extra_ancilla = ancilla
        for ancilla_mode in ["pass", "project", "trace"]:
            CFG.ancilla_mode = ancilla_mode
            # Compute input and target states:
            total_input_state: np.matrix = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            total_target_state: np.matrix = get_total_target_state(total_input_state)
            
            # Also try different ansatz and number of layers for gen:
            for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
                CFG.gen_ansatz = ansatz
                for num_layers in [1, 3]:
                    CFG.gen_layers = num_layers
                    gen = Generator()
                    total_gen_state: np.matrix = gen.get_total_gen_state(total_input_state)
                    gen_and_total_states_for_discriminator.append((ancilla, ancilla_mode, gen, total_target_state, total_gen_state))
            
    return gen_and_total_states_for_discriminator
        
class TestCostFunctions():
    
    def test_compute_fidelity(self, final_states_for_discriminator):
        for final_states in final_states_for_discriminator:
            _, _, final_states = final_states
            result = compute_fidelity(*final_states)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
        
    def test_compute_fidelity_equals_1_for_same_state(self, final_states_for_discriminator):
        for final_states in final_states_for_discriminator:
            _, _, final_states = final_states
            final_target_state, final_gen_state = final_states
            result_1 = compute_fidelity(final_target_state, final_target_state)
            result_2 = compute_fidelity(final_gen_state, final_gen_state)
            assert np.isclose(result_1, 1.0)
            assert np.isclose(result_2, 1.0)
    
   
    def test_compute_cost_with_gradient(self, gen_and_total_states_for_discriminator):
        for _ in range(5):  # Run multiple times to ensure stability (randomness in Discriminator params)
            for gen_and_total_states in gen_and_total_states_for_discriminator: # Run multiple times to ensure stability (random in Gen)
                ancilla, ancilla_mode, gen, final_target_state, final_gen_state = gen_and_total_states
                CFG.extra_ancilla = ancilla
                CFG.ancilla_mode = ancilla_mode
                dis = Discriminator()
                
                # Gradient of Discriminator should decrease the cost:
                result_1 = compute_cost(dis, final_target_state, final_target_state)
                dis.update_dis(final_target_state, final_gen_state)
                result_2 = compute_cost(dis, final_target_state, final_target_state)
                dis.update_dis(final_target_state, final_gen_state)
                result_3 = compute_cost(dis, final_target_state, final_target_state)
                assert isinstance(result_1, float) and isinstance(result_2, float) and isinstance(result_3, float)
        
    def test_compute_cost_is_smaller_for_similar_states(self, gen_and_total_states_for_discriminator):
        for _ in range(5):  # Run multiple times to ensure stability (randomness in Discriminator params)
            for gen_and_total_states in gen_and_total_states_for_discriminator: # Run multiple times to ensure stability (random in Gen)
                ancilla, ancilla_mode, gen, final_target_state, final_gen_state = gen_and_total_states
                CFG.extra_ancilla = ancilla
                CFG.ancilla_mode = ancilla_mode
                dis = Discriminator()
                dis.update_dis(final_target_state, final_gen_state)
                small_result_1 = compute_cost(dis, final_target_state, final_target_state)
                small_result_2 = compute_cost(dis, final_gen_state, final_gen_state)
                big_result = compute_cost(dis, final_target_state, final_gen_state)
                
                # Cost should be float non-negative
                assert isinstance(small_result_1, float) and isinstance(small_result_2, float) and isinstance(big_result, float)
            
                # Small costs should be smaller than the big cost
                assert small_result_1 < big_result
                assert small_result_2 < big_result


    def test_compute_cost_and_fidelity(self, final_states_for_discriminator):
        for _ in range(5):  # Run multiple times to ensure stability (randomness in Discriminator params)
            for final_states in final_states_for_discriminator: # Run multiple times to ensure stability (random in Gen)
                _, _, final_states = final_states
                dis = Discriminator()
                cost = compute_cost(dis, *final_states)
                fidelity = compute_fidelity(*final_states)
                assert isinstance(cost, float)
                assert isinstance(fidelity, float)
                assert 0.0 <= fidelity <= 1.0
