import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from config import CFG
from qgan.ancilla import get_final_gen_state_for_discriminator, get_final_target_state_for_discriminator, project_ancilla_zero, trace_out_ancilla
import numpy as np
from qgan.target import get_target_unitary, initialize_target_state
from qgan.ancilla import get_max_entangled_state_with_ancilla_if_needed
from tools.qobjects.qgates import Identity

class TestAncilla():
    def test_final_gen_state_project(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "project"
        result = get_final_gen_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        expected = state[::2]
        exp_norm = np.linalg.norm(expected)
        assert (result == expected/exp_norm).all() # Keep even indices

    def test_final_gen_state_pass(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "pass"
        result = get_final_gen_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2 + 1)  # size unchanged
        assert result.shape[1] == 1
        assert (result == state).all() # Unchanged state

    def test_final_gen_state_trace(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "trace"
        result = get_final_gen_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        # Don't know how to compare state easily

    def test_final_target_state_project(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "project"
        result = get_final_target_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        assert (result == state[::2]).all() # Keep even indices (no need to renormalize)

    def test_final_target_state_pass(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "pass"
        result = get_final_target_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2 + 1)  # size unchanged
        assert result.shape[1] == 1
        assert (result == state).all() # Unchanged state

    def test_final_target_state_trace(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "trace"
        result = get_final_target_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        # Don't know how to compare state easily

    ####################################################################################
    # MORE CONCRETE TESTS FOR PROJECTING AND TRACING OUT ANCILLA
    ####################################################################################
    def test_project_ancilla_zero_correct_subspace(self):
        # |00> + |10> = [1,0,1,0] (ancilla is last qubit)
        state = np.array([[1,0,1,0]]).T / np.sqrt(2)
        projected, prob = project_ancilla_zero(state)
        expected = np.array([[1,1]]).T / np.sqrt(2)
        assert np.allclose(projected, expected)
        assert abs(prob - 1.0) < 1e-8

    def test_project_ancilla_zero_zero_norm(self):
        # All zeros
        CFG.system_size = 2
        state = np.zeros((2**(CFG.system_size*2+1),1))
        projected, prob = project_ancilla_zero(state)
        assert np.allclose(projected, np.zeros((2**(CFG.system_size*2),1)))
        assert prob == 0.0

    def test_trace_out_ancilla_pure_state(self):
        # |00> + |11> = [1,0,0,1] (Bell state)
        state = np.array([[1,0,0,1]]).T / np.sqrt(2)
        sampled = trace_out_ancilla(state)
        assert sampled.shape == (2,1)
        assert np.allclose(np.linalg.norm(sampled), 1.0)
    
    ####################################################################################
    # INTEGRATION TESTS FOR ANCILLA LOGIC WITH DISCRIMINATOR AND TARGET
    ####################################################################################
    def test_ancilla_logic_integration_with_Identity_target(self):
        # Check that ancilla logic matches for all modes
        CFG.target_hamiltonian = "custom_h"
        CFG.custom_hamiltonian_terms = ["I"]
        CFG.extra_ancilla = True
        total_input_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        total_target_state = initialize_target_state(total_input_state)
        
        # Expected final state, without ancilla.
        expected_state = np.zeros(2 ** (2 * CFG.system_size), dtype=complex)
        dim_register = 2**CFG.system_size
        for i in range(dim_register):
            expected_state[i * dim_register + i] = 1.0
        expected_state /= np.sqrt(dim_register)

        # Comparison:
        for mode in ['pass', 'project', 'trace']:
            CFG.ancilla_mode = mode
            gen_out = get_final_gen_state_for_discriminator(total_target_state)
            target_out = get_final_target_state_for_discriminator(total_target_state)
            # For 'pass', both should be the same as input
            if mode == 'pass':
                assert np.allclose(gen_out, total_target_state)
                assert np.allclose(target_out, total_target_state)
            else:
                # For 'project' and 'trace', target_out should be system part
                assert np.allclose(gen_out, expected_state)
                assert np.allclose(target_out, expected_state)
                
    def test_ancilla_logic_integration_with_actual_Hamiltonians(self):
        # Check that ancilla logic matches for all modes
        CFG.custom_hamiltonian_terms = ["Z", "ZZ", "ZZZ"]
        CFG.extra_ancilla = True
        total_input_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        
        # Expected final state, without ancilla:
        max_entangled_no_ancilla = np.zeros(2 ** (2 * CFG.system_size), dtype=complex)
        dim_register = 2**CFG.system_size
        for i in range(dim_register):
            max_entangled_no_ancilla[i * dim_register + i] = 1.0
        max_entangled_no_ancilla /= np.sqrt(dim_register)
        
        for ham in ["cluster_h", "custom_h"]:
            CFG.target_hamiltonian = ham
        
            total_target_state = initialize_target_state(total_input_state)
            
            # Expected Target state without ancilla:
            target_unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
            target_op = np.kron(Identity(CFG.system_size), target_unitary)
            expected_state = np.matmul(target_op, max_entangled_no_ancilla)


            # Comparison:
            for mode in ['pass', 'project', 'trace']:
                CFG.ancilla_mode = mode
                gen_out = get_final_gen_state_for_discriminator(total_target_state)
                target_out = get_final_target_state_for_discriminator(total_target_state)
                # For 'pass', both should be the same as input
                if mode == 'pass':
                    assert np.allclose(gen_out, total_target_state)
                    assert np.allclose(target_out, total_target_state)
                else:
                    # For 'project' and 'trace', target_out should be system part
                    assert np.allclose(gen_out, expected_state)
                    assert np.allclose(target_out, expected_state)
                    assert np.linalg.norm(gen_out) == 1.0, "Generator output should be normalized"
                    assert np.linalg.norm(target_out) == 1.0, "Target output should be normalized, without need to explicitly doing so (T x|0>)"