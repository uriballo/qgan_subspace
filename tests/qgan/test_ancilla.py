import unittest
from config import CFG
from qgan.ancilla import get_final_gen_state_for_discriminator, get_final_target_state_for_discriminator
import numpy as np

class TestAncilla(unittest.TestCase):
    def test_final_gen_state_project(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "project"
        result = get_final_gen_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 2 ** (CFG.system_size * 2))  # size halved
        self.assertEqual(result.shape[1], 1)
        expected = state[::2]
        exp_norm = np.linalg.norm(expected)
        assert (result == expected/exp_norm).all() # Keep even indices

    def test_final_gen_state_pass(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "pass"
        result = get_final_gen_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 2 ** (CFG.system_size * 2 + 1))  # size unchanged
        self.assertEqual(result.shape[1], 1)
        assert (result == state).all() # Unchanged state

    def test_final_gen_state_trace(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "trace"
        result = get_final_gen_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 2 ** (CFG.system_size * 2))  # size halved
        self.assertEqual(result.shape[1], 1)
        # Don't know how to compare state easily

    def test_final_target_state_project(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "project"
        result = get_final_target_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 2 ** (CFG.system_size * 2))  # size halved
        self.assertEqual(result.shape[1], 1)
        assert (result == state[::2]).all() # Keep even indices (no need to renormalize)

    def test_final_target_state_pass(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "pass"
        result = get_final_target_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 2 ** (CFG.system_size * 2 + 1))  # size unchanged
        self.assertEqual(result.shape[1], 1)
        assert (result == state).all() # Unchanged state

    def test_final_target_state_trace(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "trace"
        result = get_final_target_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 2 ** (CFG.system_size * 2))  # size halved
        self.assertEqual(result.shape[1], 1)
        # Don't know how to compare state easily
