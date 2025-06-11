import unittest
from qgan.target.target_hamiltonian import get_target_unitary
from qgan.target.target_state import get_maximally_entangled_state, initialize_target_state
from config import CFG

class TestTarget(unittest.TestCase):
    def test_get_target_unitary(self):
        unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
        self.assertIsNotNone(unitary)
        self.assertEqual(unitary.shape[0], unitary.shape[1])

    def test_get_maximally_entangled_state(self):
        state = get_maximally_entangled_state(CFG.system_size)
        self.assertIsNotNone(state)
        self.assertEqual(len(state.shape), 2)

    def test_initialize_target_state(self):
        for ancilla in [False, True]:
            CFG.extra_ancilla = ancilla
            unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
            input_state = get_maximally_entangled_state(CFG.system_size)
            state = initialize_target_state(input_state)
            self.assertIsNotNone(state)
            self.assertEqual(state.shape[0], 2**(unitary.shape[0]+(1 if CFG.extra_ancilla else 0)))
