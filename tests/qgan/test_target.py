import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from qgan.target import get_target_unitary, initialize_target_state
from qgan.ancilla import get_max_entangled_state_with_ancilla_if_needed
from config import CFG

class TestTarget():
    def test_get_target_unitary(self):
        unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
        assert unitary is not None
        assert unitary.shape[0] == unitary.shape[1]

    def test_get_maximally_entangled_state(self):
        state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        assert state is not None
        assert len(state.shape) == 2

    def test_initialize_target_state(self):
        for ancilla in [False, True]:
            CFG.extra_ancilla = ancilla
            unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
            input_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            state = initialize_target_state(input_state)
            assert state is not None
            assert state.shape[0] == 2**(unitary.shape[0]+(1 if CFG.extra_ancilla else 0))
