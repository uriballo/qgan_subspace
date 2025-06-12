import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pickle
from qgan.generator import Generator
from config import CFG

class TestGeneratorPartialLoading():
    def __init__(self):
        CFG.system_size = 2
        CFG.gen_layers = 1
        CFG.extra_ancilla = False
        self.gen = Generator()

    def test_partial_angle_loading(self):
        # Save a model with a known angle
        for gate in self.gen.qc.gates:
            gate.angle = 0.123
        path = "test_gen_partial.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.gen, f)
        # Create a new model with an ancilla (one more qubit)
        CFG.extra_ancilla = True
        gen_with_ancilla = Generator()
        # All angles should be different before loading
        assert not all(g.angle == 0.123 for g in gen_with_ancilla.qc.gates)
        # Load
        gen_with_ancilla.load_model_params(path)
        # At least some angles should now be 0.123 (for gates not involving ancilla)
        assert any(g.angle == 0.123 for g in gen_with_ancilla.qc.gates)
        os.remove(path)
