import unittest
import os
import pickle
import numpy as np
from qgan.generator.generator import Generator
from qgan.generator.ansatz import get_ansatz_func
from config import CFG

class TestGeneratorPartialLoading(unittest.TestCase):
    def setUp(self):
        self.cfg = CFG
        self.cfg.system_size = 2
        self.cfg.gen_layers = 1
        self.gen = Generator(self.cfg.system_size)
        self.gen.set_qcircuit(get_ansatz_func(self.cfg.gen_ansatz)(self.gen.qc, self.gen.size, self.cfg.gen_layers))

    def test_partial_angle_loading(self):
        # Save a model with a known angle
        for gate in self.gen.qc.gates:
            gate.angle = 0.123
        path = "test_gen_partial.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.gen, f)
        # Create a new model with an ancilla (one more qubit)
        gen_with_ancilla = Generator(self.cfg.system_size + 1)
        gen_with_ancilla.set_qcircuit(get_ansatz_func(self.cfg.gen_ansatz)(gen_with_ancilla.qc, gen_with_ancilla.size, self.cfg.gen_layers))
        # All angles should be different before loading
        self.assertFalse(all(g.angle == 0.123 for g in gen_with_ancilla.qc.gates))
        # Load
        gen_with_ancilla.load_model_params(path)
        # At least some angles should now be 0.123 (for gates not involving ancilla)
        self.assertTrue(any(g.angle == 0.123 for g in gen_with_ancilla.qc.gates))
        os.remove(path)
