import unittest
import os
import pickle
import numpy as np
from src.qgan.generator.generator import Generator
from src.qgan.generator.ansatz import get_ansatz_func
from src.config import CFG

class TestGeneratorAncillaLoading(unittest.TestCase):
    def setUp(self):
        # Create a generator with and without ancilla
        self.cfg_no_ancilla = CFG
        self.cfg_no_ancilla.extra_ancilla = False
        self.cfg_no_ancilla.system_size = 2
        self.cfg_no_ancilla.gen_layers = 1
        self.gen_no_ancilla = Generator(self.cfg_no_ancilla.system_size)
        self.gen_no_ancilla.set_qcircuit(get_ansatz_func(self.cfg_no_ancilla.gen_ansatz)(self.gen_no_ancilla.qc, self.gen_no_ancilla.size, self.cfg_no_ancilla.gen_layers))

        self.cfg_with_ancilla = CFG
        self.cfg_with_ancilla.extra_ancilla = True
        self.cfg_with_ancilla.system_size = 2
        self.cfg_with_ancilla.gen_layers = 1
        self.gen_with_ancilla = Generator(self.cfg_with_ancilla.system_size + 1)
        self.gen_with_ancilla.set_qcircuit(get_ansatz_func(self.cfg_with_ancilla.gen_ansatz)(self.gen_with_ancilla.qc, self.gen_with_ancilla.size, self.cfg_with_ancilla.gen_layers))

    def test_load_from_no_ancilla_to_with_ancilla(self):
        # Save no-ancilla model
        path = "test_gen_no_ancilla.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.gen_no_ancilla, f)
        # Load into with-ancilla model
        result = self.gen_with_ancilla.load_model_params(path)
        self.assertTrue(result)
        os.remove(path)

    def test_load_from_with_ancilla_to_no_ancilla(self):
        # Save with-ancilla model
        path = "test_gen_with_ancilla.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.gen_with_ancilla, f)
        # Load into no-ancilla model
        result = self.gen_no_ancilla.load_model_params(path)
        self.assertTrue(result)
        os.remove(path)

    def test_load_incompatible(self):
        # Change layers to make models incompatible
        gen_other = Generator(3)
        path = "test_gen_incompatible.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen_other, f)
        result = self.gen_no_ancilla.load_model_params(path)
        self.assertFalse(result)
        os.remove(path)

if __name__ == "__main__":
    unittest.main()
