import unittest
import os
import pickle
import numpy as np
from qgan.generator.generator import Generator
from qgan.generator.ansatz import get_ansatz_func
from config import CFG
from tools.data.data_managers import save_model

class TestGeneratorAncillaLoading(unittest.TestCase):
    def setUp(self):
        # Create a generator with and without ancilla
        CFG.extra_ancilla = False
        CFG.system_size = 2
        CFG.gen_layers = 1
        self.gen_no_ancilla = Generator()
        save_model(self.gen_no_ancilla, "tests/qgan/test_gen_no_ancilla.pkl")
        
        CFG.extra_ancilla = True
        self.gen_with_ancilla = Generator()
        save_model(self.gen_with_ancilla, "tests/qgan/test_gen_with_ancilla.pkl")

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
        CFG.gen_layers = 2
        gen_other = Generator()
        path = "test_gen_incompatible.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen_other, f)
        result = self.gen_no_ancilla.load_model_params(path)
        self.assertFalse(result)
        os.remove(path)
