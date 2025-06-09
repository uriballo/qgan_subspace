import unittest
import os
import pickle
import numpy as np
from src.qgan.discriminator import Discriminator
from src.config import CFG

class TestDiscriminatorAncillaLoading(unittest.TestCase):
    def setUp(self):
        self.herm = [np.eye(2)]*4
        self.cfg_no_ancilla = CFG
        self.cfg_no_ancilla.extra_ancilla = False
        self.cfg_no_ancilla.system_size = 2
        self.dis_no_ancilla = Discriminator(self.herm, self.cfg_no_ancilla.system_size)
        self.cfg_with_ancilla = CFG
        self.cfg_with_ancilla.extra_ancilla = True
        self.cfg_with_ancilla.system_size = 2
        self.dis_with_ancilla = Discriminator(self.herm, self.cfg_with_ancilla.system_size + 1)

    def test_load_from_no_ancilla_to_with_ancilla(self):
        path = "test_dis_no_ancilla.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.dis_no_ancilla, f)
        result = self.dis_with_ancilla.load_model_params(path)
        self.assertTrue(result)
        os.remove(path)

    def test_load_from_with_ancilla_to_no_ancilla(self):
        path = "test_dis_with_ancilla.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.dis_with_ancilla, f)
        result = self.dis_no_ancilla.load_model_params(path)
        self.assertTrue(result)
        os.remove(path)

    def test_load_incompatible(self):
        dis_other = Discriminator(self.herm, 3)
        path = "test_dis_incompatible.pkl"
        with open(path, "wb") as f:
            pickle.dump(dis_other, f)
        result = self.dis_no_ancilla.load_model_params(path)
        self.assertFalse(result)
        os.remove(path)

if __name__ == "__main__":
    unittest.main()
