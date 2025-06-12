import os
import pickle
import numpy as np
from qgan.discriminator import Discriminator
from config import CFG
from tools.data.data_managers import save_model

class TestDiscriminatorAncillaLoading():
    def __init__(self):
        self.herm = [np.eye(2)]*4
        CFG.extra_ancilla = False
        CFG.system_size = 2
        self.dis_no_ancilla = Discriminator()
        save_model(self.dis_no_ancilla, "tests/qgan/test_dis_no_ancilla.pkl")

        CFG.extra_ancilla = True
        CFG.ancilla_mode = "pass"
        self.dis_ancilla = Discriminator()
        save_model(self.dis_ancilla, "tests/qgan/test_dis_ancilla.pkl")
        
    # Test that you can load models with different ancillas:
    def test_load_from_no_ancilla_to_with_ancilla(self):
        path = "tests/qgan/test_dis_no_ancilla..pkl"
        with open(path, "wb") as f:
            pickle.dump(self.dis_no_ancilla, f)
        result = self.dis_ancilla.load_model_params(path)
        assert result

    def test_load_from_with_ancilla_to_no_ancilla(self):
        path = "tests/qgan/test_dis_ancilla.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.dis_ancilla, f)
        result = self.dis_no_ancilla.load_model_params(path)
        assert result

    # But you cannot load models with different Target sizes
    def test_load_incompatible(self):
        CFG.system_size = 3  # Different system size
        dis_other = Discriminator()
        path = "test_dis_incompatible.pkl"
        with open(path, "wb") as f:
            pickle.dump(dis_other, f)
        result = self.dis_no_ancilla.load_model_params(path)
        assert not result
