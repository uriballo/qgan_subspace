import numpy as np
import os
from qgan.discriminator import Discriminator
from qgan.generator import Generator
from config import CFG

class TestGenerator():
    def __init__(self):
        self.gen = Generator()

    def test_init(self):
        assert self.gen.size == CFG.system_size + (1 if CFG.extra_ancilla else 0)
        assert self.gen.qc is not None

    def test_update_gen(self):
        dis = Discriminator()
        total_target_state = np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1)))
        total_input_state = np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1)))
        try:
            self.gen.update_gen(dis, total_target_state, total_input_state)
        except Exception as e:
            self.fail(f"update_gen raised {e}")

    def test_load_model_params_self(self):
        # Save and load self
        import pickle
        path = "test_gen.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.gen, f)
        self.assertTrue(self.gen.load_model_params(path))
        os.remove(path)
