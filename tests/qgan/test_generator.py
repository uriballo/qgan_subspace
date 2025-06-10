import unittest
import numpy as np
import os
from qgan.generator.generator import Generator
from qgan.generator.ansatz import get_ansatz_func
from config import CFG

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = Generator(CFG.system_size)
        self.gen.set_qcircuit(get_ansatz_func(CFG.gen_ansatz)(self.gen.qc, self.gen.size, CFG.gen_layers))

    def test_init(self):
        self.assertEqual(self.gen.size, CFG.system_size)
        self.assertIsNotNone(self.gen.qc)

    def test_update_gen(self):
        # Dummy objects for dis, total_real_state, total_input_state
        class DummyDis:
            def getPhi(self): return np.eye(2**CFG.system_size)
            def getPsi(self): return np.eye(2**CFG.system_size)
        dis = DummyDis()
        total_real_state = np.ones((2**CFG.system_size, 1))
        total_input_state = np.ones((2**CFG.system_size, 1))
        try:
            self.gen.update_gen(dis, total_real_state, total_input_state)
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

if __name__ == "__main__":
    unittest.main()
