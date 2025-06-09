import unittest
from src.tools.data import loading_helpers
from src.config import CFG

class TestTools(unittest.TestCase):
    def test_load_models_if_specified(self):
        class DummyTraining:
            class DummyGen:
                def load_model_params(self, path): return True
            class DummyDis:
                def load_model_params(self, path): return True
            gen = DummyGen()
            dis = DummyDis()
        try:
            loading_helpers.load_models_if_specified(DummyTraining(), CFG)
        except Exception as e:
            self.fail(f"load_models_if_specified raised {e}")

if __name__ == "__main__":
    unittest.main()
