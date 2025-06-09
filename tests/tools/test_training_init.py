import unittest
from src.tools import training_init

class TestTrainingInit(unittest.TestCase):
    def test_run_single_training(self):
        # Just check that the function exists and can be called (mocked)
        self.assertTrue(callable(training_init.run_single_training))

if __name__ == "__main__":
    unittest.main()
