import unittest
from config import CFG
from qgan.training import Training

class TestTraining(unittest.TestCase):
    def test_training_init(self):
        t = Training()
        self.assertIsNotNone(t.gen)
        self.assertIsNotNone(t.dis)
        self.assertIsNotNone(t.total_input_state)
        self.assertIsNotNone(t.total_target_state)

    def test_training_run(self):
        # No need for this, since I'll do that with running main_testing.py
        ...