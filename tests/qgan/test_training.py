import unittest
from qgan.training import Training

class TestTraining(unittest.TestCase):
    def test_training_init(self):
        t = Training()
        self.assertIsNotNone(t.gen)
        self.assertIsNotNone(t.dis)
        self.assertIsNotNone(t.total_input_state)
        self.assertIsNotNone(t.total_target_state)

    def test_training_run(self):
        t = Training()
        # Only test that it runs for a few iterations, not full training
        t.gen.qc.depth = 1  # speed up
        t.dis.size = 1
        try:
            t.run()
        except Exception as e:
            self.fail(f"Training.run raised {e}")
