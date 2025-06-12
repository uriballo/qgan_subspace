import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from config import CFG
from qgan.training import Training

class TestTraining():
    def test_training_init(self):
        t = Training()
        assert t.gen is not None
        assert t.dis is not None
        assert t.total_input_state is not None
        assert t.total_target_state is not None

    def test_training_run(self):
        # No need for this, since I'll do that with running main_testing.py
        ...