import sys
import os

from tools.data.data_managers import print_and_train_log
from tools.training_init import run_test_configurations
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
        # Change results directory to TESTING:
        CFG.base_data_path = f"./generated_data/TESTING/{CFG.run_timestamp}"
        CFG.set_results_paths()

        # Run the test configurations:
        print_and_train_log("Running in TESTING mode.\n", CFG.log_path)
        run_test_configurations()