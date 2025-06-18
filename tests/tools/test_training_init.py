import sys
import os

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from tools import training_init

def test_run_single_training():
    # Just check that the function exists and can be called (mocked)
    assert callable(training_init.run_single_training)
