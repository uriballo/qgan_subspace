import sys
import os

from qgan.training import Training
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from tools.data import loading_helpers
from config import CFG

def test_tools():
    try:
        loading_helpers.load_models_if_specified(Training())
    except Exception as e:
        assert False, f"load_models_if_specified raised {e}"
