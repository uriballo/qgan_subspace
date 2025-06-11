import sys
import os

from qgan.training import Training
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import unittest
from tools.data import loading_helpers
from config import CFG

class TestTools(unittest.TestCase):
    def test_load_models_if_specified(self):
        try:
            loading_helpers.load_models_if_specified(Training())
        except Exception as e:
            self.fail(f"load_models_if_specified raised {e}")
