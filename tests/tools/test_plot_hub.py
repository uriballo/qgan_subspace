import sys
import os
# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from tools import plot_hub
from config import CFG
import numpy as np

class TestPlotHub():
    def test_plt_fidelity_vs_iter(self):
        # Should not raise error
        fidelities = np.random.rand(10)
        losses = np.random.rand(10)
        
        plot_hub.plt_fidelity_vs_iter(fidelities, losses, CFG, 1)
