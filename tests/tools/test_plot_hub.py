import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from tools import plot_hub
from config import CFG
import numpy as np

class TestPlotHub():
    def test_plt_fidelity_vs_iter(self):
        # Should not raise error
        fidelities = np.random.rand(10)
        losses = np.random.rand(10)
        try:
            plot_hub.plt_fidelity_vs_iter(fidelities, losses, CFG, 1)
        except Exception as e:
            self.fail(f"plt_fidelity_vs_iter raised {e}")
