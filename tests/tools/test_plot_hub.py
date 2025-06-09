import unittest
from src.tools import plot_hub
from src.config import CFG
import numpy as np

class TestPlotHub(unittest.TestCase):
    def test_plt_fidelity_vs_iter(self):
        # Should not raise error
        fidelities = np.random.rand(10)
        losses = np.random.rand(10)
        try:
            plot_hub.plt_fidelity_vs_iter(fidelities, losses, CFG, 1)
        except Exception as e:
            self.fail(f"plt_fidelity_vs_iter raised {e}")

if __name__ == "__main__":
    unittest.main()
