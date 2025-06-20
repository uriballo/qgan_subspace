import sys
import os
# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
from tools.optimizer import MomentumOptimizer

class TestMomentumOptimizer():
    def test_compute_grad(self):
        optimizer = MomentumOptimizer()
        theta = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        new_theta = optimizer.move_in_grad(theta, grad, "min")
        assert new_theta.shape == theta.shape
        new_theta2 = optimizer.move_in_grad(theta, grad, "max")
        assert new_theta2.shape == theta.shape
        
        # Check that the new theta is different from the old theta
        assert np.all(new_theta != theta)
        assert np.all(new_theta2 != theta)
        