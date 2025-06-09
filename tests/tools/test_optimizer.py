import unittest
import numpy as np
from src.tools.optimizer import MomentumOptimizer

class TestMomentumOptimizer(unittest.TestCase):
    def test_compute_grad(self):
        optimizer = MomentumOptimizer()
        theta = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        new_theta = optimizer.compute_grad(theta, grad, "min")
        self.assertEqual(new_theta.shape, theta.shape)
        new_theta2 = optimizer.compute_grad(theta, grad, "max")
        self.assertEqual(new_theta2.shape, theta.shape)

if __name__ == "__main__":
    unittest.main()
