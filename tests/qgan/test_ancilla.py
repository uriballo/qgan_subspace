import unittest
from qgan.ancilla import get_final_fake_state_for_discriminator, get_final_real_state_for_discriminator
import numpy as np

class TestAncilla(unittest.TestCase):
    def test_final_fake_state(self):
        state = np.ones((4, 1))
        result = get_final_fake_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[1], 1)

    def test_final_real_state(self):
        state = np.ones((4, 1))
        result = get_final_real_state_for_discriminator(state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[1], 1)

if __name__ == "__main__":
    unittest.main()
