import unittest
import numpy as np
from src.qgan.discriminator import Discriminator
from src.config import CFG

class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.dis = Discriminator([np.eye(2)]*4, CFG.system_size)

    def test_init(self):
        self.assertEqual(self.dis.size, CFG.system_size)
        self.assertEqual(self.dis.alpha.shape[0], CFG.system_size)
        self.assertEqual(self.dis.beta.shape[0], CFG.system_size)

    def test_getPsi_getPhi(self):
        psi = self.dis.getPsi()
        phi = self.dis.getPhi()
        self.assertEqual(psi.shape, phi.shape)

if __name__ == "__main__":
    unittest.main()
