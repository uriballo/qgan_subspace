import unittest
import numpy as np
from qgan.discriminator import Discriminator
from config import CFG

class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.dis = Discriminator()

    def test_init(self):
        dis_size=CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        self.assertEqual(self.dis.size, dis_size)
        self.assertEqual(self.dis.alpha.shape[0], dis_size)
        self.assertEqual(self.dis.beta.shape[0], dis_size)

    def test_getPsi_getPhi(self):
        psi, phi = self.dis.getPsiPhi()
        assert psi.shape == phi.shape == self.dis.alpha.shape == self.dis.beta.shape
