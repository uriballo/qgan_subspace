import unittest
import numpy as np
from qgan.discriminator import Discriminator
from config import CFG
from tools.qobjects.qgates import I, X, Y, Z

class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.dis = Discriminator()

    def test_init(self):
        dis_size=CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        
        # Asserts
        self.assertEqual(self.dis.size, dis_size)
        self.assertEqual(self.dis.alpha.shape, (dis_size, 4))
        self.assertEqual(self.dis.beta.shape, (dis_size, 4))
        assert self.dis.ancilla == CFG.extra_ancilla
        assert self.dis.ancilla_mode == CFG.ancilla_mode
        assert self.dis.target_size == CFG.system_size
        assert self.dis.herm == [I, X, Y, Z]

    def test_getPsi_getPhi(self):
        psi, phi = self.dis.getPsiPhi()
        assert psi.shape == phi.shape == (2**self.dis.alpha.shape[0], 2**self.dis.alpha.shape[0])
