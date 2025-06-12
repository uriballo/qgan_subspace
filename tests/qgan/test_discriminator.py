import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from qgan.discriminator import Discriminator
from config import CFG
from tools.qobjects.qgates import I, X, Y, Z

class TestDiscriminator():

    def test_init(self):
        dis_size=CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        dis = Discriminator()
        # Asserts
        assert dis.size == dis_size
        assert dis.alpha.shape == (dis_size, 4)
        assert dis.beta.shape == (dis_size, 4)
        assert dis.ancilla == CFG.extra_ancilla
        assert dis.ancilla_mode == CFG.ancilla_mode
        assert dis.target_size == CFG.system_size
        assert dis.herm == [I, X, Y, Z]

    def test_getPsi_getPhi(self):
        dis = Discriminator()
        psi, phi = dis.getPsiPhi()
        assert psi.shape == phi.shape == (2**dis.alpha.shape[0], 2**dis.alpha.shape[0])
