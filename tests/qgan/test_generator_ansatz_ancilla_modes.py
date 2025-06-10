import unittest
import os
import pickle
from qgan.generator.generator import Generator
from qgan.generator.ansatz import get_ansatz_func
from config import CFG

class TestGeneratorAnsatzAncillaModes(unittest.TestCase):
    def test_all_ansatz_types(self):
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            gen = Generator(2)
            gen.set_qcircuit(get_ansatz_func(ansatz)(gen.qc, gen.size, 1))
            self.assertIsNotNone(gen.qc)
            self.assertGreaterEqual(len(gen.qc.gates), 1)

    def test_ancilla_modes_and_topologies(self):
        # Try all combinations of ancilla_mode and ancilla_topology
        modes = ["pass", "project", "trace"]
        topologies = ["trivial", "disconnected", "ansatz", "bridge", "total"]
        for mode in modes:
            for topo in topologies:
                cfg = CFG
                cfg.extra_ancilla = True
                cfg.ancilla_mode = mode
                cfg.ancilla_topology = topo
                cfg.system_size = 2
                cfg.gen_layers = 1
                gen = Generator(cfg.system_size + 1)
                gen.set_qcircuit(get_ansatz_func(cfg.gen_ansatz)(gen.qc, gen.size, cfg.gen_layers))
                self.assertIsNotNone(gen.qc)
                self.assertGreaterEqual(len(gen.qc.gates), 1)

    def test_save_and_load_with_various_ansatz(self):
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            gen = Generator(2)
            gen.set_qcircuit(get_ansatz_func(ansatz)(gen.qc, gen.size, 1))
            for gate in gen.qc.gates:
                gate.angle = 0.456
            path = f"test_gen_{ansatz}.pkl"
            with open(path, "wb") as f:
                pickle.dump(gen, f)
            gen2 = Generator(2)
            gen2.set_qcircuit(get_ansatz_func(ansatz)(gen2.qc, gen2.size, 1))
            self.assertTrue(gen2.load_model_params(path))
            self.assertTrue(any(g.angle == 0.456 for g in gen2.qc.gates))
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
