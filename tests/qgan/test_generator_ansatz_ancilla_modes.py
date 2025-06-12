import os
import pickle
from qgan.generator import Generator, Ansatz
from config import CFG

class TestGeneratorAnsatzAncillaModes():
    def test_all_ansatz_types(self):
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            gen = Generator()
            gen.qc = Ansatz.get_ansatz_type_circuit(ansatz)(gen.qc, gen.size, 1)
            assert gen.qc is not None
            assert len(gen.qc.gates) >= 1

    def test_ancilla_modes_and_topologies(self):
        # Try all combinations of ancilla_mode and ancilla_topology
        modes = ["pass", "project", "trace"]
        topologies = ["trivial", "disconnected", "ansatz", "bridge", "total"]
        for mode in modes:
            for topo in topologies:
                CFG.extra_ancilla = True
                CFG.ancilla_mode = mode
                CFG.ancilla_topology = topo
                CFG.system_size = 2
                CFG.gen_layers = 1
                gen = Generator()
                gen.qc = Ansatz.get_ansatz_type_circuit(CFG.gen_ansatz)(gen.qc, gen.size, CFG.gen_layers)
                assert gen.qc is not None
                assert len(gen.qc.gates) >= 1

    def test_save_and_load_with_various_ansatz(self):
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            gen = Generator()
            gen.qc = Ansatz.get_ansatz_type_circuit(ansatz)(gen.qc, gen.size, 1)
            for gate in gen.qc.gates:
                gate.angle = 0.456
            path = f"test_gen_{ansatz}.pkl"
            with open(path, "wb") as f:
                pickle.dump(gen, f)
            gen2 = Generator()
            gen2.qc = Ansatz.get_ansatz_type_circuit(ansatz)(gen2.qc, gen2.size, 1)
            assert gen2.load_model_params(path)
            assert any(g.angle == 0.456 for g in gen2.qc.gates)
            os.remove(path)
