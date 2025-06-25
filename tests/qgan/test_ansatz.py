import sys
import os

import numpy as np

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))



from tools.data.data_managers import save_model
from qgan.generator import Generator, Ansatz
from config import CFG

# Set up the same configuration than inside the tests.
CFG.extra_ancilla = True
CFG.system_size = 3
total_input_state = np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1)))

class TestGeneratorAnsatzAncillaModes():
    def test_all_ansatz_types(self):
        CFG.extra_ancilla = True
        CFG.system_size = 3
        
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            gen = Generator(total_input_state)
            gen.qc = Ansatz.get_ansatz_type_circuit(ansatz)(gen.qc, gen.size, 1)
            assert gen.qc is not None
            assert len(gen.qc.gates) >= 1
            
        # Check that one ansatz is bigger than the other in parameters
        gen1 = Generator(total_input_state)
        gen1.qc = Ansatz.get_ansatz_type_circuit("XX_YY_ZZ_Z")(gen1.qc, gen1.size, 1)
        gen2 = Generator(total_input_state)
        gen2.qc = Ansatz.get_ansatz_type_circuit("ZZ_X_Z")(gen2.qc, gen2.size, 1)
        assert len(gen1.qc.gates) > len(gen2.qc.gates), "XX_YY_ZZ_Z should have more gates than ZZ_X_Z"
    
    def test_number_of_gates_duplicates_with_layers(self):
        CFG.extra_ancilla = True
        CFG.system_size = 3
        
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            CFG.gen_ansatz = ansatz
            
            CFG.gen_layers = 1
            gen = Generator(total_input_state)
            old_num_gates = len(gen.qc.gates)
            
            CFG.gen_layers = 2
            gen = Generator(total_input_state)
            new_num_gates = len(gen.qc.gates)
            
            # Check that the number of gates is doubled with layers
            assert new_num_gates == old_num_gates * 2, "Number of gates should double with each layer"
            
    def test_ancilla_adds_more_gates(self):
        CFG.extra_ancilla = False
        CFG.system_size = 2
        CFG.gen_layers = 1
        
        gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        gen.qc = Ansatz.get_ansatz_type_circuit(CFG.gen_ansatz)(gen.qc, gen.size, CFG.gen_layers)
        
        old_num_gates = len(gen.qc.gates)
        
        # Add ancilla and check that the number of gates increases
        CFG.extra_ancilla = True
        for topo in ["trivial", "disconnected", "ansatz", "bridge", "total"]:
            CFG.ancilla_topology = topo
            gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            gen.qc = Ansatz.get_ansatz_type_circuit(CFG.gen_ansatz)(gen.qc, gen.size, CFG.gen_layers)
            new_num_gates = len(gen.qc.gates)
        
            if topo != "trivial":
                assert new_num_gates > old_num_gates
            else:
                assert new_num_gates == old_num_gates, "In trivial topology, the number of gates should not change"
            
    def test_ancilla_modes_and_topologies(self):
        # Try all combinations of ancilla_mode and ancilla_topology
        modes = ["pass", "project", "trace"]
        topologies = ["trivial", "disconnected", "ansatz", "bridge", "total"]

        CFG.extra_ancilla = True
        CFG.system_size = 4
        CFG.gen_layers = 1
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            CFG.gen_ansatz = ansatz

            for mode in modes:
                CFG.ancilla_mode = mode
                for topo in topologies:
                    CFG.ancilla_topology = topo

                    gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
                    assert gen.qc is not None
                    assert len(gen.qc.gates) >= 1

                    # Check that the topology is applied correctly
                    last_qubit = gen.qc.size - 1
                    ancilla_all_gates = [g for g in gen.qc.gates if last_qubit in [g.qubit1, g.qubit2]]
                    ancilla_2q_gates = [g for g in ancilla_all_gates if g.qubit2 is not None]

                    if topo != "trivial":
                        assert len(ancilla_all_gates) != 0, f"Ancilla gates should not be empty in {topo} topology"

                    qubit0_all_gates = [g for g in gen.qc.gates if 0 in [g.qubit2, g.qubit1]]
                    qubit1_all_gates = [g for g in gen.qc.gates if 1 in [g.qubit2, g.qubit1]]
                    qubit0_1q_gates = [g for g in qubit0_all_gates if g.qubit2 is None]
                    qubit0_2q_gates = [g for g in qubit0_all_gates if g.qubit2 is not None]
                    qubit1_2q_gates = [g for g in qubit1_all_gates if g.qubit2 is not None]

                    # No ancilla gates, should not be empty
                    for gates in [qubit0_all_gates, qubit1_all_gates, qubit0_1q_gates, qubit0_2q_gates, qubit1_2q_gates]:
                        assert len(gates) > 0

                    # Assert that for topologies different than "ansatz" the number gates of the ancilla are different than the rest:
                    if topo in ["total"]:
                        # In total topo, for sufficient qubits >4, ancilla will have more gates than the rest
                        assert len(ancilla_2q_gates) > len(qubit0_2q_gates)
                        assert len(ancilla_all_gates) > len(qubit0_all_gates)
                        assert len(ancilla_2q_gates) > len(qubit1_2q_gates)
                        assert len(ancilla_all_gates) > len(qubit1_all_gates)
                    elif topo == "bridge":
                        # In bridge topology, all qubits should have the same number of gates (for close neighbors ansatz at least)
                        assert len(ancilla_2q_gates) == len(qubit0_2q_gates)
                        assert len(ancilla_all_gates) == len(qubit0_all_gates)
                        assert len(ancilla_2q_gates) == len(qubit1_2q_gates)
                        assert len(ancilla_all_gates) == len(qubit1_all_gates)
                    elif topo == "ansatz":
                        # For ansatz, it will have same as the other extremes, and less than middle qubits
                        assert len(ancilla_2q_gates) == len(qubit0_2q_gates)
                        assert len(ancilla_all_gates) == len(qubit0_all_gates)
                        assert len(ancilla_2q_gates) < len(qubit1_2q_gates)
                        assert len(ancilla_all_gates) < len(qubit1_all_gates)
                    elif topo == "disconnected":
                        # In disconnected topology, ancilla gates should only be the 1q gates
                        assert not ancilla_2q_gates
                        assert len(ancilla_all_gates) != 0
                        assert len(ancilla_all_gates) == len(qubit0_1q_gates)
                    elif topo == "trivial":
                        # In trivial topology, there should be no ancilla gates
                        assert not ancilla_all_gates, f"Ancilla gates should be empty in {topo} topology"

    def test_save_and_load_with_various_ansatz(self):
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            # Create a generator with the specified ansatz and angles set to a known value
            gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            gen.qc = Ansatz.get_ansatz_type_circuit(ansatz)(gen.qc, gen.size, 1)
            for gate in gen.qc.gates:
                gate.angle = 0.456
            # Save the model
            path = f"tests/qgan/data/test_gen_{ansatz}.pkl"
            save_model(gen, path)
            # Load the model
            gen2 = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            gen2.qc = Ansatz.get_ansatz_type_circuit(ansatz)(gen2.qc, gen2.size, 1)
            # Assert that the model loads correctly and angles are preserved
            assert gen2.load_model_params(path)
            assert any(g.angle == 0.456 for g in gen2.qc.gates)
            os.remove(path)

    def test_that_ancilla_adds_last_subspace_in_qc_get_mat_rep(self):
        CFG.extra_ancilla = False
        CFG.system_size = 2
        CFG.gen_layers = 1
        
        gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        # Set fixed angles for the gates
        for gate in gen.qc.gates:
            gate.angle = 0.456
            
        old_matrix = gen.qc.get_mat_rep().copy()
        
        CFG.extra_ancilla = True
        for topo in ["trivial"]: # We only do this, since its where ancilla remains in |0> without entangling
            CFG.ancilla_topology = topo
            gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            # Set fixed angles for the gates
            for gate in gen.qc.gates:
                gate.angle = 0.456
            
            new_matrix = gen.qc.get_mat_rep().copy()
            
            # Assert that tracing out the last qubit gives the same matrix as before
            # Keep every other row and column (i.e., trace out the last qubit)
            assert np.allclose(old_matrix, new_matrix[::2, ::2]), f"Matrix should be the same when tracing out last qubit in {topo} topology"
            