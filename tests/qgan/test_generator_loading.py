import sys
import os

import numpy as np

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from qgan.generator import Generator
from config import CFG
from tools.data.data_managers import save_model

############################################################################
# SAVE DISCRIMINATOR MODELS WITH DIFFERENT ANCILLA MODES
############################################################################
CFG.system_size = 2
CFG.gen_layers = 1
CFG.gen_ansatz = "ZZ_X_Z"
CFG.ancilla_topology = "ansatz"

def save_test_model(path):
    gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
    os.remove(path) if os.path.exists(path) else None
    save_model(gen, path)

# Save no ancilla
CFG.extra_ancilla = False
save_test_model("tests/qgan/data/test_gen_no_ancilla.pkl")

# Save with ancilla
CFG.extra_ancilla = True
save_test_model("tests/qgan/data/test_gen_with_ancilla.pkl")
        
class TestGeneratorAncillaLoading():
    ################################################
    # If we only change adding or removing ancilla, we can load models with different ancilla settings.
    ################################################

    def test_from_any_to_any_combination_ancilla(self):
        paths = ["tests/qgan/data/test_gen_no_ancilla.pkl", 
                 "tests/qgan/data/test_gen_with_ancilla.pkl"]
        # Load other params, for possible problems of parallelism in tests
        CFG.system_size = 2
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.ancilla_topology = "ansatz"
        CFG.target_hamiltonian = "cluster_h"
        
        ##############################################################################
        # Successful load, if we only change ancilla (same other settings).
        ##############################################################################
        
        for path in paths:
            for extra_ancilla in [True, False]:
                CFG.extra_ancilla = extra_ancilla
                gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
                result = gen.load_model_params(path)
                assert result is True # Successfully loaded model
 
    def test_cases_we_shouldnt_load(self):
        paths = ["tests/qgan/data/test_gen_no_ancilla.pkl", 
                 "tests/qgan/data/test_gen_with_ancilla.pkl"]
        
        # Load other params, for possible problems of parallelism in tests
        CFG.system_size = 2
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.ancilla_topology = "ansatz"
                
        ###############################################################
        # Change layers to make models incompatible
        ###############################################################
        CFG.gen_layers = 2
        gen_other = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
       
        for path in paths:
            result = gen_other.load_model_params(path)
            assert result is False # Incompatible model
            
        CFG.gen_layers = 1  # Reset to original value
        
        ###############################################################
        # Change Ansatz to make models incompatible
        ###############################################################
        CFG.gen_layers = 1
        CFG.gen_ansatz = "XX_YY_ZZ_Z"
        gen_other = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
       
        for path in paths:
            result = gen_other.load_model_params(path)
            assert result is False
        
        CFG.gen_ansatz = "ZZ_X_Z"  # Reset to original value
        
        ###############################################################
        # Change size of Target to make models incompatible
        ###############################################################
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.system_size = 3  # Change system size to make models incompatible
        gen_other = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        
        for path in paths:
            for ancilla in [True, False]: 
                CFG.extra_ancilla = ancilla
                result = gen_other.load_model_params(path)
                # For any ancilla setting should fail, if target size is incompatible
                assert result is False
        
        CFG.system_size = 2
            
        ###############################################################
        # Change ancilla topology to make models incompatible
        ###############################################################
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.ancilla_topology = "total"
        CFG.extra_ancilla = True  # Set to True to test with ancilla
        gen_other = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
       
        for path in paths:
            result = gen_other.load_model_params(path)
            
            if path == "tests/qgan/data/test_gen_no_ancilla.pkl":
                # This should succeed because the model has no ancilla, so the topology of the other does not matter.
                assert result is True
            elif path == "tests/qgan/data/test_gen_with_ancilla.pkl":
                # This should fail because the topology is incompatible with the one used in the saved model.
                # The saved model has "ansatz" topology, while we are trying to load it with "total".
                assert result is False
        
        CFG.ancilla_topology = "ansatz"  # Reset to original value
        
        ################################################################
        # Change hamiltonian to make models incompatible
        ################################################################
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.ancilla_topology = "ansatz"
        CFG.extra_ancilla = True  # Set to True to test with ancilla
        CFG.target_hamiltonian = "custom_h"
        CFG.custom_hamiltonian_terms = ["ZZ"]
        
        gen_other = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        for path in paths:
            result = gen_other.load_model_params(path)
            # This should fail because the hamiltonian is incompatible with the one used in the saved model.
            assert result is False

    def test_partial_angle_loading(self):
        CFG.system_size = 2
        CFG.gen_layers = 1
        CFG.extra_ancilla = False

        # Create a model with a known angle
        gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        for gate in gen.qc.gates:
            gate.angle = 0.123
        original_gates_num = len(gen.qc.gates)

        # Save the model
        path = "tests/qgan/data/test_gen_partial.pkl"
        os.remove(path) if os.path.exists(path) else None
        save_model(gen, path)

        # Create a new model with different configurations:
        for extra_ancilla in [True, False]:
            CFG.extra_ancilla = extra_ancilla
            for ancilla_topology in ["disconnected", "ansatz", "bridge", "total"]:
                CFG.ancilla_topology = ancilla_topology

                gen_with_ancilla = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
                # All angles should be different before loading
                assert any(g.angle != 0.123 for g in gen_with_ancilla.qc.gates)
                # Load
                gen_with_ancilla.load_model_params(path)

                # At least some angles should now be 0.123 (for gates not involving ancilla)
                assert any(g.angle == 0.123 for g in gen_with_ancilla.qc.gates)
                # Concretely the number of gates with angle 0.123 should be the same as in the original model without ancilla
                assert len([g for g in gen_with_ancilla.qc.gates if g.angle == 0.123]) == original_gates_num
