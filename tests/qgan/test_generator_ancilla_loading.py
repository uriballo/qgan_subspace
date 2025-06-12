import sys
import os
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
    gen = Generator()
    os.remove(path)  if os.path.exists(path) else None
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
    def test_load_from_any_to_any_combination_ancilla(self):
        paths = ["tests/qgan/data/test_gen_no_ancilla.pkl", 
                 "tests/qgan/data/test_gen_with_ancilla.pkl"]
       
        for path in paths:
            for extra_ancilla in [True, False]:
                CFG.extra_ancilla = extra_ancilla
                gen = Generator()
                result = gen.load_model_params(path)
                assert result is True # Successfully loaded model

    def test_load_incompatible(self):
        ###############################################################
        # Change layers to make models incompatible
        ###############################################################
        CFG.gen_layers = 2
        gen_other = Generator()
        
        paths = ["tests/qgan/data/test_gen_no_ancilla.pkl", 
                 "tests/qgan/data/test_gen_with_ancilla.pkl"]
       
        for path in paths:
            result = gen_other.load_model_params(path)
            assert result is False # Incompatible model
            
        CFG.gen_layers = 1  # Reset to original value
        
        ###############################################################
        # Change Ansatz to make models incompatible
        ###############################################################
        CFG.gen_layers = 1
        CFG.gen_ansatz = "XX_YY_ZZ_Z"
        gen_other = Generator()
        
        paths = ["tests/qgan/data/test_gen_no_ancilla.pkl", 
                 "tests/qgan/data/test_gen_with_ancilla.pkl"]
       
        for path in paths:
            result = gen_other.load_model_params(path)
            assert result is False
        
        CFG.gen_ansatz = "ZZ_X_Z"  # Reset to original value
            
        ###############################################################
        # Change ancilla topology to make models incompatible
        ###############################################################
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.ancilla_topology = "total"
        CFG.extra_ancilla = True  # Set to True to test with ancilla
        gen_other = Generator()
        
        paths = ["tests/qgan/data/test_gen_no_ancilla.pkl", 
                 "tests/qgan/data/test_gen_with_ancilla.pkl"]
       
        for path in paths:
            result = gen_other.load_model_params(path)
            
            if path == "tests/qgan/data/test_gen_no_ancilla.pkl":
                # This should succeed because the model has no ancilla, so the topology of the other does not matter.
                assert result is True
            if path == "tests/qgan/data/test_gen_with_ancilla.pkl":
                # This should fail because the topology is incompatible with the one used in the saved model.
                # The saved model has "ansatz" topology, while we are trying to load it with "total".
                assert result is False
        
        CFG.ancilla_topology = "ansatz"  # Reset to original value
                
        ###############################################################
        # Change size of Target to make models incompatible
        ###############################################################
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.ancilla_topology = "ansatz"
        CFG.system_size = 3  # Change system size to make models incompatible
        gen_other = Generator()
        
        paths = ["tests/qgan/data/test_gen_no_ancilla.pkl", 
                 "tests/qgan/data/test_gen_with_ancilla.pkl"]
        
        # For any ancilla setting should fail, if target size is incompatible
        for path in paths:
            for ancilla in [True, False]: 
                CFG.extra_ancilla = ancilla
            
                result = gen_other.load_model_params(path)
                
                if path == "tests/qgan/data/test_gen_no_ancilla.pkl":
                    # This should succeed because the model has no ancilla, so the topology of the other does not matter.
                    assert result is True
                if path == "tests/qgan/data/test_gen_with_ancilla.pkl":
                    # This should fail because the topology is incompatible with the one used in the saved model.
                    # The saved model has "ansatz" topology, while we are trying to load it with "total".
                    assert result is False
        
        CFG.system_size = 2

