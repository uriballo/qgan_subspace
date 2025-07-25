import sys
import os
# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
from qgan.discriminator import Discriminator
from config import CFG
from tools.data.data_managers import save_model

############################################################################
# SAVE DISCRIMINATOR MODELS WITH DIFFERENT ANCILLA MODES
############################################################################
CFG.system_size = 2
CFG.target_hamiltonian = "cluster_h"  # Set a target Hamiltonian for the tests

def save_test_model(path):
    dis = Discriminator()
    os.remove(path)  if os.path.exists(path) else None
    save_model(dis, path)

# Save no ancilla
CFG.extra_ancilla = False
save_test_model("tests/qgan/data/test_dis_no_ancilla.pkl")

# Save with ancilla pass
CFG.extra_ancilla = True
CFG.ancilla_mode = "pass"
save_test_model("tests/qgan/data/test_dis_ancilla_pass.pkl")

# Save with ancilla pass
CFG.extra_ancilla = True
CFG.ancilla_mode = "project"
save_test_model("tests/qgan/data/test_dis_ancilla_project.pkl")



class TestDiscriminatorAncillaLoading():
    
    # Test that you can load models with different ancillas:
    def test_load_from_any_to_any_combination_of_ancilla(self):
        # Load for compatibility with test paralelism
        CFG.system_size = 2
        CFG.target_hamiltonian = "cluster_h"
        
        paths = ["tests/qgan/data/test_dis_no_ancilla.pkl", 
                 "tests/qgan/data/test_dis_ancilla_pass.pkl", 
                 "tests/qgan/data/test_dis_ancilla_project.pkl"]
        for path in paths:
            for extra_ancilla in ["none", "pass", "project"]:
                if extra_ancilla == "none":
                    CFG.extra_ancilla = False
                elif extra_ancilla == "pass":
                    CFG.extra_ancilla = True
                    CFG.ancilla_mode = "pass"
                elif extra_ancilla == "project":
                    CFG.extra_ancilla = True
                    CFG.ancilla_mode = "project"
                self.dis = Discriminator()
                result = self.dis.load_model_params(path)
                assert result is True # Successfully loaded model

    # But you cannot load models with different Target sizes
    def test_load_incompatible(self):
        # Load for compatibility with test paralelism
        CFG.system_size = 2
        CFG.target_hamiltonian = "cluster_h"
        
        paths = ["tests/qgan/data/test_dis_no_ancilla.pkl", 
            "tests/qgan/data/test_dis_ancilla_pass.pkl", 
            "tests/qgan/data/test_dis_ancilla_project.pkl"]
        
        #######################################################################
        # Change target size
        #######################################################################
        # Set a bigger system size, without ancilla, which should have same size as original with ancilla
        # but that we don't want to load anyway, since Targets have different sizes.
        for extra_ancilla in [False, True]:
            CFG.extra_ancilla = extra_ancilla
            CFG.system_size = 3
            dis_other = Discriminator()
            
            for path in paths:
                result = dis_other.load_model_params(path)
                assert result is False # Failed to load model due to size mismatch
                
        ####################################################################
        # Change the target hamiltonian
        ####################################################################
        # Set a different target hamiltonian, which should make the model incompatible
        CFG.target_hamiltonian = "custom_h"
        CFG.custom_hamiltonian_terms = ["ZZ"]
        
        dis_other = Discriminator()
            
        for path in paths:
            result = dis_other.load_model_params(path)
            assert result is False # Failed to load model due to size mismatch

    # def 