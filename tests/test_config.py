import unittest
from src.config import Config

class TestConfig(unittest.TestCase):
    def test_config_defaults(self):
        cfg = Config()
        self.assertIsInstance(cfg.system_size, int)
        self.assertIn(cfg.type_of_warm_start, ["none", "all", "some"])
        self.assertTrue(cfg.epochs > 0)
        self.assertTrue(cfg.iterations_epoch > 0)
        self.assertIsInstance(cfg.extra_ancilla, bool)
        self.assertIsInstance(cfg.gen_layers, int)
        self.assertIsInstance(cfg.l_rate, float)
        self.assertIsInstance(cfg.momentum_coeff, float)
        self.assertIsInstance(cfg.base_data_path, str)
        self.assertIsInstance(cfg.model_gen_path, str)
        self.assertIsInstance(cfg.model_dis_path, str)
        self.assertIsInstance(cfg.log_path, str)
        self.assertIsInstance(cfg.fid_loss_path, str)
        self.assertIsInstance(cfg.gen_final_params_path, str)

if __name__ == "__main__":
    unittest.main()
