from config import Config

class TestConfig():
    def test_config_defaults(self):
        cfg = Config()
        assert isinstance(cfg.system_size, int)
        assert cfg.type_of_warm_start in ["none", "all", "some"]
        assert cfg.epochs > 0
        assert cfg.iterations_epoch > 0
        assert isinstance(cfg.extra_ancilla, bool)
        assert isinstance(cfg.gen_layers, int)
        assert isinstance(cfg.l_rate, float)
        assert isinstance(cfg.momentum_coeff, float)
        assert isinstance(cfg.base_data_path, str)
        assert isinstance(cfg.model_gen_path, str)
        assert isinstance(cfg.model_dis_path, str)
        assert isinstance(cfg.log_path, str)
        assert isinstance(cfg.fid_loss_path, str)
        assert isinstance(cfg.gen_final_params_path, str)
