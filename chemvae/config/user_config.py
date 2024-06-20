"""User configuration."""

from .default_config import DefaultConfig


class UserConfig(DefaultConfig):
    """Set your configuration here."""

    def __init__(self):
        """Define the parameters.

        Ideally leave these below as default, change only below "USER".
        """
        super().__init__()

        # GENERAL
        self.MAX_LEN = 40  # 120
        self.do_prop_pred = False
        self.TRAIN_MODEL = True
        self.ENC_DEC_TEST = False
        self.PADDING = "right"
        self.RAND_SEED = 101
        self.epochs = 20
        self.batch_size = 50  # 126
        self.lr = 0.00039192162392520126
        self.momentum = 0.9717090063868800
        self.latent_dim = 20  # 60

        # FILES
        self.data_normalization_out_file: str | None = None
        self.data_file = "qm9.csv"
        self.char_file = "qm9.json"
        self.encoder_weights_file = "qm9_encoder_1.keras"
        self.decoder_weights_file = "qm9_decoder_1.keras"
        self.prop_pred_weights_file = "qm9_prop_pred_1.keras"
        self.test_idx_file = "test_idx_1.npy"
        self.history_file = "history_1.csv"
        self.checkpoint_path = "./"

        self.vae_annealer_start = 29  # where does this come from?
        self.dropout_rate_mid = 0.082832929704794792
        self.anneal_sigmod_slope = 0.51066543057913916
        self.recurrent_dim = 128  # 488
        self.tgru_dropout = 0.19617749608323892
        self.hg_growth_factor = 1.2281884874932403
        self.middle_layer = 3  # 3


user_config = UserConfig()
