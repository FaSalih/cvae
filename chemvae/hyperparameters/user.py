"""User configuration."""

from .default import DefaultConfig

from pathlib import Path
import os
import yaml

env = os.getenv("CVAE_DATA")
dir_path = Path(env) if env else Path(__file__).parent
# data relative to here.
DATA_DIR = dir_path / ".." / ".." / "models" / "qm9"


class UserConfig(DefaultConfig):
    """Set your configuration here."""

    def __init__(self, data_dir: Path):
        """Define the parameters.

        Ideally leave these below as default, change only below "USER".
        """
        super().__init__()

        # GENERAL
        self.MAX_LEN = 80  # 120
        self.do_prop_pred = False
        self.TRAIN_MODEL = True
        self.ENC_DEC_TEST = False
        self.PADDING = "right"
        self.RAND_SEED = 101
        self.epochs = 50
        self.batch_size = 126
        self.lr = 0.00039192162392520126
        self.momentum = 0.97170900638688007
        self.latent_dim = 120

        # FILES
        self.checkpoint_path = "./"
        self.data_normalization_out_file: str | None = None
        self.data_file = "qm9.csv"
        self.char_file = "qm9.json"
        self.encoder_weights_file = "encoder_19.keras"
        self.decoder_weights_file = "decoder_19.keras"
        self.prop_pred_weights_file = "qm9_prop_pred_1.keras"
        self.test_idx_file = "test_idx_1.npy"
        self.history_file = "history_1.csv"

        self.vae_annealer_start = 29  # where does this come from?
        self.dropout_rate_mid = 0.082832929704794792
        self.anneal_sigmod_slope = 0.51066543057913916
        self.recurrent_dim = 244  # 488
        self.tgru_dropout = 0.19617749608323892
        self.hg_growth_factor = 1.2281884874932403
        self.middle_layer = 3

        # updates paths.
        for k, v in list(vars(self).items()):
            if k.endswith("file") and isinstance(v, str):
                setattr(self, k, data_dir.joinpath(v).resolve())
            elif k.endswith("path"):
                setattr(self, k, data_dir.joinpath(v).resolve())
        # load chars and set them

        self.CHARS = yaml.safe_load(open(self.char_file))
        self.NCHARS = len(self.CHARS)


config = UserConfig(DATA_DIR)
