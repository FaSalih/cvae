"""User configuration."""

from pathlib import Path
from typing import Literal

import yaml


class Config:
    """Set your configuration here."""

    def __init__(self, data_dir: Path):
        """Define the parameters.

        Ideally leave these below as default, change only below "USER".
        """

        # FILES
        self.checkpoint_path: str = "./"
        self.data_normalization_out_file: str | None = None
        self.data_file: str = "qm9.csv"
        self.char_file: str = "qm9.json"
        self.encoder_weights_file: str = "encoder_2.keras"
        self.decoder_weights_file: str = "decoder_2.keras"
        self.prop_pred_weights_file: str = "qm9_prop_pred.keras"
        self.test_idx_file: str = "test_idx.npy"
        self.history_file: str = "history_4.csv"

        # GENERAL PARAMETERS
        self.batch_size: int = 100
        self.epochs: int = 70
        self.batch_size: int = 126
        self.latent_dim: int = 156  # 196
        self.val_split: float = 0.1  # validation split
        self.loss: str = "categorical_crossentropy"  # set reconstruction loss
        self.do_prop_pred: bool = False

        self.MAX_LEN: int = 34  # 120
        self.TRAIN_MODEL: bool = True
        self.ENC_DEC_TEST: bool = False
        self.PADDING: Literal["right", "left", "none"] = "right"
        self.RAND_SEED: int = 42

        # OPTIMIZATION PARAMETERS
        self.lr: float = 0.000312087049936
        self.momentum: float = 0.936948773087
        self.optim: str = "adam"  # optimizer to be used

        # CONVOLUTION PARAMETERS
        self.batchnorm_conv: bool = True
        # block: Conv+Batch Norm. But BN can be disabled above.
        self.conv_layer_blocks: int = 3  # conv depth
        self.conv_filters: int = 5
        self.conv_filter_growth_factor: float = 0.5
        self.conv_kernel_height: int = 6
        self.conv_kernel_growth_factor: float = 0.85

        self.conv_activation: str = "tanh"

        # MIDDLE LAYER PARAMETERS
        # growth factor applied to determine size of next middle layer
        self.hg_growth_factor: float = 1.4928245388
        self.middle_layer: int = 1
        self.dropout_rate_mid: float = 0.0
        self.batchnorm_mid: bool = True  # apply batch normalization to middle layer
        self.activation: str = "tanh"

        # DECODER PARAMETERS
        self.gru_depth: int = 4
        self.rnn_activation: str = "tanh"
        self.recurrent_dim: int = 50
        # use CPU intensive implementation; other implementation modes
        # (1 - GPU 2- memory) are not yet implemented
        self.temperature: float = 1.00  # amount of noise for sampling the final output

        # VAE PARAMETERS
        self.vae_annealer_start: int = (
            22  # Center for variational weight annealer. What is that?
        )
        self.batchnorm_vae: bool = (
            False  # apply batch normalization to output of the variational layer
        )
        self.vae_activation: str = "tanh"
        self.xent_loss_weight: float = (
            1.0  # loss weight to assign to reconstruction error
        )
        self.kl_loss_weight: float = 1.0  # loss weight to assign to KL loss
        self.anneal_sigmod_slope: float = (
            1.0  # slope of sigmoid variational weight annealer
        )
        # Choice of freezing the variational layer until close to the anneal starting epoch
        self.freeze_logvar_layer: bool = False
        # the number of epochs before vae_annealer_start where the variational layer should be unfrozen
        self.freeze_offset: int = 1

        # PROPERTY PREDICTION PARAMETERS
        self.reg_prop_tasks: list[str] | None = None
        self.logit_prop_tasks: list[str] | None = None
        # path to write out normalized regression data (or None)
        # ratio between consecutive layers in property pred.
        self.prop_pred_depth: int = 3
        self.prop_hidden_dim: int = 36
        self.prop_growth_factor: float = 0.8
        self.prop_pred_activation: str = "tanh"
        # loss function to use with property prediction error for regression task
        self.reg_prop_pred_loss: str = "mse"
        # loss function to use with property prediction for logistic task
        self.logit_prop_pred_loss: str = "binary_crossentropy"
        self.prop_pred_loss_weight: float = 0.5
        self.prop_pred_dropout: float = 0.0
        self.prop_batchnorm: bool = True

        self.verbosity: int = 1
        self.limit_data: int | None = None
        self.reload_model: bool = False  # startmodel from checkpoint
        self.prev_epochs: int = 0

        self.vae_annealer_start: int = 22  # 29  # where does this come from?
        self.dropout_rate_mid: float = 0.082832929704794792
        self.anneal_sigmod_slope: float = 0.51066543057913916
        self.recurrent_dim: int = 488
        self.hg_growth_factor: float = 1.2281884874932403
        self.middle_layer: int = 1

        # concat and make them paths
        for k, v in list(vars(self).items()):
            if k.endswith("file") and isinstance(v, str):
                setattr(self, k, data_dir.joinpath(v).resolve())
            elif k.endswith("path"):
                setattr(self, k, data_dir.joinpath(v).resolve())

        # load chars and set them
        self.CHARS: str = yaml.safe_load(open(self.char_file))
        self.NCHARS: int = len(self.CHARS)
