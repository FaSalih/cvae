"""Recommended that this isn't modified, rather overwrite the user config."""


class DefaultConfig:
    def __init__(self):
        # DEFAULT
        self.limit_data: int | None = None
        self.reload_model = False  # startmodel from checkpoint
        self.prev_epochs = 0

        # GENERAL PARAMETERS
        self.batch_size = 100
        self.epochs = 5
        self.val_split = 0.1  # validation spli
        self.latent_dim = 50  # 100
        self.loss = "categorical_crossentropy"  # set reconstruction los

        # CONVOLUTION PARAMETERS
        self.batchnorm_conv = True
        # block includes Batch Norm, but BN can be disabled above.
        self.conv_layer_blocks = 4
        self.conv_filters = 8
        self.conv_kernel_size = 8
        self.conv_filter_growth_factor = 1.15875438383
        self.conv_kernel_growth_factor = 1.1758149644
        self.conv_activation = "tanh"

        # DECODER PARAMETERS
        self.gru_depth = 4
        self.rnn_activation = "tanh"
        self.recurrent_dim = 50
        self.do_tgru = False  # use custom terminal gru laye
        # use CPU intensive implementation; other implementation modes
        # (1 - GPU 2- memory) are not yet implemente
        self.terminal_GRU_implementation = 0
        self.tgru_dropout = 0.0

        # ENCODER -> MIDDLE LAYER PARAMETERS
        # growth factor applied to determine size of next middle layer
        self.hg_growth_factor = 1.4928245388
        self.middle_layer = 1
        self.dropout_rate_mid = 0.0
        self.batchnorm_mid = True  # apply batch normalization to middle layer
        self.activation = "tanh"

        # OPTIMIZATION PARAMETERS
        self.lr = 0.000312087049936
        self.momentum = 0.936948773087
        self.optim = "adam"  # optimizer to be use

        # VAE PARAMETERS
        self.vae_annealer_start = (
            22  # Center for variational weigh anneale. What is that?
        )
        self.batchnorm_vae = (
            False  # apply batch normalization to output of the variational laye
        )
        self.vae_activation = "tanh"
        self.xent_loss_weight = 1.0  # loss weight to assign to reconstruction error
        self.kl_loss_weight = 1.0  # loss weight to assing to KL los
        self.anneal_sigmod_slope = 1.0  # slope of sigmoid variational weight anneale
        # Choice of freezing the variational layer until close to the anneal starting epoc
        self.freeze_logvar_layer = False
        # the number of epochs before vae_annealer_start where the variational layer should be unfroze
        self.freeze_offset = 1

        # PROPERTY PREDICTION PARAMETERS
        self.reg_prop_tasks: list[str] | None = None
        self.logit_prop_tasks: list[str] | None = None
        # path to write out normalised regression data (or None)
        # ratio between consecutive layer in property pred.
        self.do_prop_pred = False  # whether to do property predictio
        self.prop_pred_depth = 3
        self.prop_hidden_dim = 36
        self.prop_growth_factor = 0.8
        self.prop_pred_activation = "tanh"
        # loss function to use with property prediction error for regression task
        self.reg_prop_pred_loss = "mse"
        # loss function to use with property prediction for logistic task
        self.logit_prop_pred_loss = "binary_crossentropy"
        self.prop_pred_loss_weight = 0.5
        self.prop_pred_dropout = 0.0
        self.prop_batchnorm = True
        # print output parameters
        self.verbose_print = 1
