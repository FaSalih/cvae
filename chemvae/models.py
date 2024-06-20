"""Encoder,Decoder, and Predictor."""

from typing import cast

import keras
from keras import Variable, ops
from keras.api.layers import (
    GRU,
    BatchNormalization,
    Concatenate,
    Convolution1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    RepeatVector,
)
from keras.api.models import Model, load_model
from keras.src.models import Functional

from .hyperparameters import Config
from .tgru_k2_gpu import TerminalGRU


# =============================
# Encoder functions
# =============================
def encoder_model(params: Config) -> Functional:
    """Define hot-encoded smile to vector processing.

    params: program and network configuration.
    Returns: encoder model

    1. Conv Blocks
    2. Flatten
    3. Dense Blocks
    4. middle (nn 1)
    5. middle->Dense Layer-> z (nn 2)

    """

    # sanity check
    if params.middle_layer < 0:
        raise Exception("params.middle_layer must be >=0")

    # smiles is the batch dimension.
    # starts the symbolic descriotion
    x_in = Input(shape=(params.MAX_LEN, params.NCHARS), name="input_molecule_smi")

    # Convolution layers
    # removed some code here, condensed below

    x = x_in  # they seem to do it in keras docs as well.

    # 1D Convolutions only need the n rows specified.
    # What happens if kernel_size grows > input rows ???
    for j in range(params.conv_layer_blocks):
        x = Convolution1D(
            filters=int(
                params.conv_filters * params.conv_filter_growth_factor ** (j or 1)
            ),
            kernel_size=int(
                params.conv_kernel_size * params.conv_kernel_growth_factor ** (j or 1)
            ),
            activation="tanh",
            name="encoder_conv{}".format(j),
        )(x)
        if params.batchnorm_conv:
            x = BatchNormalization(axis=-1, name="encoder_norm{}".format(j))(x)

    x = Flatten()(x)  # flatten in order to use the Dense layers.
    # from here to the output, 1D vector (+ batch dimension)

    # Middle layers
    # Dense->Dropout->Batchnorm pattern.
    shared = x

    for i in range(params.middle_layer):
        # note that this shinks, it's a bottleneck
        shared = Dense(
            units=int(  # a fn of the latent dim
                params.latent_dim * params.hg_growth_factor ** (params.middle_layer - i)
            ),
            activation=params.activation,
            name="encoder_dense{}".format(i),
        )(x)
        if params.dropout_rate_mid > 0:
            shared = Dropout(params.dropout_rate_mid)(shared)
        if params.batchnorm_mid:
            shared = BatchNormalization(axis=-1, name=f"encoder_dense{i}_norm")(shared)

    z_mean = Dense(units=params.latent_dim, name="z_mean")(shared)
    z_log_var = Dense(units=params.latent_dim, name="z_log_var")(shared)

    # return z & previous encoding layer for std dev sampling
    return Model(x_in, [z_mean, z_log_var], name="encoder")


def load_encoder(params: Config):
    """Reload a saved encoder."""
    return cast(Functional, load_model(params.encoder_weights_file))


##====================
## Middle part (var)
##====================


def sample_latent_vector(z_mean, z_log_var, kl_loss_var: Variable, params: Config):
    """Create variational layers. Adds noise to z.

    z_mean: from encoder's Dense()(z).
    z_log_var: forks into z_mean and z_mean_log_var
    kl_loss_var: ...
    params: parameter dictionary passed throughout entire model.
    """
    # why concatenate?
    z_mean_z_log_var_output = Concatenate(name="z_mean_z_log_var")([z_mean, z_log_var])

    z_samp = Sampling(
        kl_loss_var=kl_loss_var, rand_seed=params.RAND_SEED, name="z_samp"
    )([z_mean, z_log_var])

    if params.batchnorm_vae:
        z_samp = BatchNormalization(axis=-1)(z_samp)

    return z_samp, z_mean_z_log_var_output


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a smile."""

    def __init__(self, kl_loss_var: float, rand_seed: int, **kwargs):
        """Configure Sampling.

        kl_loss_var: a float that will tend to zero as epochs pass.
        It is recalculated on each epoch end.

        rand_seed: random seed
        """
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(rand_seed)
        self.kl_loss_var = kl_loss_var

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # changed: this was normal variable
        epsilon = keras.random.normal(
            shape=ops.shape(z_mean), mean=0.0, stddev=1.0, seed=self.seed_generator
        )

        return z_mean + ops.exp(0.5 * z_log_var) * self.kl_loss_var * epsilon

    def compute_output_shape(self, input_shape):
        """Compute shape. Keras isn't able to infer it."""
        return input_shape[0]


# ===========================================
# Decoder functions
# ===========================================


def decoder_model(params: Config) -> Functional:
    """Create decoder model. Retrieves a SMILES tensor from vector."""

    # sanity checks
    if params.middle_layer < 0:
        raise Exception("params.middle_layer must be >=0")
    elif not isinstance(params.gru_depth, int):
        raise Exception("params.gru_depth must be an integer>0.")
    elif params.gru_depth < 1 and not params.do_tgru:
        raise Exception("Either gru depth should be > 1 or do_tgru params be true.")

    z_in = Input(shape=(params.latent_dim,), name="decoder_input")

    # in the case we use TerminalGRU
    true_seq_in = Input(
        shape=(params.MAX_LEN, params.NCHARS), name="decoder_true_seq_input"
    )

    z = z_in

    for i in range(params.middle_layer):
        # this is un-bottleneck / expand vector.
        z = Dense(
            units=int(params.latent_dim * params.hg_growth_factor ** (i)),
            activation=params.activation,
            name=f"decoder_dense{i}",
        )(z)
        if params.dropout_rate_mid > 0:
            z = Dropout(rate=params.dropout_rate_mid)(z)
        if params.batchnorm_mid:
            z = BatchNormalization(axis=-1, name=f"decoder_dense{i}_norm")(z)

    # Necessary for using GRU vectors
    # expand to original row-size
    z_reps = RepeatVector(n=params.MAX_LEN)(z)

    # gru_depth includes tgru if true.
    # that way we can compare easily.
    x_out = z_reps
    if params.gru_depth >= 2:
        for i in range(params.gru_depth - 1):  # if 2, runs once
            x_out = GRU(
                units=params.recurrent_dim,
                return_sequences=True,
                activation="tanh",
                name=f"decoder_gru{i}",
            )(x_out)

    if params.do_tgru:
        x_out = TerminalGRU(
            units=params.NCHARS,
            rnd_seed=params.RAND_SEED,
            recurrent_dropout=params.tgru_dropout,
            return_sequences=True,
            activation="softmax",
            temperature=0.01,
            name="decoder_tgru",
            implementation=params.terminal_GRU_implementation,
        )([x_out, true_seq_in])
        return Model([z_in, true_seq_in], x_out, name="decoder")
    else:
        x_out = GRU(
            units=params.NCHARS,
            return_sequences=True,
            activation="softmax",
            name="decoder_gru_out",
        )(x_out)
        print(x_out[0])
        return Model(z_in, x_out, name="decoder")


def load_decoder(params: Config):
    """Load decoder model weights file."""
    if params.do_tgru:
        custom_objects = {"TerminalGRU": TerminalGRU}
        with keras.utils.custom_object_scope(custom_objects):
            return cast(
                Functional, keras.models.load_model(params.decoder_weights_file)
            )
    else:
        return cast(Functional, load_model(params.decoder_weights_file))


# ====================
# Property Prediction
# ====================


def property_predictor_model(params: Config) -> Functional:
    """One or two independent property predictors.

    params: Program and Network configuration.

    Composed of: Dense Blocks.
    They share the inner layers, but not the out layer.
    """
    reg_prop_tasks: list[str] | None = params.reg_prop_tasks
    logit_prop_tasks: list[str] | None = params.logit_prop_tasks

    # the input to this model is z-dim, not including the batch dimension.
    ls_in = Input(shape=(params.latent_dim,), name="prop_pred_input")

    prop_mid = ls_in
    # shared weights part
    for p_i in range(params.prop_pred_depth):
        prop_mid = Dense(
            int(params.prop_hidden_dim * params.prop_growth_factor**p_i),
            activation=params.prop_pred_activation,
            name=f"property_predictor_dense{p_i}",
        )(prop_mid)
        if params.prop_pred_dropout > 0:
            prop_mid = Dropout(params.prop_pred_dropout)(prop_mid)
        if params.prop_batchnorm:
            prop_mid = BatchNormalization()(prop_mid)

    # for regression tasks
    logit_prop_pred = None
    reg_prop_pred = None

    if isinstance(reg_prop_tasks, list) and len(reg_prop_tasks) > 0:
        reg_prop_pred = Dense(
            units=len(reg_prop_tasks),
            activation="linear",  # linear for the output, since may be -
            name="reg_property_output",
        )(prop_mid)

    # for logistic tasks
    if isinstance(logit_prop_tasks, list) and len(logit_prop_tasks) > 0:
        logit_prop_pred = Dense(
            units=len(logit_prop_tasks),
            activation="sigmoid",  # may be better softmax
            name="logit_property_output",
        )(prop_mid)

    # both regression and logistic
    if logit_prop_pred and reg_prop_pred:
        return Model(ls_in, [reg_prop_pred, logit_prop_pred], name="property_predictor")

        # regression only scenario
    elif reg_prop_pred:
        return Model(ls_in, reg_prop_pred, name="property_predictor")

        # logit only scenario
    elif logit_prop_pred:
        return Model(ls_in, logit_prop_pred, name="property_predictor")

    raise Exception(
        "No model was created. To run property prediction add the tasks' list in the configuration."
    )


def load_property_predictor(params):
    """Load the property predictor network from file path."""
    return cast(Functional, load_model(params.prop_pred_weights_file))
