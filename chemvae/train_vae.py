"""Train the variational autoencoder."""

import os
import time
from functools import partial  # function wrapper to pre-load some parameters.
from typing import cast, overload

import keras

# import argparse # we will put this back in the future.
import numpy as np
import tensorflow as tf
from keras import (
    Variable,
    ops,  # these are numpy operations.
)

# keras api
from keras.api.callbacks import CSVLogger
from keras.api.layers import Layer
from keras.api.models import Model
from keras.api.optimizers import SGD, Adam, RMSprop

# types aren't available in keras.api, but in .src
from keras.src.models.functional import Functional

# For padding right,left,none
from typing_extensions import Literal

# local modules
from . import mol_callbacks as mol_cb
from . import mol_utils as mu
from .hyperparameters.user import UserConfig, config
from .models import (
    decoder_model,
    encoder_model,
    load_decoder,
    load_encoder,
    load_property_predictor,
    property_predictor_model,
    sample_latent_vector,
)

# TODO: use multibackend.
os.environ["KERAS_BACKEND"] = "tensorflow"

print("".center(20, "#"))
print("HELLO!".center(20, "#"))
print("".center(20, "#"))
print(f"\n TF Version:{tf.__version__}\n")
print(f"\n KERAS Version:{keras.__version__}\n")

"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network (from the training callback.)

(see `./mol_callbacks.py`)

"""


class NameLayer(Layer):
    """Name layer between models.

    This doesn't have any computational meaning.
    It's like a `<div>`.

    :name: the name of the division layer.
    """

    def __init__(self, name: str):
        """Name the layer."""
        super().__init__(name=name)

    def call(self, x):
        """Return itself."""
        return x


@overload
def vectorize_data(
    config: UserConfig, do_prop_pred: Literal[True]
) -> tuple[np.ndarray, np.ndarray, list, list]: ...


@overload
def vectorize_data(
    config: UserConfig, do_prop_pred: Literal[False]
) -> tuple[np.ndarray, np.ndarray]: ...


def vectorize_data(
    config: UserConfig, do_prop_pred: bool = True
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list, list]:
    """Split the dataframe to: smiles_tensor[, prediction data]

    Returns:
      If true: (smiles_train, smiles_val, [reg_tasks_train, val], [log_tasks_t, v] )
      If false: (smiles_train, smiles_val)

    The smiles are hot-encoded to a 3-D tensor.
    It makes sure that it's multiple of batch size, and data is randomly selected.
    """
    MAX_LEN = config.MAX_LEN
    CHARS = config.CHARS
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))

    ## Load data for properties
    if do_prop_pred and config.data_normalization_out_file:
        normalize_out = config.data_normalization_out_file
    else:
        normalize_out = None

    ################ matches columns in csv file ###########
    reg_props = config.reg_prop_tasks  # list of names
    logit_props = config.logit_prop_tasks

    if do_prop_pred and not reg_props and not logit_props:
        raise ValueError("please especify logit and/or reg tasks")

    # here we get the csv-data split, and optionally a "normed write out" for reg data.
    smiles, Y_reg, Y_logit = mu.load_smiles_and_data_df(
        csv_file_path=config.data_file,  # smiles + data
        max_len=MAX_LEN,
        reg_tasks=reg_props,
        logit_tasks=logit_props,
        normalize_out=normalize_out,  # path to write normalised reg data out|None
    )

    # subset of data if needed.
    if config.limit_data:
        # sample indices within the range. The number collected is limit_data value.
        sample_idx = np.random.choice(
            np.arange(len(smiles)), config.limit_data, replace=False
        )
        smiles = list(np.array(smiles)[sample_idx])  # sublist of size config.limit_data
        if config.do_prop_pred and config.data_file:
            if Y_reg:
                Y_reg = Y_reg[sample_idx]  # basically the rows of the original DF.
            if Y_logit:
                Y_logit = Y_logit[sample_idx]

    print("Training set size is", len(smiles))
    print("total chars:", NCHARS)

    X = mu.smiles_to_hot(smiles, MAX_LEN, config.PADDING, CHAR_INDICES, NCHARS)
    # if less than the batch size the `//` gives 0.
    if X.shape[0] % config.batch_size != 0:
        # make it multiple of batch_size, discard excedent.
        to_length = X.shape[0] // config.batch_size * config.batch_size
        X = X[:to_length]
        if config.do_prop_pred:
            if Y_reg:
                Y_reg = Y_reg[:to_length]
            if Y_logit:
                Y_logit = Y_logit[:to_length]

    np.random.seed(config.RAND_SEED)
    rand_idx = np.arange(X.shape[0])
    np.random.shuffle(rand_idx)  # shuffles the rows' indices.

    TRAIN_FRAC = 1 - config.val_split
    num_train = int(X.shape[0] * TRAIN_FRAC)

    # or gets 0
    if num_train % config.batch_size != 0:
        num_train = num_train // config.batch_size * config.batch_size
        print("num_train ", num_train)

    # makes the indices for each
    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train) :]

    if config.test_idx_file:
        np.save(config.test_idx_file, test_idx)

    # grab the rows with an list of random indices.
    X_train, X_test = X[train_idx], X[test_idx]
    print(f"shape of training input vector : {X_train.shape}")

    if do_prop_pred:
        # !# add Y_train and Y_test here
        Y_train = []
        Y_test = []
        if Y_reg:
            Y_reg_train, Y_reg_test = Y_reg[train_idx], Y_reg[test_idx]
            Y_train.append(Y_reg_train)
            Y_test.append(Y_reg_test)
        if Y_logit:
            Y_logit_train, Y_logit_test = Y_logit[train_idx], Y_logit[test_idx]
            Y_train.append(Y_logit_train)
            Y_test.append(Y_logit_test)

        return X_train, X_test, Y_train, Y_test

    else:
        return X_train, X_test


@overload
def load_models(
    config: UserConfig, do_prop_pred: Literal[True]
) -> tuple[Functional, Functional, Functional, Functional, Functional, Variable]: ...


@overload
def load_models(
    config: UserConfig, do_prop_pred: Literal[False]
) -> tuple[Functional, Functional, Functional, Variable]: ...


def load_models(config: UserConfig, do_prop_pred: bool = False):
    """Load and create each model.

    config: configuration for networks and program.
    do_prop_pred: whether to use the property predictor and csv data file.

    """
    kl_loss_var = Variable(config.kl_loss_weight)
    if config.reload_model:
        encoder = load_encoder(config)
        decoder = load_decoder(config)
    else:  # new models
        encoder = encoder_model(config)
        decoder = decoder_model(config)

    # note that this are all symbolic operations
    # lists are not unpacked for 1 variable, so use [0]
    x_in = encoder.inputs[0]  # flat list of symbolic inputs (1 here.)

    # z_mean, z_log_var = encoder(x_in)
    z_mean, z_log_var = encoder.outputs  # simpler syntax

    # this is what makes it variational.
    z_samp, z_mean_z_log_var_output = sample_latent_vector(
        z_mean, z_log_var, kl_loss_var, config
    )

    # Decoder
    if config.do_tgru:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    x_out = NameLayer(name="x_pred")(x_out)  # decoder output
    model_outputs = [x_out, z_mean_z_log_var_output]

    AE_only_model = Model(x_in, model_outputs, name="AE_ONLY")

    if config.verbose_print:
        print("CONFIGURATION:")
        print("\n".join("%s : %s" % item for item in vars(config).items()))
        time.sleep(5)

        encoder.summary()
        decoder.summary()
        time.sleep(2)

        AE_only_model.summary()
        model_name = AE_only_model.name + ".png"
        time.sleep(2)

        print(f"plotting model to {model_name}.")
        keras.utils.plot_model(
            model=AE_only_model,
            to_file=model_name,
            show_dtype=True,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            expand_nested=True,
        )

    # extends the output
    if do_prop_pred:
        if config.reload_model:
            property_predictor = load_property_predictor(config)
        else:
            property_predictor = property_predictor_model(config)

        if (
            isinstance(config.reg_prop_tasks, list)
            and (len(config.reg_prop_tasks) > 0)
            and isinstance(config.logit_prop_tasks, list)
            and (len(config.logit_prop_tasks) > 0)
        ):
            reg_prop_pred, logit_prop_pred = property_predictor(z_mean)
            reg_prop_pred = NameLayer(name="reg_prop_pred")(reg_prop_pred)
            logit_prop_pred = NameLayer(name="logit_prop_pred")(logit_prop_pred)
            model_outputs.extend([reg_prop_pred, logit_prop_pred])

        # regression only scenario
        elif isinstance(config.reg_prop_tasks, list) and (
            len(config.reg_prop_tasks) > 0
        ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = NameLayer(name="reg_prop_pred")(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif (
            isinstance(config.logit_prop_tasks, list)
            and len(config.logit_prop_tasks) > 0
        ):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = NameLayer(name="logit_prop_pred")(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError(
                "no logit tasks or regression tasks specified for property prediction"
            )

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return (
            AE_only_model,
            AE_PP_model,
            encoder,
            decoder,
            property_predictor,
            kl_loss_var,
        )

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


def kl_loss(truth_dummy, x_mean_log_var_output):
    """Review this function."""
    x_mean, x_log_var = ops.split(x_mean_log_var_output, 2, axis=1)
    print("x_mean shape in kl_loss: ", x_mean.get_shape())
    kl_loss = -0.5 * ops.mean(
        1 + x_log_var - ops.square(x_mean) - ops.exp(x_log_var), axis=-1
    )
    print("KL LOSS IS: ", kl_loss)
    return kl_loss


def main_no_prop(config: UserConfig):
    start_time = time.time()
    if config.do_prop_pred:
        raise Exception(
            'expected do_prop_pred to be "false". Found {}'.format(config.do_prop_pred)
        )
    X_train, X_test = vectorize_data(config, config.do_prop_pred)
    # loads kl_loss_var from param.kl_loss_weight
    AE_only_model, encoder, decoder, kl_loss_var = load_models(
        config, config.do_prop_pred
    )

    # compile models
    if config.optim == "adam":
        optim = Adam(learning_rate=config.lr, beta_1=config.momentum)
    elif config.optim == "rmsprop":
        optim = RMSprop(leargning_rate=config.lr, rho=config.momentum)
    elif config.optim == "sgd":
        optim = SGD(learning_rate=config.lr, momentum=config.momentum)
    else:
        raise NotImplementedError("Please define valid optimizer")

    model_losses = {"x_pred": config.loss, "z_mean_z_log_var": kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(
        mol_cb.sigmoid_schedule,
        slope=config.anneal_sigmod_slope,
        start=config.vae_annealer_start,
    )
    # remains the epoch, that is used by the annealer.
    vae_anneal_callback = mol_cb.WeightAnnealerEpoch(
        vae_sig_schedule, kl_loss_var, config.kl_loss_weight, "vae"
    )

    csv_clb = CSVLogger(config.history_file, append=False)
    callbacks = [vae_anneal_callback, csv_clb]

    xent_loss_weight = Variable(config.xent_loss_weight, dtype=float)
    model_train_targets = {
        "x_pred": X_train,
        "z_mean_z_log_var": np.ones((X_train.shape[0], config.latent_dim * 2)),
    }
    model_test_targets = {
        "x_pred": X_test,
        "z_mean_z_log_var": np.ones((X_test.shape[0], config.latent_dim * 2)),
    }

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    keras_verbose = config.verbose_print

    AE_only_model.compile(
        loss=model_losses,
        loss_weights={
            "x_pred": xent_loss_weight.numpy().item(),
            "z_mean_z_log_var": kl_loss_var.numpy().item(),
        },
        optimizer=cast(str, optim),
        metrics={"x_pred": ["categorical_accuracy", vae_anneal_metric]},
    )
    if config.checkpoint_path:
        callbacks.append(
            mol_cb.EncoderDecoderCheckpoint(
                encoder,
                decoder,
                parameters=config,
                prop_pred_model=None,
                save_best_only=True,
            )
        )
    AE_only_model.fit(
        x=X_train,
        y=model_train_targets,
        batch_size=config.batch_size,
        epochs=config.epochs,
        initial_epoch=config.prev_epochs,
        callbacks=callbacks,
        verbose=str(keras_verbose),
        validation_data=[X_test, model_test_targets],
    )

    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")
    return


def main_property_run(parameters: UserConfig):
    start_time = time.time()

    if not parameters.do_prop_pred:
        raise Exception("expected parameters.do_prop_pred to be true")

    # load data
    X_train, X_test, Y_train, Y_test = vectorize_data(parameters, do_prop_pred=True)

    # load full models:
    (
        AE_only_model,
        AE_PP_model,
        encoder,
        decoder,
        property_predictor,
        kl_loss_var,
    ) = load_models(parameters, do_prop_pred=True)

    # compile models
    if parameters.optim == "adam":
        optim = Adam(learning_rate=parameters.lr, beta_1=parameters.momentum)
    elif parameters.optim == "rmsprop":
        optim = RMSprop(learning_rate=parameters.lr, rho=parameters.momentum)
    elif parameters.optim == "sgd":
        optim = SGD(learning_rate=parameters.lr, momentum=parameters.momentum)
    else:
        raise NotImplementedError("Please define valid optimizer")

    model_train_targets = {
        "x_pred": X_train,
        "z_mean_z_log_var": np.ones((np.shape(X_train)[0], parameters.latent_dim * 2)),
    }
    model_test_targets = {
        "x_pred": X_test,
        "z_mean_z_log_var": np.ones((np.shape(X_test)[0], parameters.latent_dim * 2)),
    }
    model_losses = {"x_pred": parameters.loss, "z_mean_z_log_var": kl_loss}

    xent_loss_weight = Variable(parameters.xent_loss_weight)
    ae_loss_weight = 1.0 - parameters.prop_pred_loss_weight
    model_loss_weights = {
        "x_pred": ae_loss_weight * xent_loss_weight,
        "z_mean_z_log_var": ae_loss_weight * kl_loss_var,
    }

    prop_pred_loss_weight = parameters.prop_pred_loss_weight

    if parameters.reg_prop_tasks and (len(parameters.reg_prop_tasks) > 0):
        model_train_targets["reg_prop_pred"] = Y_train[0]
        model_test_targets["reg_prop_pred"] = Y_test[0]
        model_losses["reg_prop_pred"] = parameters.reg_prop_pred_loss
        model_loss_weights["reg_prop_pred"] = prop_pred_loss_weight
    if parameters.logit_prop_tasks and (len(parameters.logit_prop_tasks) > 0):
        if parameters.reg_prop_tasks and (len(parameters.reg_prop_tasks) > 0):
            model_train_targets["logit_prop_pred"] = Y_train[1]
            model_test_targets["logit_prop_pred"] = Y_test[1]
        else:
            model_train_targets["logit_prop_pred"] = Y_train[0]
            model_test_targets["logit_prop_pred"] = Y_test[0]
        model_losses["logit_prop_pred"] = parameters.logit_prop_pred_loss
        model_loss_weights["logit_prop_pred"] = prop_pred_loss_weight

    # vae metrics, callbacks
    vae_sig_schedule = partial(
        mol_cb.sigmoid_schedule,
        slope=parameters.anneal_sigmod_slope,
        start=parameters.vae_annealer_start,
    )
    vae_anneal_callback = mol_cb.WeightAnnealerEpoch(
        vae_sig_schedule, kl_loss_var, parameters.kl_loss_weight, "vae"
    )

    csv_clb = CSVLogger(parameters.history_file, append=False)

    callbacks = [vae_anneal_callback, csv_clb]

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    # control verbose output
    keras_verbose = parameters.verbose_print

    if parameters.checkpoint_path:
        callbacks.append(
            mol_cb.EncoderDecoderCheckpoint(
                encoder,
                decoder,
                parameters=config,
                prop_pred_model=property_predictor,
                save_best_only=False,
            )
        )

    AE_PP_model.compile(
        loss=model_losses,
        loss_weights=model_loss_weights,
        optimizer=cast(str, optim),
        metrics={"x_pred": ["categorical_accuracy", vae_anneal_metric]},
    )

    AE_PP_model.fit(
        X_train,
        model_train_targets,
        batch_size=parameters.batch_size,
        epochs=parameters.epochs,
        initial_epoch=parameters.prev_epochs,
        callbacks=callbacks,
        verbose=str(keras_verbose),
        validation_data=[X_test, model_test_targets],
    )

    encoder.save(parameters.encoder_weights_file)
    decoder.save(parameters.decoder_weights_file)
    property_predictor.save(parameters.prop_pred_weights_file)

    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")

    return
