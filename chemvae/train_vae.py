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
from .hyperparameters import Config, params
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
    params: Config, do_prop_pred: Literal[True]
) -> tuple[np.ndarray, np.ndarray, list, list]: ...


@overload
def vectorize_data(
    params: Config, do_prop_pred: Literal[False]
) -> tuple[np.ndarray, np.ndarray]: ...


def vectorize_data(
    params: Config, do_prop_pred: bool = True
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list, list]:
    """Split the dataframe to: smiles_tensor[, prediction data]

    Returns:
      If true: (smiles_train, smiles_val, [reg_tasks_train, val], [log_tasks_t, v] )
      If false: (smiles_train, smiles_val)

    The smiles are hot-encoded to a 3-D tensor.
    It makes sure that it's multiple of batch size, and data is randomly selected.
    """
    MAX_LEN = params.MAX_LEN
    CHARS = params.CHARS
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))

    ## Load data for properties
    if do_prop_pred and params.data_normalization_out_file:
        normalize_out = params.data_normalization_out_file
    else:
        normalize_out = None

    ################ matches columns in csv file ###########
    reg_props = params.reg_prop_tasks  # list of names
    logit_props = params.logit_prop_tasks

    if do_prop_pred and not reg_props and not logit_props:
        raise ValueError("please especify logit and/or reg tasks")

    # here we get the csv-data split, and optionally a "normed write out" for reg data.
    smiles, Y_reg, Y_logit = mu.load_smiles_and_data_df(
        csv_file_path=params.data_file,  # smiles + data
        max_len=MAX_LEN,
        reg_tasks=reg_props,
        logit_tasks=logit_props,
        normalize_out=normalize_out,  # path to write normalised reg data out|None
    )

    # subset of data if needed.
    if params.limit_data:
        # sample indices within the range. The number collected is limit_data value.
        sample_idx = np.random.choice(
            np.arange(len(smiles)), params.limit_data, replace=False
        )
        smiles = list(np.array(smiles)[sample_idx])  # sublist of size params.limit_data
        if params.do_prop_pred and params.data_file:
            if Y_reg:
                Y_reg = Y_reg[sample_idx]  # basically the rows of the original DF.
            if Y_logit:
                Y_logit = Y_logit[sample_idx]

    print("Training set size is", len(smiles))
    print("total chars:", NCHARS)

    X = mu.smiles_to_hot(smiles, MAX_LEN, params.PADDING, CHAR_INDICES, NCHARS)
    # if less than the batch size the `//` gives 0.
    if X.shape[0] % params.batch_size != 0:
        # make it multiple of batch_size, discard excedent.
        to_length = X.shape[0] // params.batch_size * params.batch_size
        X = X[:to_length]
        if params.do_prop_pred:
            if Y_reg:
                Y_reg = Y_reg[:to_length]
            if Y_logit:
                Y_logit = Y_logit[:to_length]

    np.random.seed(params.RAND_SEED)
    rand_idx = np.arange(X.shape[0])
    np.random.shuffle(rand_idx)  # shuffles the rows' indices.

    TRAIN_FRAC = 1 - params.val_split
    num_train = int(X.shape[0] * TRAIN_FRAC)

    # or gets 0
    if num_train % params.batch_size != 0:
        num_train = num_train // params.batch_size * params.batch_size
        print("num_train ", num_train)

    # makes the indices for each
    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train) :]

    if params.test_idx_file:
        np.save(params.test_idx_file, test_idx)

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
    params: Config, do_prop_pred: Literal[True]
) -> tuple[Functional, Functional, Functional, Functional, Functional, Variable]: ...


@overload
def load_models(
    params: Config, do_prop_pred: Literal[False]
) -> tuple[Functional, Functional, Functional, Variable]: ...


def load_models(params: Config, do_prop_pred: bool = False):
    """Load and create each model.

    params: configuration for networks and program.
    do_prop_pred: whether to use the property predictor and csv data file.

    """
    kl_loss_var = Variable(params.kl_loss_weight)
    if params.reload_model:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:  # new models
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    # note that this are all symbolic operations
    # lists are not unpacked for 1 variable, so use [0]
    x_in = encoder.inputs[0]  # flat list of symbolic inputs (1 here.)

    # z_mean, z_log_var = encoder(x_in)
    z_mean, z_log_var = encoder.outputs  # simpler syntax

    # this is what makes it variational.
    z_samp, z_mean_z_log_var_output = sample_latent_vector(
        z_mean, z_log_var, kl_loss_var, params
    )

    # Decoder
    if params.do_tgru:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    x_out = NameLayer(name="x_pred")(x_out)  # decoder output
    model_outputs = [x_out, z_mean_z_log_var_output]

    AE_only_model = Model(x_in, model_outputs, name="AE_ONLY")

    if params.verbose_print:
        print("CONFIGURATION:")
        print("\n".join("%s : %s" % item for item in vars(params).items()))
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
        if params.reload_model:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (
            isinstance(params.reg_prop_tasks, list)
            and (len(params.reg_prop_tasks) > 0)
            and isinstance(params.logit_prop_tasks, list)
            and (len(params.logit_prop_tasks) > 0)
        ):
            reg_prop_pred, logit_prop_pred = property_predictor(z_mean)
            reg_prop_pred = NameLayer(name="reg_prop_pred")(reg_prop_pred)
            logit_prop_pred = NameLayer(name="logit_prop_pred")(logit_prop_pred)
            model_outputs.extend([reg_prop_pred, logit_prop_pred])

        # regression only scenario
        elif isinstance(params.reg_prop_tasks, list) and (
            len(params.reg_prop_tasks) > 0
        ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = NameLayer(name="reg_prop_pred")(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif (
            isinstance(params.logit_prop_tasks, list)
            and len(params.logit_prop_tasks) > 0
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


def main_no_prop(params: Config):
    start_time = time.time()
    if params.do_prop_pred:
        raise Exception(
            'expected do_prop_pred to be "false". Found {}'.format(params.do_prop_pred)
        )
    X_train, X_test = vectorize_data(params, params.do_prop_pred)
    # loads kl_loss_var from param.kl_loss_weight
    AE_only_model, encoder, decoder, kl_loss_var = load_models(
        params, params.do_prop_pred
    )

    # compile models
    if params.optim == "adam":
        optim = Adam(learning_rate=params.lr, beta_1=params.momentum)
    elif params.optim == "rmsprop":
        optim = RMSprop(leargning_rate=params.lr, rho=params.momentum)
    elif params.optim == "sgd":
        optim = SGD(learning_rate=params.lr, momentum=params.momentum)
    else:
        raise NotImplementedError("Please define valid optimizer")

    model_losses = {"x_pred": params.loss, "z_mean_z_log_var": kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(
        mol_cb.sigmoid_schedule,
        slope=params.anneal_sigmod_slope,
        start=params.vae_annealer_start,
    )
    # remains the epoch, that is used by the annealer.
    vae_anneal_callback = mol_cb.WeightAnnealerEpoch(
        vae_sig_schedule, kl_loss_var, params.kl_loss_weight, "vae"
    )

    csv_clb = CSVLogger(params.history_file, append=False)
    callbacks = [vae_anneal_callback, csv_clb]

    xent_loss_weight = Variable(params.xent_loss_weight, dtype=float)
    model_train_targets = {
        "x_pred": X_train,
        "z_mean_z_log_var": np.ones((X_train.shape[0], params.latent_dim * 2)),
    }
    model_test_targets = {
        "x_pred": X_test,
        "z_mean_z_log_var": np.ones((X_test.shape[0], params.latent_dim * 2)),
    }

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    keras_verbose = params.verbose_print

    AE_only_model.compile(
        loss=model_losses,
        loss_weights={
            "x_pred": xent_loss_weight.numpy().item(),
            "z_mean_z_log_var": kl_loss_var.numpy().item(),
        },
        optimizer=cast(str, optim),
        metrics={"x_pred": ["categorical_accuracy", vae_anneal_metric]},
    )
    if params.checkpoint_path:
        callbacks.append(
            mol_cb.EncoderDecoderCheckpoint(
                encoder,
                decoder,
                params=params,
                prop_pred_model=None,
                save_best_only=True,
            )
        )
    AE_only_model.fit(
        x=X_train,
        y=model_train_targets,
        batch_size=params.batch_size,
        epochs=params.epochs,
        initial_epoch=params.prev_epochs,
        callbacks=callbacks,
        verbose=str(keras_verbose),
        validation_data=[X_test, model_test_targets],
    )

    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")
    return


def main_property_run(params: Config):
    start_time = time.time()

    if not params.do_prop_pred:
        raise Exception("expected params.do_prop_pred to be true")

    # load data
    X_train, X_test, Y_train, Y_test = vectorize_data(params, do_prop_pred=True)

    # load full models:
    (
        AE_only_model,
        AE_PP_model,
        encoder,
        decoder,
        property_predictor,
        kl_loss_var,
    ) = load_models(params, do_prop_pred=True)

    # compile models
    if params.optim == "adam":
        optim = Adam(learning_rate=params.lr, beta_1=params.momentum)
    elif params.optim == "rmsprop":
        optim = RMSprop(learning_rate=params.lr, rho=params.momentum)
    elif params.optim == "sgd":
        optim = SGD(learning_rate=params.lr, momentum=params.momentum)
    else:
        raise NotImplementedError("Please define valid optimizer")

    model_train_targets = {
        "x_pred": X_train,
        "z_mean_z_log_var": np.ones((np.shape(X_train)[0], params.latent_dim * 2)),
    }
    model_test_targets = {
        "x_pred": X_test,
        "z_mean_z_log_var": np.ones((np.shape(X_test)[0], params.latent_dim * 2)),
    }
    model_losses = {"x_pred": params.loss, "z_mean_z_log_var": kl_loss}

    xent_loss_weight = Variable(params.xent_loss_weight)
    ae_loss_weight = 1.0 - params.prop_pred_loss_weight
    model_loss_weights = {
        "x_pred": ae_loss_weight * xent_loss_weight,
        "z_mean_z_log_var": ae_loss_weight * kl_loss_var,
    }

    prop_pred_loss_weight = params.prop_pred_loss_weight

    if params.reg_prop_tasks and (len(params.reg_prop_tasks) > 0):
        model_train_targets["reg_prop_pred"] = Y_train[0]
        model_test_targets["reg_prop_pred"] = Y_test[0]
        model_losses["reg_prop_pred"] = params.reg_prop_pred_loss
        model_loss_weights["reg_prop_pred"] = prop_pred_loss_weight
    if params.logit_prop_tasks and (len(params.logit_prop_tasks) > 0):
        if params.reg_prop_tasks and (len(params.reg_prop_tasks) > 0):
            model_train_targets["logit_prop_pred"] = Y_train[1]
            model_test_targets["logit_prop_pred"] = Y_test[1]
        else:
            model_train_targets["logit_prop_pred"] = Y_train[0]
            model_test_targets["logit_prop_pred"] = Y_test[0]
        model_losses["logit_prop_pred"] = params.logit_prop_pred_loss
        model_loss_weights["logit_prop_pred"] = prop_pred_loss_weight

    # vae metrics, callbacks
    vae_sig_schedule = partial(
        mol_cb.sigmoid_schedule,
        slope=params.anneal_sigmod_slope,
        start=params.vae_annealer_start,
    )
    vae_anneal_callback = mol_cb.WeightAnnealerEpoch(
        vae_sig_schedule, kl_loss_var, params.kl_loss_weight, "vae"
    )

    csv_clb = CSVLogger(params.history_file, append=False)

    callbacks = [vae_anneal_callback, csv_clb]

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    # control verbose output
    keras_verbose = params.verbose_print

    if params.checkpoint_path:
        callbacks.append(
            mol_cb.EncoderDecoderCheckpoint(
                encoder,
                decoder,
                params=params,
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
        batch_size=params.batch_size,
        epochs=params.epochs,
        initial_epoch=params.prev_epochs,
        callbacks=callbacks,
        verbose=str(keras_verbose),
        validation_data=[X_test, model_test_targets],
    )

    encoder.save(params.encoder_weights_file)
    decoder.save(params.decoder_weights_file)
    property_predictor.save(params.prop_pred_weights_file)

    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")

    return


# __name__ is set when executed from cli.
if __name__ == "__main__":
    # print("All params:", params)

    if params.do_prop_pred:
        main_property_run(params)
    else:
        print("no property pred.")
        main_no_prop(params)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--exp_file',
    #                     help='experiment file', default='exp.json')
    # parser.add_argument('-d', '--directory',
    #                     help='exp directory', default=None)
    # args = vars(parser.parse_args())
    # if args['directory'] is not None:

    # curdir = os.getcwd()
    # os.chdir(args['directory'])
    # args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    # os.chdir(curdir)
