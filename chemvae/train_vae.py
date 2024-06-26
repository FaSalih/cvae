"""Train the variational autoencoder."""

import time
from functools import partial  # function wrapper to pre-load some parameters.
from typing import cast, overload

import keras

# import argparse # we will put this back in the future.
import numpy as np
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
from .default_config import Config
from .vectorize_data import vectorize_data
from .mol_callbacks import (
    sigmoid_schedule,
    WeightAnnealerEpoch,
    EncoderDecoderCheckpoint,
)
from .models import (
    decoder_model,
    encoder_model,
    load_decoder,
    load_encoder,
    load_property_predictor,
    property_predictor_model,
    sample_latent_vector,
)


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
def load_models(
    config: Config, do_prop_pred: Literal[True]
) -> tuple[Functional, Functional, Functional, Functional, Functional, Variable]: ...


@overload
def load_models(
    config: Config, do_prop_pred: Literal[False]
) -> tuple[Functional, Functional, Functional, Variable]: ...


def load_models(config: Config, do_prop_pred: bool = False):
    """Load and create each model.

    config: configuration for networks and program.
    do_prop_pred: whether to use the property predictor and csv data file.

    """
    kl_loss_var = Variable(config.kl_loss_weight, dtype=np.float32)
    if config.reload_model:
        encoder = load_encoder(config)
        decoder = load_decoder(config)
    else:  # new models
        encoder = encoder_model(config)
        decoder = decoder_model(config)

    x_in = encoder.inputs[0]  # flat list of symbolic inputs (1 here.)
    z_mean, enc_output = encoder(x_in)
    # this is what makes it variational.
    z_samp, z_mean_z_log_var_output = sample_latent_vector(
        z_mean, enc_output, kl_loss_var, config
    )

    # Decoder
    x_out = decoder(z_samp)

    x_out = NameLayer(name="x_pred")(x_out)  # decoder output
    model_outputs = [x_out, z_mean_z_log_var_output]

    AE_only_model = Model(x_in, model_outputs, name="AE_ONLY")

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
    """Loss function.

    x_mean_log_var_output: contains both."""

    x_mean, x_log_var = ops.split(x_mean_log_var_output, 2, axis=1)
    loss = -0.5 * ops.mean(
        1 + x_log_var - ops.square(x_mean) - ops.exp(x_log_var), axis=-1
    )
    return loss  # kl_loss


def main_no_prop(config: Config):
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

    if config.verbosity > 0:
        print("CONFIGURATION:")
        print("\n".join("%s : %s" % item for item in vars(config).items()))
        encoder.summary()
        decoder.summary()
        AE_only_model.summary()
        model_name = AE_only_model.name + ".png"

        print(f"Model schema saved to {model_name}.")
        keras.utils.plot_model(
            model=AE_only_model,
            to_file=model_name,
            show_dtype=True,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            expand_nested=True,
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

    # vae metrics, callbacks
    vae_sig_schedule = partial(
        sigmoid_schedule,
        slope=config.anneal_sigmod_slope,
        start=config.vae_annealer_start,
    )

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

    xent_loss_var = Variable(config.xent_loss_weight, dtype=np.float32)

    AE_only_model.compile(
        loss={"x_pred": config.loss, "z_mean_z_log_var": kl_loss},
        loss_weights={
            "x_pred": xent_loss_var.numpy().item(),
            "z_mean_z_log_var": kl_loss_var.numpy().item(),
        },
        optimizer=optim,
        metrics={"x_pred": ["categorical_accuracy", vae_anneal_metric]},
    )

    # CALLBACKS
    # remains the epoch, that is used by the annealer.
    vae_anneal_callback = WeightAnnealerEpoch(
        vae_sig_schedule, kl_loss_var, config.kl_loss_weight, "vae"
    )

    callbacks = [vae_anneal_callback, CSVLogger(config.history_file, append=False)]
    callbacks.append(
        EncoderDecoderCheckpoint(
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
        validation_data=[X_test, model_test_targets],
        batch_size=config.batch_size,
        epochs=config.epochs,
        initial_epoch=config.prev_epochs,
        callbacks=callbacks,
        verbose=str(config.verbosity),
    )

    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")
    return


# not important ATM
def main_property_run(config: Config):
    start_time = time.time()

    if not config.do_prop_pred:
        raise Exception("expected parameters.do_prop_pred to be true")

    # load data
    X_train, X_test, Y_train, Y_test = vectorize_data(config, do_prop_pred=True)

    # load full models:
    (
        AE_only_model,
        AE_PP_model,
        encoder,
        decoder,
        property_predictor,
        kl_loss_var,
    ) = load_models(config, do_prop_pred=True)

    # compile models
    if config.optim == "adam":
        optim = Adam(learning_rate=config.lr, beta_1=config.momentum)
    elif config.optim == "rmsprop":
        optim = RMSprop(learning_rate=config.lr, rho=config.momentum)
    elif config.optim == "sgd":
        optim = SGD(learning_rate=config.lr, momentum=config.momentum)
    else:
        raise NotImplementedError("Please define valid optimizer")

    model_train_targets = {
        "x_pred": X_train,
        "z_mean_z_log_var": np.ones((np.shape(X_train)[0], config.latent_dim * 2)),
    }
    model_test_targets = {
        "x_pred": X_test,
        "z_mean_z_log_var": np.ones((np.shape(X_test)[0], config.latent_dim * 2)),
    }
    model_losses = {"x_pred": config.loss, "z_mean_z_log_var": kl_loss}

    xent_loss_weight = Variable(config.xent_loss_weight)
    ae_loss_weight = 1.0 - config.prop_pred_loss_weight
    model_loss_weights = {
        "x_pred": ae_loss_weight * xent_loss_weight,
        "z_mean_z_log_var": ae_loss_weight * kl_loss_var,
    }

    prop_pred_loss_weight = config.prop_pred_loss_weight

    if config.reg_prop_tasks and (len(config.reg_prop_tasks) > 0):
        model_train_targets["reg_prop_pred"] = Y_train[0]
        model_test_targets["reg_prop_pred"] = Y_test[0]
        model_losses["reg_prop_pred"] = config.reg_prop_pred_loss
        model_loss_weights["reg_prop_pred"] = prop_pred_loss_weight
    if config.logit_prop_tasks and (len(config.logit_prop_tasks) > 0):
        if config.reg_prop_tasks and (len(config.reg_prop_tasks) > 0):
            model_train_targets["logit_prop_pred"] = Y_train[1]
            model_test_targets["logit_prop_pred"] = Y_test[1]
        else:
            model_train_targets["logit_prop_pred"] = Y_train[0]
            model_test_targets["logit_prop_pred"] = Y_test[0]
        model_losses["logit_prop_pred"] = config.logit_prop_pred_loss
        model_loss_weights["logit_prop_pred"] = prop_pred_loss_weight

    # vae metrics, callbacks
    vae_sig_schedule = partial(
        sigmoid_schedule,
        slope=config.anneal_sigmod_slope,
        start=config.vae_annealer_start,
    )
    vae_anneal_callback = WeightAnnealerEpoch(
        vae_sig_schedule, kl_loss_var, config.kl_loss_weight, "vae"
    )

    csv_clb = CSVLogger(config.history_file, append=False)

    callbacks = [vae_anneal_callback, csv_clb]

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    if config.checkpoint_path:
        callbacks.append(
            EncoderDecoderCheckpoint(
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
        batch_size=config.batch_size,
        epochs=config.epochs,
        initial_epoch=config.prev_epochs,
        callbacks=callbacks,
        verbose=str(config.verbosity),
        validation_data=[X_test, model_test_targets],
    )

    encoder.save(config.encoder_weights_file)
    decoder.save(config.decoder_weights_file)
    property_predictor.save(config.prop_pred_weights_file)

    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")

    return
