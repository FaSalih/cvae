"""Callbacks for the training process."""

from pathlib import Path
from typing import Callable  # function types

import numpy as np
from keras import Variable  # or from keras.api.backend
from keras.api.callbacks import Callback, ModelCheckpoint
from keras.src.models.functional import Functional  # just the type.

from .hyperparameters import Config  # out set up configuration.

Schedule = Callable[[int], float]  # call back for the training.


class WeightAnnealerEpoch(Callback):
    """Weight of VAE's scheduler.

    Class called `on_epoch_begin` during training.

    Currently just adjust kl weight, will keep xent weight constant
    """

    def __init__(
        self,
        schedule: Schedule,
        kl_loss_var: Variable,
        param_kl_loss_weight: float,
        weight_name: str,
    ):
        """Configuration.

        schedule: a function that takes an epoch index as input
        and returns a new weight for the VAE (float).
        kl_loss_var: The variable to be updated.
        param_kl_loss_weight: The original weight, from the parameters file.
        weight_name: friendly identified for printing.
        """
        super().__init__()  # inits Callback's constructor.
        self.schedule = schedule  # takes epoch, returns new weight
        self.weight_var = kl_loss_var
        self.weight_orig = param_kl_loss_weight
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch: int, logs=None):
        """Make new numeric weight (float) for the VAE."""
        print("LOGS IS:", logs)
        if logs is None:
            logs = {}
        new_weight = self.schedule(epoch)  #
        new_value = new_weight * self.weight_orig
        print("Current {} annealer weight is {}".format(self.weight_name, new_value))
        print("weight_var: ", self.weight_var.value)
        # orig is a parameter, and new weight / weight var
        # changes with the epoch.
        self.weight_var.assign(new_value)


# Schedules for VAEWeightAnnealer
def no_schedule(epoch_num) -> float:
    return 1.0


def sigmoid_schedule(time_step: float | int, start: float, slope=1.0) -> float:
    """Make new kl weight.

    This function be passed with partial.
    time_step: epoch/index.
    start: param from config (fixed val)
    slope: param from config (fixed val)
    convert back from numpy float to float

    After a few epochs this function will increase from a very small value to 1.0
    """
    return float(1.0 / (1.0 + np.exp(slope * (start - time_step))))


class EncoderDecoderCheckpoint(ModelCheckpoint):
    """Adapted from ModelCheckpoint, but for saving Encoder, Decoder and Property predictor."""

    def __init__(
        self,
        encoder_model: Functional,
        decoder_model: Functional,
        params: Config,
        prop_pred_model=None,
        prop_to_monitor="val_x_pred_categorical_accuracy",
        save_best_only=True,
        monitor_op=np.greater,
        monitor_best_init=-np.Inf,
    ):
        """Save models at the end of every epoch if better than previous models.

        encoder_model: the .Model encoder architecture
        decoder_model: the .Model decoder architecture
        params: program, data, and network configuration.
        prop_pred_model: the .Model prop_pred architecture

        Next 4 arguments are very much related (to saving checkpoints):

        prop_to_monitor: a property that is a valid name in the model.
        Where do these properties come from ? From the training epoch-logs?
        Where do we specify which metrics the training tracks/monitors?
        To which of the models is it referred?
        save_best_only: whether to compare to previous or save all checkpoints.
        monitor_op: The operation to use when monitoring the property
        (e.g. accuracy to be maximized so use `np.greater`,
        loss to minimized, so use `np.less`)
        monitor_best_init: starting point for monitor
        (use -np.Inf for maximization tests, and np.Inf for minimization tests)
        """
        self.p = params

        super().__init__("lie.keras")  # directory path; models within.
        self.save_best_only = save_best_only
        self.monitor = prop_to_monitor
        self.monitor_op = monitor_op
        self.best = monitor_best_init
        self.verbose = 1
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.prop_pred_model = prop_pred_model
        self.save_path = Path(self.p.checkpoint_path)
        print("initialized checkpoint")

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        print("LOGS:", logs)
        current = logs.get(self.monitor)
        print(current)
        if not isinstance(current, float):
            raise Exception(f"Current should be a float, found {type(current)}")

        if self.save_best_only:
            # safety check
            # get "monitored property" logs.
            # initially, whether the accuracy is larger than -inf.
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    # up to 5 digits, up to 5 decimals, adds 0s otherwise.
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f,"
                        " saving model" % (epoch, self.monitor, self.best, current)
                    )
                self.best = current
                # now saving keras models + use value of "monitored property"
                self.encoder.save(
                    self.save_path.joinpath(
                        "encoder_{}_{}.keras".format(epoch, current)
                    )
                )
                self.decoder.save(
                    self.save_path.joinpath(
                        "decoder_{}_{}.keras".format(epoch, current)
                    )
                )
                if self.prop_pred_model is not None:
                    self.prop_pred_model.save(
                        self.save_path.joinpath(
                            "prop_pred_{}_{}.keras".format(epoch, current)
                        )
                    )
            else:
                if self.verbose > 0:
                    print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print("Epoch %05d: saving model to " % (epoch))
            self.encoder.save(
                self.save_path.joinpath("encoder_{}_{}.keras".format(epoch, current))
            )
            self.decoder.save(
                self.save_path.joinpath("decoder_{}_{}.keras".format(epoch, current))
            )
            if self.prop_pred_model is not None:
                self.prop_pred_model.save(
                    self.save_path, "prop_pred_{}_{}.keras".format(epoch, current)
                )


# class RmseCallback(Callback):
#     """
#     This is executed in this fashion:

#     model.fit(np.array([[1.0]]), np.array([[1.0]]),
#            callbacks=[MyCallback()])

#     Is there a way to get `self.model` type ?

#     This fn isn't actually used
#     """
#     def __init__(self, X_test, Y_test, params, df_norm: pd.DataFrame|None = None):
#         super().__init__()
#         self.df_norm = df_norm
#         self.X_test = X_test
#         self.Y_test = Y_test
#         self.config = params

#     def on_epoch_end(self, epoch, logs=None):
#         df_norm = self.df_norm
#         X_test = self.X_test
#         Y_test = self.Y_test

#         if isinstance(self.model, Model):
#             y_pred = self.model.predict(X_test,self.config['batch_size'])
#         if type(y_pred) is list:
#             if 'reg_prop_tasks' in self.config and 'logit_prop_tasks' in self.config:
#                 y_pred = y_pred[-2]
#             elif 'reg_prop_tasks' in self.config:
#                 y_pred = y_pred[-1]
#         if df_norm is not None:
#             std = df_norm['std'].values
#             mean =df_norm['mean'].values
#             # unsure how to especify the type
#             y_pred = y_pred *  std + mean #un norm the predictions
#             Y_test = Y_test * std + mean # and the tests.

#         rmse = np.sqrt(np.mean(np.square(y_pred - Y_test), axis=0))
#         mae = np.mean(np.abs(y_pred - Y_test), axis=0)
#         if df_norm is not None:
#             df_norm['rmse'] = rmse
#             df_norm['mae'] = mae
#             print("RMSE test set:", df_norm['rmse'].to_dict())
#             print("MAE test set:", df_norm['mae'].to_dict())
#         else:
#             if "reg_prop_tasks" in self.config:
#                 print("RMSE test set:", self.config["reg_prop_tasks"], rmse)
#                 print("MAE test set:", self.config["reg_prop_tasks"], mae)
#             else:
#                 print("RMSE test set:", rmse)
#                 print("MAE test set:", mae)

# def sample(a, temperature=0.01):
#     """Sample from the list of probabilities (derived from `a`).

#     This function does not seem to be used anywhere.
#     a: list of logits.
#     Think of `a` the probability of each outcome, for dice with len(a) faces
#     [p_0, p_1, p_2,...]
#     temperature: a modulator.

#     In practice this will likely be used to select the dice's face (mol) with that statistics.
#     """
#     a = np.log(a) / temperature
#     a = np.exp(a) / np.sum(np.exp(a))  # probabilities
#     return np.argmax(np.random.multinomial(1, a, 1))  # rolls the dice 1 time.
