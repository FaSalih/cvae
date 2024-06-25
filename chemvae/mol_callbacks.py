"""Callbacks for the training process."""

from pathlib import Path
from typing import Callable  # function types

import numpy as np
from keras import Variable  # or from keras.api.backend
from keras.api.callbacks import Callback
from keras.src.models.functional import Functional  # just the type.

from .default_config import Config  # out set up configuration.

Schedule = Callable[[int], float]  # call back for the training.


class WeightAnnealerEpoch(Callback):
    """Adjust kl weight.

    Called `on_epoch_begin` during training.
    """

    def __init__(
        self,
        schedule: Schedule,
        kl_loss_var: Variable,
        param_kl_loss_weight: float,
        weight_name: str,
    ):
        """Configure WeightAnnealerEpoch.

        - `schedule`:
            - fn taking an epoch's index as input
            - Returns updated kl loss.
        - `kl_loss_var`: The loss-variable to be updated.
        - `param_kl_loss_weight`: initial weight, from the parameters file.
        - `weight_name`: friendly identified for printing.
        """
        super().__init__()
        self.schedule = schedule  # takes epoch, returns new weight
        self.weight_var = kl_loss_var
        self.param_kl_loss_weight = param_kl_loss_weight
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch: int, logs=None):
        """Make new numeric weight (float) for the VAE."""
        logs = logs or {}
        new_weight = self.schedule(epoch)  #
        new_value = new_weight * self.param_kl_loss_weight
        self.weight_var.assign(new_value)
        print("Current {} annealer weight is {}".format(self.weight_name, new_value))


# Schedules for VAEWeightAnnealer
def no_schedule(epoch_num) -> float:
    return 1.0


def sigmoid_schedule(time_step: float | int, start: float, slope: float = 1.0):
    """Make new kl weight.

    This function is passed into `partial`.
    - `time_step`: epoch/index.
    - `start`: param from config (fixed val)
    - `slope`: param from config (fixed val)
    convert back from numpy float to float

    The output should increase towards 1.0
    """
    return float(1.0 / (1.0 + np.exp(slope * (start - time_step))))


class EncoderDecoderCheckpoint(Callback):
    """Save Encoder, Decoder and optionally the Property predictor."""

    def __init__(
        self,
        encoder_model: Functional,
        decoder_model: Functional,
        parameters: Config,
        prop_pred_model: Functional | None = None,
        prop_to_monitor="val_x_pred_categorical_accuracy",
        save_best_only=True,
        monitor_op=np.greater,
        monitor_best_init=-np.Inf,
    ):
        """Save models.

        If `save_best_only=True`, uses the accuracy to decide
        whether to save it or not.

        - `encoder_model`: the encoder.
        - `decoder_model`: the decoder.
        - `parameters`: our program configuration.
        - `prop_pred_model`: the property predictor.

        - `prop_to_monitor`: a property that is a valid name in the model.
            These properties come from the training epoch-logs?
            Where do we specify which metrics the training tracks/monitors?
            To which of the models is it referred?
        - `save_best_only`: whether to compare to previous or save all checkpoints.
        - `monitor_op`: The operation to use when monitoring the property
            (e.g. accuracy to be maximized so use `np.greater`,
            loss to minimized, so use `np.less`)
        - `monitor_best_init`: starting point for monitor
        (use `-np.Inf` for maximization tests, and `np.Inf` for minimization tests)
        """

        super().__init__()

        self.p = parameters
        self.save_path = Path(self.p.checkpoint_path)

        self.encoder = encoder_model
        self.decoder = decoder_model
        self.prop_pred_model = prop_pred_model

        self.save_best_only = save_best_only

        self.monitor = prop_to_monitor
        self.monitor_op = monitor_op
        self.best = monitor_best_init

        self.verbose = self.p.verbosity

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if not isinstance(current, float):
            raise Exception(f"Current should be a float, found {type(current)}")

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                self.save_models(epoch)
        else:
            self.save_models(epoch)

        if self.verbose > 0:
            print(
                "Epoch {:05d}: {:s} previous value was {:0.5f} new is {:0.5f},".format(
                    epoch, self.monitor, self.best, current
                )
            )

    def save_models(self, epoch: int):
        self.encoder.save(self.save_path / f"encoder_{epoch}.keras")
        self.decoder.save(self.save_path / f"decoder_{epoch}.keras")

        if self.prop_pred_model:
            self.prop_pred_model.save(self.save_path / f"prop_pred_{epoch}.keras")
