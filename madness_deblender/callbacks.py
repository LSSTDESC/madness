"""Define callbacks for training."""

import os

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


def define_callbacks(weights_save_path, lr_scheduler_epochs=None, patience=40):
    """Define callbacks for a network to train.

    Parameters
    ----------
    weights_save_path: String
        path at which weights are to be saved.path at which weights are to be saved.
        By default, it saves weights in the data folder.
    lr_scheduler_epochs: int
        number of iterations after which the learning rate is decreased by a factor of $e$.
        The default is None, and a constant learning rate is used
    patience: int
        number of iterations after which training is stopped if the loss does not decrease.

    """
    checkpointer_val_mse = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_save_path, "val_mse", "weights.ckpt"),
        monitor="val_mse",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    checkpointer_val_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_save_path, "val_loss", "weights.ckpt"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience
    )

    callbacks = [checkpointer_val_mse, checkpointer_val_loss, early_stopping]

    if lr_scheduler_epochs is not None:

        def scheduler(epoch, lr):
            if (epoch + 1) % lr_scheduler_epochs != 0:
                return lr
            else:
                return lr / 2.5

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks += [lr_scheduler]

    callbacks += [tf.keras.callbacks.TerminateOnNaN()]

    return callbacks


class changeAlpha(Callback):
    """Update SSIM weight over epochs."""

    def __init__(self, max_epochs):
        """Initialize the weight parameter.

        Parameters
        ----------
        max_epochs: int
            number of epochs after which SSIM weight should go to 0.

        """
        super().__init__()
        self.alpha = tf.Variable(1.0)
        self.max_epochs = max_epochs

    def on_epoch_begin(self, epoch, logs={}):
        """Update the SSIM weight.

        Parameters
        ----------
        epoch: int
            current epoch
        logs: dict
            logs

        """
        K.set_value(self.alpha, max(0, 1 - epoch / self.max_epochs))
        print("Setting alpha to =", str(self.alpha))
