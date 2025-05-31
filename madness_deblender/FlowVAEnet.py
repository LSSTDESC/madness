"""Defines models and training functions for the Deblender."""

import logging

import galcheat
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from madness_deblender.losses import flow_loss_fn
from madness_deblender.model import create_encoder, create_model_fvae

tfd = tfp.distributions
tfb = tfp.bijectors
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


class FlowVAEnet:
    """Initialize and train the Neural Network model."""

    def __init__(
        self,
        stamp_shape=45,
        latent_dim=16,
        filters_encoder=[32, 128, 256, 512],
        filters_decoder=[64, 96, 128],
        kernels_encoder=[5, 5, 5, 5],
        kernels_decoder=[5, 5, 5],
        dense_layer_units=512,
        num_nf_layers=6,
        kl_prior=None,
        kl_weight=None,
        survey=galcheat.get_survey("LSST"),
    ):
        """Create the required models according to the specifications.

        Parameters
        ----------
        stamp_shape: int
            size of input postage stamp
        latent_dim: int
            size of the latent space
        filters_encoder: list
            filters used for the convolutional layers in the encoder
        filters_decoder: list
            filters used for the convolutional layers in the decoder
        kernels_encoder: list
            kernels used for the convolutional layers in the encoder
        kernels_decoder: list
            kernels used for the convolutional layers in the decoder
        num_nf_layers: int
            number of layers in the flow network
        kl_prior: tf distribution
            KL prior to be applied to the latent space.
        kl_weight: float
            Weight to be multiplied tot he kl_prior
        survey: galcheat.survey object
            galcheat survey object to fetch survey details
        dense_layer_units: int
            number of units in the dense layer

        """
        self.input_shape = [stamp_shape, stamp_shape, len(survey.available_filters)]
        self.latent_dim = latent_dim

        self.filters_encoder = filters_encoder
        self.kernels_encoder = kernels_encoder

        self.filters_decoder = filters_decoder
        self.kernels_decoder = kernels_decoder

        self.nb_of_bands = len(survey.available_filters)
        self.num_nf_layers = num_nf_layers

        (
            self.vae_model,
            self.flow_model,
            self.encoder,
            self.decoder,
            self.flow,
            self.td,
        ) = create_model_fvae(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            filters_encoder=self.filters_encoder,
            kernels_encoder=self.kernels_encoder,
            filters_decoder=self.filters_decoder,
            kernels_decoder=self.kernels_decoder,
            dense_layer_units=dense_layer_units,
            num_nf_layers=self.num_nf_layers,
            kl_prior=kl_prior,
            kl_weight=kl_weight,
        )

        self.optimizer = None
        self.callbacks = None

    def train_vae(
        self,
        train_generator,
        validation_generator,
        callbacks,
        loss_function,
        train_encoder=True,
        train_decoder=True,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        track_kl=False,
        epochs=35,
        verbose=1,
    ):
        """Train only the the components of VAE model (the encoder and/or decoder).

        Parameters
        ----------
        train_generator:
            generator to be used for training the network.
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        validation_generator:
            generator to be used for validation
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        callbacks: list
            List of keras.callbacks.Callback instances.
            List of callbacks to apply during training.
            See tf.keras.callbacks
        loss_function: python function
            function that can compute the loss.
        train_encoder: bool
            Flag to decide if the encoder is to be trained.
        train_decoder: bool
            Flag to decide if the decoder is to be trained.
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        track_kl:bool
            To decide if the KL loss is to be tracked through the iterations.
        epochs: int
            number of epochs for which the model is going to be trained
        verbose: int
            verbose option for training.
            'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
            Note that the progress bar is not particularly useful when logged to a file,
            so verbose=2 is recommended when not running interactively (eg, in a production environment).

        """
        if (not train_decoder) and (not train_encoder):
            raise ValueError(
                "Training failed because both encoder and decoder are not trainable"
            )

        self.encoder.trainable = train_encoder
        self.decoder.trainable = train_decoder

        self.vae_model.summary()
        LOG.info("\n--- Training only VAE network ---")
        LOG.info("Encoder status: " + str(train_encoder))
        LOG.info("Decoder status: " + str(train_decoder))

        metrics = ["mse"]

        # Custom metric to display the KL divergence during training
        def kl_metric(y_true, y_pred):
            return K.sum(self.vae_model.losses)

        if track_kl:
            metrics += [kl_metric]

        LOG.info("Number of epochs: " + str(epochs))

        if loss_function is None:
            print("pass valid loss function")
        self.vae_model.compile(
            optimizer=optimizer,
            loss={"decoder": loss_function},
            experimental_run_tf_function=False,
            metrics=metrics,
        )

        hist = self.vae_model.fit(
            x=(
                train_generator[0]
                if isinstance(train_generator, tuple)
                else train_generator
            ),
            y=train_generator[1] if isinstance(train_generator, tuple) else None,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            validation_data=validation_generator,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True,
        )
        return hist

    def train_encoder(
        self,
        train_generator,
        validation_generator,
        callbacks,
        loss_function,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        epochs=35,
        verbose=1,
    ):
        """Train only the the components of VAE model (the encoder and/or decoder).

        Parameters
        ----------
        train_generator:
            generator to be used for training the network.
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        validation_generator:
            generator to be used for validation
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        callbacks: list
            List of keras.callbacks.Callback instances.
            List of callbacks to apply during training.
            See tf.keras.callbacks
        loss_function: python function
            function that can compute the loss.
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        epochs: int
            number of epochs for which the model is going to be trained
        verbose: int
            verbose option for training.
            'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
            Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).

        """
        self.encoder.summary()
        LOG.info("\n--- Training only encoder network ---")
        LOG.info("Number of epochs: " + str(epochs))

        if loss_function is None:
            print("pass valid loss function")
        self.encoder.compile(
            optimizer=optimizer,
            loss=loss_function,
            experimental_run_tf_function=False,
        )
        hist = self.encoder.fit(
            x=(
                train_generator[0]
                if isinstance(train_generator, tuple)
                else train_generator
            ),
            y=train_generator[1] if isinstance(train_generator, tuple) else None,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            validation_data=validation_generator,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True,
        )
        return hist

    def train_flow(
        self,
        train_generator,
        validation_generator,
        callbacks,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        epochs=35,
        verbose=1,
    ):
        """Train only the flow part of the flow_model while keeping the encoder constant.

        Parameters
        ----------
        train_generator:
            generator to be used for training the network.
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        validation_generator:
            generator to be used for validation
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        callbacks: list
            List of keras.callbacks.Callback instances.
            List of callbacks to apply during training.
            See tf.keras.callbacks
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        epochs: int
            number of epochs for which the model is going to be trained
        num_scheduler_epochs: int
            number of epochs after which learning rate is reduced by a factor of `e`
        verbose: int
            verbose option for training.
            'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
            Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).

        """
        self.flow.trainable = True
        self.encoder.trainable = False
        self.flow_model.compile(
            optimizer=optimizer,
            loss={"flow": flow_loss_fn},
            experimental_run_tf_function=False,
        )
        self.flow_model.summary()

        LOG.info("\n--- Training only FLOW network ---")
        LOG.info("Number of epochs: " + str(epochs))

        hist = self.flow_model.fit(
            x=(
                train_generator[0]
                if isinstance(train_generator, tuple)
                else train_generator
            ),
            y=train_generator[1] if isinstance(train_generator, tuple) else None,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            validation_data=validation_generator,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True,
        )

        return hist

    def load_vae_weights(self, weights_path, is_folder=True):
        """Load the trained weights of the VAE model (encoder and decoder).

        Parameters
        ----------
        weights_path: str
            path to the weights of the vae_model.
            The path must point to either a folder with the saved checkpoints or directly to the checkpoint.
        is_folder: bool
            specifies if the weights_path points to a folder.
            If True, then the latest checkpoint is loaded.
            else, the checkpoint specified in the path is loaded

        """
        if is_folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.vae_model.load_weights(weights_path).expect_partial()

    def load_flow_weights(self, weights_path, is_folder=True):
        """Load the trained weights encoder and NF (encoder and Flow).

        Parameters
        ----------
        weights_path: str
            path to the weights of the vae_model.
            The path must point to either a folder with the saved checkpoints or directly to the checkpoint.
        is_folder: bool
            specifies if the weights_path points to a folder.
            If True, then the latest checkpoint is loaded.
            else, the checkpoint specified in the path is loaded

        """
        if is_folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.flow_model.load_weights(weights_path).expect_partial()

    def load_encoder_weights(self, weights_path, is_folder=True):
        """Load the trained weights encoder and NF (encoder and Flow).

        Parameters
        ----------
        weights_path: str
            path to the weights of the vae_model.
            The path must point to either a folder with the saved checkpoints or directly to the checkpoint.
        is_folder: bool
            specifies if the weights_path points to a folder.
            If True, then the latest checkpoint is loaded.
            else, the checkpoint specified in the path is loaded

        """
        if is_folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.encoder.load_weights(weights_path).expect_partial()

    def randomize_encoder(self):
        """Randomize the weights of the encoder to restart training."""
        new_encoder = create_encoder(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            filters=self.filters_encoder,
            kernels=self.kernels_encoder,
        )
        self.encoder.set_weights(new_encoder.get_weights())
