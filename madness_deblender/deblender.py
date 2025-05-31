"""Perform Deblending."""

import logging
import os
import time

import galcheat
import numpy as np
import sep
import tensorflow as tf
import tensorflow_probability as tfp

from madness_deblender.extraction import extract_cutouts
from madness_deblender.FlowVAEnet import FlowVAEnet
from madness_deblender.utils import get_data_dir_path

tfd = tfp.distributions

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


def vectorized_compute_reconst_loss(args):
    """Compute reconstruction loss after being passed to tf.map_fn.

    Parameters
    ----------
    args: nested tensor
        (blended_field, reconstructions, index_pos_to_sub, num_components, sig_sq).
        passes all parameters to the compute_residual function.

    Returns
    -------
    reconstruction loss: tensor
        reconstruction loss term of the loss function.

    """
    (
        blended_field,
        reconstructions,
        index_pos_to_sub,
        num_components,
        sig_sq,
    ) = args
    # tf.print("reached 1")
    residual_field = compute_residual(
        blended_field,
        reconstructions,
        index_pos_to_sub=index_pos_to_sub,
        num_components=num_components,
    )
    reconst_loss = residual_field**2 / sig_sq
    # tf.print("reached 2")
    return tf.math.reduce_sum(reconst_loss) / 2


# @tf.function(jit_compile=True)
def compute_residual(
    blended_field,
    reconstructions,
    use_scatter_and_sub=True,
    index_pos_to_sub=None,
    num_components=1,
    padding_infos=None,
):
    """Compute residual in a field.

    Parameters
    ----------
    blended_field: tf tensor
        field with all the galaxies
    reconstructions: tf tensor
        reconstructions to be subtracted
    use_scatter_and_sub: bool
        uses tf.scatter_and_sub for subtraction instead of padding.
    index_pos_to_sub:
        index position for subtraction is `use_scatter_and_sub` is True
    num_components: int
        number of components/galaxies in the field
    padding_infos:
        padding parameters for reconstructions so that they can be subtracted from the field.
        Used when `use_scatter_and_sub` is False.

    Returns
    -------
    residual_field: tf tensor
        residual of the field after subtracting the reconstructions.

    """
    residual_field = tf.convert_to_tensor(blended_field, dtype=tf.float32)

    if use_scatter_and_sub:

        def one_step(i, residual_field):
            indices = index_pos_to_sub[i]
            reconstruction = reconstructions[i]

            residual_field = tf.tensor_scatter_nd_sub(
                residual_field,
                indices,
                tf.reshape(reconstruction, [tf.math.reduce_prod(reconstruction.shape)]),
            )

            return i + 1, residual_field

        c = lambda i, *_: i < num_components

        _, residual_field = tf.while_loop(
            c,
            one_step,
            (0, residual_field),
            maximum_iterations=num_components,
            parallel_iterations=1,
        )

        return residual_field

    else:

        if padding_infos is None:
            raise ValueError(
                "Pass padding infos or use the scatter_and_sub function instead"
            )

        def one_step(i, residual_field):
            # padding = tf.cast(padding_infos[i], dtype=tf.int32)
            padding = padding_infos[i]
            reconstruction = tf.pad(
                tf.gather(reconstructions, i), padding, "CONSTANT", name="padding"
            )
            # tf.where(mask, tf.zeros_like(tensor), tensor)
            residual_field = residual_field - reconstruction
            return i + 1, residual_field

        c = lambda i, _: i < num_components

        _, residual_field = tf.while_loop(
            c,
            one_step,
            (tf.constant(0, dtype=tf.int32), residual_field),
            maximum_iterations=num_components,
            parallel_iterations=1,
        )
    return residual_field


class Deblender:
    """Run the deblender."""

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
        weights_path=None,
        load_weights=True,
        survey=galcheat.get_survey("LSST"),
    ):
        """Initialize class variables.

        Parameters
        ----------
        stamp_shape: int
            size of input postage stamp
        latent_dim: int
            size of latent space.
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
        dense_layer_units: int
            number of units in the dense layer
        weights_path: string
            base path to load weights.
            flow weights are loaded from weights_path/flow6/val_loss
            vae weights are loaded from weights_path/deblender/val_loss
        survey: galcheat.survey object
            galcheat survey object to fetch survey details
        load_weights: bool
            Should be used as True to load pre-trained weights.
            if False, random weights are used(used for testing purposes).

        """
        self.latent_dim = latent_dim
        self.survey = survey
        self.flow_vae_net = FlowVAEnet(
            stamp_shape=stamp_shape,
            latent_dim=latent_dim,
            filters_encoder=filters_encoder,
            kernels_encoder=kernels_encoder,
            filters_decoder=filters_decoder,
            kernels_decoder=kernels_decoder,
            dense_layer_units=dense_layer_units,
            num_nf_layers=num_nf_layers,
            survey=survey,
        )

        if load_weights:
            if weights_path is None:
                data_dir_path = get_data_dir_path()
                weights_path = os.path.join(data_dir_path, survey.name)
            self.flow_vae_net.load_flow_weights(
                weights_path=os.path.join(weights_path, "flow/val_loss")
            )
            self.flow_vae_net.flow_model.trainable = False

            self.flow_vae_net.load_vae_weights(
                weights_path=os.path.join(weights_path, "vae/val_loss")
            )
            self.flow_vae_net.load_encoder_weights(
                weights_path=os.path.join(weights_path, "deblender/val_loss")
            )
        self.flow_vae_net.vae_model.trainable = False

        self.blended_fields = None
        self.detected_positions = None
        self.cutout_size = stamp_shape
        self.num_components = None
        self.channel_last = None
        self.noise_sigma = None
        self.num_bands = len(survey.available_filters)
        self.field_size = None
        self.use_log_prob = None
        self.linear_norm_coeff = None

        self.optimizer = None
        self.max_iter = None
        self.z = None

    def __call__(
        self,
        blended_fields,
        detected_positions,
        num_components,
        noise_sigma=None,
        max_iter=60,
        use_log_prob=True,
        channel_last=False,
        linear_norm_coeff=10000,
        convergence_criterion=None,
        use_debvader=True,
        optimizer=None,
        map_solution=True,
    ):
        """Run the Deblending operation.

        Parameters
        ----------
        blended_fields: np.ndarray
            batch of blended fields.
        detected_positions: list
            List of detected positions.
            as in array and not image
        num_components: list
            list of number of galaxies present in the image.
        noise_sigma: list of float
            background noise-level in each band
        max_iter: int
            number of iterations in the deblending step
        use_log_prob: bool
            decides whether or not to use the log_prob output of the flow deblender in the optimization.
        channel_last: bool
            if the channels/filters are the last column of the blended_fields
        linear_norm_coeff: int/list
            list stores the bandwise linear normalizing/scaling factor.
            if int is passed, the same scaling factor is used for all.
        convergence_criterion: tfp.optimizer.convergence_criteria
            For termination of the optimization loop.
        use_debvader: bool
            Use the encoder as a deblender to set the initial position for deblending.
        optimizer: tf.keras.optimizers
            Optimizer to use used for gradient descent.
        map_solution: bool
            To obtain the map solution (MADNESS) or debvader solution.
            Both `map_solution` and `use_debvader` cannot be False simultaneously.

        """
        # tf.config.run_functions_eagerly(False)
        self.linear_norm_coeff = linear_norm_coeff
        self.max_iter = max_iter
        self.num_components = tf.convert_to_tensor(num_components, dtype=tf.int32)
        self.use_log_prob = use_log_prob
        self.components = None
        self.channel_last = channel_last

        self.noise_sigma = noise_sigma

        if self.channel_last:
            self.blended_fields = tf.convert_to_tensor(
                blended_fields / linear_norm_coeff,
                dtype=tf.float32,
            )
        else:
            self.blended_fields = tf.convert_to_tensor(
                np.transpose(blended_fields, axes=[0, 2, 3, 1]) / linear_norm_coeff,
                dtype=tf.float32,
            )

        self.detected_positions = np.array(detected_positions)
        self.max_number = self.detected_positions.shape[1]
        self.num_fields = self.detected_positions.shape[0]

        self.field_size = np.shape(blended_fields)[2]

        self.results = self.gradient_decent(
            convergence_criterion=convergence_criterion,
            use_debvader=use_debvader,
            optimizer=optimizer,
            map_solution=map_solution,
        )

    def get_components(self):
        """Return the predicted components.

        The final returned image has the same value of channel_last as the input image.
        """
        if self.channel_last:
            return self.components
        return np.moveaxis(self.components, -1, -3)

    def compute_loss(
        self,
        z,
        sig_sq,
        index_pos_to_sub,
    ):
        """Compute loss at each epoch of Deblending optimization.

        Parameters
        ----------
        z: tf tensor
            latent space representations of the reconstructions.
        sig_sq: tf tensor
            Factor for division to convert the MSE to Gaussian approx to Poisson noise.
        index_pos_to_sub:
            index position for subtraction is `use_scatter_and_sub` is True

        Returns
        -------
        final_loss: tf float
            Final loss to be minimized
        reconstruction_loss: tf float
            Loss from residuals in the field
        log_prob: tf float
            log prob evaluated by normalizing flow
        residual_field: tf tensor
            residual field after deblending.

        """
        reconstructions = self.flow_vae_net.decoder(z)

        reconstructions = tf.reshape(
            reconstructions,
            [
                self.num_fields,
                self.max_number,
                self.cutout_size,
                self.cutout_size,
                self.num_bands,
            ],
        )

        reconstruction_loss = tf.map_fn(
            vectorized_compute_reconst_loss,
            elems=(
                self.blended_fields,
                reconstructions,
                index_pos_to_sub,
                self.num_components,
                sig_sq,
            ),
            parallel_iterations=20,
            fn_output_signature=tf.TensorSpec(
                [],
                dtype=tf.float32,
            ),
        )

        log_prob = self.flow_vae_net.flow(z)

        log_prob = tf.reduce_sum(
            tf.reshape(log_prob, [self.num_fields, self.max_number]), axis=[1]
        )

        final_loss = reconstruction_loss

        if self.use_log_prob:
            final_loss = reconstruction_loss - log_prob

        return final_loss, reconstruction_loss, log_prob

    def get_index_pos_to_sub(self):
        """Get index position to run tf.tensor_scatter_nd_sub."""
        indices = (
            np.indices((self.cutout_size, self.cutout_size, self.num_bands))
            .reshape(3, -1)
            .T
        )
        indices = np.repeat(np.expand_dims(indices, axis=0), self.max_number, axis=0)
        indices = np.repeat(np.expand_dims(indices, axis=0), self.num_fields, axis=0)

        starting_positions = np.round(self.detected_positions).astype(int) - int(
            (self.cutout_size - 1) / 2
        )

        indices = np.moveaxis(indices, 2, 0)

        indices[:, :, :, 0:2] += starting_positions
        indices = np.moveaxis(indices, 0, 2)

        return indices

    def get_padding_infos(self):
        """Compute padding info to convert galaxy cutout into field."""
        padding_infos_list = []
        for field_num in range(self.num_fields):
            inner_list = []
            for detected_position in self.detected_positions[field_num]:
                starting_pos_x = round(detected_position[0]) - int(
                    (self.cutout_size - 1) / 2
                )
                starting_pos_y = round(detected_position[1]) - int(
                    (self.cutout_size - 1) / 2
                )

                padding = [
                    [
                        starting_pos_x,
                        self.field_size - (starting_pos_x + int(self.cutout_size)),
                    ],
                    [
                        starting_pos_y,
                        self.field_size - (starting_pos_y + int(self.cutout_size)),
                    ],
                    [0, 0],
                ]

                inner_list.append(padding)
            padding_infos_list.append(inner_list)
        return np.array(padding_infos_list)

    def compute_noise_sigma(self):
        """Compute noise level with sep."""
        sig = []
        for i in range(len(self.survey.available_filters)):
            sig.append(
                sep.Background(
                    np.ascontiguousarray(self.blended_fields.numpy()[0][:, :, i])
                ).globalrms
            )

        return np.array(sig)

    def gradient_decent(
        self,
        initZ=None,
        convergence_criterion=None,
        use_debvader=True,
        optimizer=None,
        map_solution=True,
    ):
        """Perform the gradient descent step to separate components (galaxies).

        Parameters
        ----------
        initZ: np.ndarray
            initial value of the latent space.
        convergence_criterion: tfp.optimizer.convergence_criteria
            For termination of the optimization loop
        use_debvader: bool
            Use the encoder as a deblender to set the initial position for deblending.
        optimizer: tf.keras.optimizers
            Optimizer to use used for gradient descent.
        map_solution: bool
            To obtain the map solution or debvader solution.
            Both `map_solution` and `use_debvader` cannot be False at the same time.

        Returns
        -------
        results: list
            variation of the loss over deblending iterations.

        """
        LOG.info("use debvader: " + str(use_debvader))
        if not map_solution and not use_debvader:
            raise ValueError(
                "Both use_debvader and map_solution cannot be False at the same time"
            )

        if not use_debvader:
            # check constraint parameter over here
            z = tf.Variable(
                self.flow_vae_net.td.sample(self.num_fields * self.max_number)
            )

        else:
            # use the encoder to find a good starting point.
            LOG.info("\nUsing encoder for initial point")
            t0 = time.time()
            cutouts = np.zeros(
                (
                    self.num_fields * self.max_number,
                    self.cutout_size,
                    self.cutout_size,
                    self.num_bands,
                )
            )
            for field_num in range(self.num_fields):
                cutouts[
                    field_num * self.max_number : field_num * self.max_number
                    + self.num_components[field_num]
                ] = extract_cutouts(
                    self.blended_fields.numpy()[field_num],
                    pos=self.detected_positions[field_num][
                        : self.num_components[field_num]
                    ],
                    cutout_size=self.cutout_size,
                    channel_last=True,
                )[
                    0
                ]
            initZ = tfp.layers.MultivariateNormalTriL(self.latent_dim)(
                self.flow_vae_net.encoder(cutouts)
            )
            LOG.info("Time taken for initialization: " + str(time.time() - t0))
            z = tf.Variable(
                tf.reshape(
                    initZ.mean(), (self.num_fields * self.max_number, self.latent_dim)
                )
            )

        if map_solution:
            if optimizer is None:
                lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.075,
                    decay_steps=30,
                    decay_rate=0.8,
                    staircase=True,
                )
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

            LOG.info("\n--- Starting gradient descent in the latent space ---")
            LOG.info(f"Maximum number of iterations: {self.max_iter}")

            LOG.info(f"Number of fields: {self.num_fields}")
            LOG.info(f"Number of Galaxies: {self.num_components}")
            LOG.info(f"Dimensions of latent space: {self.latent_dim}")

            t0 = time.time()

            index_pos_to_sub = self.get_index_pos_to_sub()

            index_pos_to_sub = tf.convert_to_tensor(
                self.get_index_pos_to_sub(),
                dtype=tf.int32,
            )

            if self.noise_sigma is None:
                noise_level = self.compute_noise_sigma()

            noise_level = tf.convert_to_tensor(
                noise_level,
                dtype=tf.float32,
            )
            # Calculate sigma^2 with Gaussian approximation to Poisson noise.
            # Note here that self.postage stamp is normalized but it must be divided again
            # to ensure that the log likelihood does not change due to scaling/normalizing

            sig_sq = self.blended_fields / self.linear_norm_coeff + noise_level**2

            results = tfp.math.minimize(
                loss_fn=self.generate_grad_step_loss(
                    z=z,
                    sig_sq=sig_sq,
                    index_pos_to_sub=index_pos_to_sub,
                ),
                trainable_variables=[z],
                num_steps=self.max_iter,
                optimizer=optimizer,
                convergence_criterion=convergence_criterion,
            )

            """ LOG.info(f"Final loss {output.objective_value.numpy()}")
            LOG.info("converged "+ str(output.converged.numpy()))
            LOG.info("converged "+ str(output.num_iterations.numpy()))

            z_flatten = output.position
            z = tf.reshape(z_flatten, shape=[self.num_components, self.latent_dim]) """

            LOG.info("--- Gradient descent complete ---")
            LOG.info("Time taken for gradient descent: " + str(time.time() - t0))
        else:
            results = None
        self.components = tf.reshape(
            self.flow_vae_net.decoder(z) * self.linear_norm_coeff,
            [
                self.num_fields,
                self.max_number,
                self.cutout_size,
                self.cutout_size,
                self.num_bands,
            ],
        )
        self.z = tf.reshape(z, (self.num_fields, self.max_number, self.latent_dim))

        return results

    def generate_grad_step_loss(
        self,
        z,
        sig_sq,
        index_pos_to_sub,
    ):
        """Return function compute training loss that has no arguments.

        Parameters
        ----------
        z: tf tensor
            latent space representations of the reconstructions.
        sig_sq: tf tensor
            Factor for the division to convert the MSE to Gaussian approx to Poisson noise.
        index_pos_to_sub:
            index position for subtraction is `use_scatter_and_sub` is True

        Returns
        -------
        training_loss: python function
            computes loss without taking any arguments.

        """

        @tf.function
        def training_loss():
            """Compute training loss."""
            loss, *_ = self.compute_loss(
                z=z,
                sig_sq=sig_sq,
                index_pos_to_sub=index_pos_to_sub,
            )

            return loss

        return training_loss
