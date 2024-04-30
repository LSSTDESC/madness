"""Losses to train the ML models."""

import logging

import tensorflow as tf
import tensorflow_probability as tfp

from madness_deblender.callbacks import changeAlpha

tfd = tfp.distributions
tfb = tfp.bijectors
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


@tf.function(autograph=False)
def vae_loss_fn_mse(x, predicted_distribution):
    """Compute the MSE loss function.

    Parameters
    ----------
    x: array/tensor
        Galaxy ground truth.
    predicted_distribution: tf distribution
        pixel wise distribution of the flux.
        The mean is used to compute the MSE.

    Returns
    -------
    objective: float
        objective to be minimized by the minimizer.

    """
    mean = predicted_distribution.sample()

    weight = tf.sqrt(tf.reduce_max(x, axis=[1, 2, 3]))
    diff = tf.subtract(mean, x)
    pixel_mse = tf.square(diff)
    mse = tf.math.reduce_sum(pixel_mse, axis=[1, 2, 3])

    objective = tf.math.reduce_mean(tf.divide(mse, weight))

    return objective


def deblender_loss_fn_wrapper(
    sigma_cutoff, use_ssim=False, ch_alpha=None, linear_norm_coeff=10000
):
    """Input field sigma into ssim loss function.

    Parameters
    ----------
    sigma_cutoff: list
        list of sigma levels (normalized) in the bands.
    use_ssim: bool
        Flag to add an extra ssim term to the loss function.
        This loss is supposed to force the network to learn information in noisy bands relatively earlier.
    ch_alpha: madness.callbacks.ChangeAlpha
        an instance of ChangeAlpha to update the weight of SSIM over epochs.
    linear_norm_coeff: int
        linear norm coefficient used for normalizing.

    Returns
    -------
    deblender_ssim_loss_fn:
        function to compute the loss using SSIM weight.

    """
    if use_ssim and not isinstance(ch_alpha, changeAlpha):
        raise ValueError(
            "Inappropriate value for changeAlpha. Must been an instance of madness_deblender.callbacks.changeAlpha"
        )

    @tf.function
    def deblender_ssim_loss_fn(y, predicted_galaxy):
        """Compute the loss under predicted distribution, weighted by the SSIM.

        Parameters
        ----------
        y: array/tensor
            Galaxy ground truth.
        predicted_galaxy: tf tensor
            pixel-wise prediction of the flux.

        Returns
        -------
        objective: float
            objective to be minimized by the minimizer.

        """
        loss = tf.reduce_sum(
            (y - predicted_galaxy) ** 2 / (sigma_cutoff**2 + y / linear_norm_coeff),
            axis=[1, 2, 3],
        )

        if use_ssim:
            band_normalizer = tf.reduce_max(y, axis=[1, 2], keepdims=True)
            ssim = tf.image.ssim(
                y / band_normalizer,
                predicted_galaxy / band_normalizer,
                max_val=1,
            )
            tf.stop_gradient(ch_alpha.alpha)
            loss = loss * (1 - ch_alpha.alpha * ssim)

        loss = tf.reduce_mean(loss)

        return loss

    return deblender_ssim_loss_fn


def deblender_encoder_loss_wrapper(
    original_encoder,
    noise_sigma,
    latent_dim=16,
):
    """Loss function wrapper.

    Parameters
    ----------
    original_encoder:
        Original trained encoder (VAE as a generative model).
    noise_sigma: list/array
        Noise level in the bands.
    latent_dim: int
        Number of latent dimensions. Defaults to 16.

    Return
    ------
    deblender_encoder_loss:
        function to compute l2 norm

    """

    @tf.function
    def deblender_encoder_loss(y, predicted_galaxy):

        z_predicted = tfp.layers.MultivariateNormalTriL(
            latent_dim,
        )(predicted_galaxy)

        y = y + tf.random.normal(y.shape[1:], [0] * noise_sigma.shape[0], noise_sigma)
        z_original = tfp.layers.MultivariateNormalTriL(
            latent_dim,
        )(original_encoder(y)).sample()

        l2_norm = tf.sqrt(tf.reduce_sum((z_predicted - z_original) ** 2))
        loss = tf.reduce_mean(l2_norm)

        return loss

    return deblender_encoder_loss


@tf.function(experimental_compile=True)
def flow_loss_fn(x, output):
    """Compute the loss under predicted distribution.

    Parameters
    ----------
    x: array/tensor
        Galaxy ground truth.
    output: tensor
        log probability output over a batch from Normalizing Flow

    Return
    ------
    objective: float
        objective to be minimized by the minimizer.

    """
    return -tf.math.reduce_mean(output)
