"""Create the models for MADNESS."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    Dense,
    Flatten,
    Input,
    PReLU,
    Reshape,
)
from tensorflow.keras.models import Model

tfd = tfp.distributions
tfb = tfp.bijectors


def create_encoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    dense_layer_units,
):
    """Create the encoder.

    Parameters
    ----------
    input_shape: list
        shape of input tensor
    latent_dim: int
        size of the latent space
    filters: list
        filters used for the convolutional layers
    kernels: list
        kernels used for the convolutional layers
    dense_layer_units: int
            number of units in the dense layer

    Returns
    -------
    encoder: tf.keras.Model
       model that takes as input the image of a galaxy and projects it to the latent space.

    """
    # Input layer
    input_layer = Input(shape=(input_shape))
    h = input_layer
    # Define the model
    for i in range(len(filters)):
        h = Conv2D(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = Dense(dense_layer_units)(h)
    h = PReLU()(h)
    h = Dense(
        tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
        activation=None,
    )(h)

    return Model(input_layer, h, name="encoder")


def create_decoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    dense_layer_units,
):
    """Create the decoder.

    Parameters
    ----------
    input_shape: list
        shape of input tensor
    latent_dim: int
        size of the latent space
    filters: list
        filters used for the convolutional layers
    kernels: list
        kernels used for the convolutional layers
    sigma_cutoff: list of float
        backgound noise-level in each band
    dense_layer_units: int
            number of units in the dense layer

    Returns
    -------
    decoder: tf.keras.Model
        model that takes as input a point in the latent space and decodes it to reconstruct a noiseless galaxy.

    """
    input_layer = Input(shape=(latent_dim,))
    h = Dense(dense_layer_units, activation=None)(input_layer)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0] / 2 ** (len(filters))))
    h = Dense(w * w * filters[-1], activation=None)(tf.cast(h, tf.float32))
    h = PReLU()(h)
    h = Reshape((w, w, filters[-1]))(h)
    for i in range(len(filters) - 1, -1, -1):
        h = Conv2DTranspose(
            filters=filters[i],
            kernel_size=(kernels[i], kernels[i]),
            activation=None,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)
    
    h = Conv2DTranspose(input_shape[-1], (3, 3), activation="relu", padding="same")(h)

    # In case the last convolutional layer does not provide an image of the size of the input image, cropp it.
    cropping = int(h.get_shape()[1] - input_shape[0])
    if cropping > 0:
        if cropping % 2 == 0:
            h = Cropping2D(cropping / 2)(h)
        else:
            h = Cropping2D(
                ((cropping // 2, cropping // 2 + 1), (cropping // 2, cropping // 2 + 1))
            )(h)

    return Model(input_layer, h, name="decoder")


def create_flow(latent_dim=10, num_nf_layers=6):
    """Create the Flow model that takes as input a point in latent space and returns the log_prob.

    Parameters
    ----------
    latent_dim: int
        size of the latent space
    num_nf_layers: int
        number of layers in the normalizing flow

    Returns
    -------
    model: tf.keras.Model
        model that takes as input a point in the latent sapce and returns the log_prob wrt the base distribution
    bijector_chain: tfp.bijectors.Chain
        bijector chain that is being applied on the base distribution

    """
    bijects = []
    zdist = tfd.Independent(
        tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
        reinterpreted_batch_ndims=1,
    )

    # loop over desired bijectors and put into list

    #  add cyclic rotation in steps of 3
    permute_arr = np.arange(0, latent_dim)[(np.arange(0, latent_dim) - 3)[:]]

    for i in range(num_nf_layers):
        # create a MAF
        anet = tfb.AutoregressiveNetwork(
            params=2,
            hidden_units=[32, 32],
            activation="tanh",
        )
        ab = tfb.MaskedAutoregressiveFlow(anet)

        # Add bijectors to a list
        bijects.append(ab)

        # Add permutation layers
        permute = tfb.Permute(permute_arr)
        bijects.append(permute)

    bijects.append(tfb.BatchNormalization())
    # combine the bijectors into a chain
    bijector_chain = tfb.Chain(list(reversed(bijects[:-1])))

    # make transformed dist
    td = tfd.TransformedDistribution(zdist, bijector=bijector_chain)

    # create and return model
    input_layer = Input(shape=(latent_dim,))
    model = Model(input_layer, td.log_prob(input_layer), name="flow")
    return model, td


# Function to define model
def create_model_fvae(
    input_shape,
    latent_dim,
    filters_encoder,
    kernels_encoder,
    filters_decoder,
    kernels_decoder,
    dense_layer_units,
    num_nf_layers=6,
    kl_prior=None,
    kl_weight=None,
):
    """Create the sinmultaneously create the VAE and the flow model.

    Parameters
    ----------
    input_shape: list
        shape of input tensor
    latent_dim: int
        size of the latent space
    filters_encoder: list
        filters used for the convolutional layers in encoder
    filters_decoder: list
        filters used for the convolutional layers in decoder
    kernels_encoder: list
        kernels used for the convolutional layers in encoder
    kernels_decoder: list
        kernels used for the convolutional layers in decoder
    num_nf_layers: int
        number of layers in the flow network
    kl_prior: tf distribution
        KL prior to be applied on the latent space.
    kl_weight: float
        Weight to be multiplied tot he kl_prior
    dense_layer_units: int
            number of units in the dense layer

    Returns
    -------
    vae_model: tf.keras.Model
        vae model which consists of the encoder and decoder.
    flow_model: tf.keras.Model
        flow model which consists of the encoder and flow transormation layers
    encoder: tf.keras.Model
        encoder which is common to both the vae_model and the flow_model
        model that takes as input the image of a galaxy and projects it to the latent space
    decoder: tf.keras.Model
        decoder which is present in the vae_model
        model that takes as input a point in the latent space and decodes it to reconstruct a noiseless galaxy.
    flow: tf.keras.Model
        flow network which is present in the flow_model
        model that takes as input a point in the latent sapce and returns the log_prob wrt the base distribution

    """
    # create the encoder
    encoder = create_encoder(
        input_shape,
        latent_dim,
        filters_encoder,
        kernels_encoder,
        dense_layer_units,
    )

    # create the decoder
    decoder = create_decoder(
        input_shape,
        latent_dim,
        filters_decoder,
        kernels_decoder,
        dense_layer_units,
    )

    # create the flow transformation
    flow, td = create_flow(latent_dim=latent_dim, num_nf_layers=num_nf_layers)

    # Define the prior for the latent space
    activity_regularizer = None
    if kl_prior is not None:
        activity_regularizer = tfp.layers.KLDivergenceRegularizer(
            kl_prior, weight=0.01 if kl_weight is None else kl_weight
        )

    # Build the model
    x_input = Input(shape=(input_shape))
    z = tfp.layers.MultivariateNormalTriL(
        latent_dim, activity_regularizer=activity_regularizer, name="latent_space"
    )(encoder(x_input))

    vae_model = Model(inputs=x_input, outputs=decoder(z))
    flow_model = Model(
        inputs=x_input, outputs=flow(z)
    )  

    return vae_model, flow_model, encoder, decoder, flow, td
