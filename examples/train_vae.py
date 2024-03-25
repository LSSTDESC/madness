"""Train all models."""

import logging
import os
import sys

import galcheat
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from galcheat.utilities import mean_sky_level
from maddeb.callbacks import changeAlpha, define_callbacks
from maddeb.dataset_generator import batched_CATSIMDataset
from maddeb.FlowVAEnet import FlowVAEnet
from maddeb.losses import deblender_encoder_loss_wrapper, deblender_loss_fn_wrapper
from maddeb.utils import get_data_dir_path, get_maddeb_config_path

tfd = tfp.distributions

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

# define the parameters
batch_size = 100
vae_epochs = 200
flow_epochs = 200
deblender_epochs = 150
lr_scheduler_epochs = 30
latent_dim = 16
linear_norm_coeff = 10000
patience = 30

with open(get_maddeb_config_path()) as f:
    maddeb_config = yaml.safe_load(f)

survey_name = maddeb_config["survey_name"]

if survey_name not in ["LSST", "HSC"]:
    raise ValueError(
        "survey should be one of: LSST or HSC"
    )  # other surveys to be added soon!

train_models = sys.argv[
    1
]  # either "all" or a list containing: ["GenerativeModel","NormalizingFlow","Deblender"]

kl_weight_exp = int(sys.argv[2])
kl_weight = 10**-kl_weight_exp
LOG.info(f"KL weight{kl_weight}")

survey = galcheat.get_survey(survey_name)

noise_sigma = []
for b, name in enumerate(survey.available_filters):
    filt = survey.get_filter(name)
    noise_sigma.append(np.sqrt(mean_sky_level(survey, filt).to_value("electron")))

noise_sigma = np.array(noise_sigma, dtype=np.float32) / linear_norm_coeff

kl_prior = tfd.Independent(
    tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
)

f_net = FlowVAEnet(
    latent_dim=latent_dim,
    kl_prior=kl_prior,
    kl_weight=kl_weight,
    survey=survey,
)

# Keras Callbacks
data_path = get_data_dir_path()

# path_weights = os.path.join(data_path, f"catsim_kl{kl_weight_exp}{latent_dim}d")
path_weights = os.path.join(data_path, survey.name + str(kl_weight))

# Define the generators
ds_isolated_train, ds_isolated_val = batched_CATSIMDataset(
    train_data_dir=None,
    val_data_dir=None,
    tf_dataset_dir=os.path.join(
        maddeb_config["TF_DATASET_PATH"][survey_name], "isolated_tfDataset"
    ),
    linear_norm_coeff=linear_norm_coeff,
    batch_size=batch_size,
    x_col_name="blended_gal_stamps",
    y_col_name="isolated_gal_stamps",
)

if train_models.lower() == "all" or "vae" in train_models:

    ssim_fraction = 0.25
    # Define all used callbacks
    callbacks = define_callbacks(
        os.path.join(path_weights, "ssim"),
        lr_scheduler_epochs=lr_scheduler_epochs,
        patience=vae_epochs,
    )

    ch_alpha = changeAlpha(max_epochs=int(ssim_fraction * vae_epochs))

    hist_vae = f_net.train_vae(
        ds_isolated_train,
        ds_isolated_val,
        callbacks=callbacks + [ch_alpha],
        epochs=int(ssim_fraction * vae_epochs),
        train_encoder=True,
        train_decoder=True,
        track_kl=True,
        optimizer=tf.keras.optimizers.Adam(1e-5, clipvalue=0.1),
        loss_function=deblender_loss_fn_wrapper(
            sigma_cutoff=noise_sigma,
            use_ssim=True,
            ch_alpha=ch_alpha,
            linear_norm_coeff=linear_norm_coeff,
        ),
        verbose=2,
        # loss_function=vae_loss_fn_wrapper(sigma=noise_sigma, linear_norm_coeff=linear_norm_coeff),
    )

    np.save(path_weights + "/train_vae_ssim_history.npy", hist_vae.history)

    # f_net.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))

    callbacks = define_callbacks(
        os.path.join(path_weights, "vae"),
        lr_scheduler_epochs=lr_scheduler_epochs,
        patience=patience,
    )

    hist_vae = f_net.train_vae(
        ds_isolated_train,
        ds_isolated_val,
        callbacks=callbacks,
        epochs=int((1 - ssim_fraction) * vae_epochs),
        train_encoder=True,
        train_decoder=True,
        track_kl=True,
        optimizer=tf.keras.optimizers.Adam(1e-5, clipvalue=0.1),
        loss_function=deblender_loss_fn_wrapper(
            sigma_cutoff=noise_sigma,
            use_ssim=False,
            linear_norm_coeff=linear_norm_coeff,
        ),
        verbose=2,
        # loss_function=vae_loss_fn_wrapper(sigma=noise_sigma, linear_norm_coeff=linear_norm_coeff),
    )

    np.save(path_weights + "/train_vae_history.npy", hist_vae.history)

if train_models.lower() == "all" or "nf" in train_models:

    num_nf_layers = 6
    f_net = FlowVAEnet(
        latent_dim=latent_dim,
        kl_prior=None,
        kl_weight=0,
        num_nf_layers=num_nf_layers,
        survey=survey,
    )

    f_net.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))
    # f_net.load_flow_weights(os.path.join(path_weights, f"flow{num_nf_layers}", "val_loss"))

    # Define all used callbacks
    callbacks = define_callbacks(
        os.path.join(path_weights, "flow"),
        lr_scheduler_epochs=lr_scheduler_epochs,
        patience=patience,
    )

    # Now train the model
    hist_flow = f_net.train_flow(
        ds_isolated_train,
        ds_isolated_val,
        callbacks=callbacks,
        optimizer=tf.keras.optimizers.Adam(1e-4, clipvalue=0.01),
        epochs=flow_epochs,
        verbose=2,
    )

    np.save(os.path.join(path_weights, "train_flow_history.npy"), hist_flow.history)


if train_models.lower() == "all" or "deblender" in train_models:

    f_net.flow.trainable = False
    # deblend_prior = f_net.td
    # deblend_prior.trainable = False
    # print(f_net.flow.trainable_variables)

    f_net = FlowVAEnet(
        latent_dim=latent_dim,
        kl_prior=kl_prior,
        kl_weight=0,
        survey=survey,
    )
    f_net.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))
    # f_net.randomize_encoder()

    f_net_original = FlowVAEnet(
        latent_dim=latent_dim,
        kl_prior=None,
        kl_weight=0,
        survey=survey,
    )
    f_net_original.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))

    # Define all used callbacks
    callbacks = define_callbacks(
        os.path.join(path_weights, "deblender"),
        lr_scheduler_epochs=lr_scheduler_epochs,
        patience=patience,
    )

    # f_net.vae_model.get_layer("latent_space").activity_regularizer=None

    ds_blended_train, ds_blended_val = batched_CATSIMDataset(
        train_data_dir=None,
        val_data_dir=None,
        tf_dataset_dir=os.path.join(
            maddeb_config["TF_DATASET_PATH"][survey_name], "blended_tfDataset"
        ),
        linear_norm_coeff=linear_norm_coeff,
        batch_size=batch_size,
        x_col_name="blended_gal_stamps",
        y_col_name="isolated_gal_stamps",
    )

    hist_deblender = f_net.train_encoder(
        ds_blended_train,
        ds_blended_val,
        callbacks=callbacks,
        epochs=deblender_epochs,
        optimizer=tf.keras.optimizers.Adam(1e-5, clipvalue=0.1),
        loss_function=deblender_encoder_loss_wrapper(
            original_encoder=f_net_original.encoder,
            noise_sigma=noise_sigma,
            latent_dim=latent_dim,
        ),
        verbose=2,
        # loss_function=vae_loss_fn_wrapper(sigma=noise_sigma, linear_norm_coeff=linear_norm_coeff),
    )

    np.save(path_weights + "/train_deblender_history.npy", hist_deblender.history)
