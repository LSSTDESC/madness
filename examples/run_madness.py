"""Run MADNESS on test dataset."""

import logging
import math
import os
import pickle
import sys

import galcheat
import numpy as np
import pandas as pd
import sep
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from maddeb.Deblender import Deblend
from maddeb.metrics import compute_aperture_photometry, compute_pixel_cosdist
from maddeb.utils import get_data_dir_path, get_maddeb_config_path

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)

with open(get_maddeb_config_path()) as f:
    maddeb_config = yaml.safe_load(f)

survey_name = maddeb_config["survey_name"]
if survey_name not in ["LSST", "HSC"]:
    raise ValueError(
        "survey should be one of: LSST, HSC"
    )  # other surveys to be added soon!
survey = galcheat.get_survey(survey_name)

LOG.info(f"Running tests with MADNESS for {survey_name}")


density = sys.argv[1]
kl_weight_exp = int(sys.argv[2])
run_name = f"kl{kl_weight_exp}b"
map_solution = sys.argv[3].lower() == "true"

if density not in ["high", "low"]:
    raise ValueError("The second argument should be either isolated or blended")

num_repetations = 300
max_number = 20

kl_weight = 10**-kl_weight_exp
weights_path = os.path.join(get_data_dir_path(), survey_name + str(kl_weight))
LOG.info(f"Loading Model weights from {weights_path}")

simulation_path = os.path.join(maddeb_config["TEST_DATA_PATH"][survey_name], density)
results_path = maddeb_config["RESULTS_PATH"][survey_name]
density_level = density + "_density"

LOG.info(f"Saving results to: {results_path}")
LOG.info(f"Density level: {density_level}")
LOG.info(f"Run name: {run_name}")

deb = Deblend(latent_dim=16, weights_path=weights_path, survey=survey)

psf_fwhm = []
for band in survey.available_filters:
    filt = survey.get_filter(band)
    psf_fwhm.append(filt.psf_fwhm.value)

for file_num in range(num_repetations):
    LOG.info(f"\n\n######### Processing file: {file_num} #########")
    file_name = os.path.join(
        simulation_path,
        str(file_num) + ".pkl",
    )
    with open(file_name, "rb") as f:
        blend = pickle.load(f)

    field_images = blend.blend_images
    isolated_images = blend.isolated_images

    # get MADNESS predictions
    madness_predictions = []
    madness_models = []
    linear_norm_coeff = 10000

    catalog_list = []
    madness_results = []

    actual_photometry = []
    madness_photometry = []
    blended_photometry = []

    detected_positions = np.zeros((len(blend.catalog_list), max_number, 2))
    num_components = []
    for field_num in range(len(blend.catalog_list)):
        for gal_num in range(len(blend.catalog_list[field_num])):
            detected_positions[field_num][gal_num][0] = blend.catalog_list[field_num][
                "y_peak"
            ][gal_num]
            detected_positions[field_num][gal_num][1] = blend.catalog_list[field_num][
                "x_peak"
            ][gal_num]
        num_components.append(len(blend.catalog_list[field_num]))

    convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
        rtol=0.05,
        min_num_steps=40,
        window_size=15,
    )
    # convergence_criterion = None
    initial_lr = 0.025
    if survey_name == "LSST":
        initial_lr = 0.05
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=30,
        decay_rate=0.8,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    deb(
        field_images,
        detected_positions,
        num_components=num_components,  # redundant parameter
        use_log_prob=True,
        linear_norm_coeff=linear_norm_coeff,
        max_iter=200,
        convergence_criterion=convergence_criterion,
        optimizer=optimizer,
        use_debvader=True,
        map_solution=map_solution,
    )
    padding_infos_all_fields = deb.get_padding_infos()

    for field_num in range(len(blend.catalog_list)):
        # LOG.info(field_num)

        current_field_predictions = []
        current_madness_models = {"images": [], "field_num": [], "galaxy_num": []}

        current_blend = blend.catalog_list[field_num]

        padding_infos = padding_infos_all_fields[field_num]
        for component_num in range(deb.num_components[field_num]):
            # LOG.info(component_num)
            # LOG.info(padding_infos)
            prediction = np.pad(
                deb.components[field_num][component_num],
                padding_infos[component_num],
            )
            prediction = np.transpose(prediction, axes=(2, 0, 1))
            current_field_predictions.append(prediction)

        # madness_predictions.append(current_field_predictions)

        num_galaxies = len(blend.catalog_list[field_num])

        isolated_images = blend.isolated_images[field_num][0:num_galaxies]

        madness_current_res = compute_pixel_cosdist(
            current_field_predictions,
            isolated_images,
            blend.blend_images[field_num],
            survey=survey,
        )

        bkg_rms = {}
        for band in range(len(survey.available_filters)):
            bkg_rms[band] = sep.Background(
                blend.blend_images[field_num][band]
            ).globalrms

        # madness_current_res["images"] = current_field_predictions
        madness_current_res["field_num"] = [field_num] * num_galaxies
        madness_current_res["file_num"] = [file_num] * num_galaxies

        if survey_name == "LSST":
            madness_current_res["r_band_snr"] = blend.catalog_list[field_num][
                "r_band_snr"
            ]
            for band_name in survey.available_filters:
                madness_current_res[band_name + "_ab"] = blend.catalog_list[field_num][
                    band_name + "_ab"
                ]

            a = blend.catalog_list[field_num]["a_d"].value
            b = blend.catalog_list[field_num]["b_d"].value
            theta = blend.catalog_list[field_num]["pa_disk"].value

            cond = (
                blend.catalog_list[field_num]["a_d"]
                < blend.catalog_list[field_num]["a_b"]
            )
            a = np.where(cond, blend.catalog_list[field_num]["a_b"].value, a)
            b = np.where(cond, blend.catalog_list[field_num]["b_b"].value, b)
            theta = np.where(
                cond, blend.catalog_list[field_num]["pa_bulge"].value, theta
            )

            theta = theta % 180
            theta = theta * math.pi / 180

            theta = np.where(theta > math.pi / 2, theta - math.pi, theta)
            madness_current_res["size"] = (a * b) ** 0.5

        if survey_name == "HSC":
            madness_current_res["r_band_snr"] = 0
            for band_name in survey.available_filters:
                madness_current_res[band_name + "_ab"] = blend.catalog_list[field_num][
                    "HSC_" + band_name
                ]
            a = (
                blend.catalog_list[field_num]["flux_radius"]
                * blend.catalog_list[field_num]["PIXEL_SCALE"]
            )
            b = a
            theta = [0] * len(blend.catalog_list[field_num])
            madness_current_res["size"] = a
        # make this a table

        # madness_results.append(madness_current_res)

        madness_photometry_current = compute_aperture_photometry(
            field_image=blend.blend_images[field_num],
            predictions=current_field_predictions,
            xpos=blend.catalog_list[field_num]["x_peak"],
            ypos=blend.catalog_list[field_num]["y_peak"],
            a=a / survey.pixel_scale.value,
            b=b / survey.pixel_scale.value,
            theta=theta,
            psf_fwhm=np.array(psf_fwhm) / survey.pixel_scale.value,
            bkg_rms=bkg_rms,
            survey=survey,
        )
        madness_current_res.update(madness_photometry_current)
        madness_current_res = pd.DataFrame.from_dict(madness_current_res)
        madness_results.append(madness_current_res)

        actual_results_current = compute_aperture_photometry(
            field_image=blend.blend_images[field_num],
            predictions=blend.isolated_images[field_num],
            xpos=blend.catalog_list[field_num]["x_peak"],
            ypos=blend.catalog_list[field_num]["y_peak"],
            a=a / survey.pixel_scale.value,
            b=b / survey.pixel_scale.value,
            theta=theta,
            psf_fwhm=np.array(psf_fwhm) / survey.pixel_scale.value,
            bkg_rms=bkg_rms,
            survey=survey,
        )
        actual_results_current["field_num"] = field_num * num_galaxies
        actual_results_current["file_num"] = file_num * num_galaxies
        actual_results_current["galaxy_num"] = np.arange(num_galaxies)
        actual_results_current = pd.DataFrame.from_dict(actual_results_current)
        actual_photometry.append(actual_results_current)

        blended_results_current = compute_aperture_photometry(
            field_image=blend.blend_images[field_num],
            predictions=None,
            xpos=blend.catalog_list[field_num]["x_peak"],
            ypos=blend.catalog_list[field_num]["y_peak"],
            a=a / survey.pixel_scale.value,
            b=b / survey.pixel_scale.value,
            theta=theta,
            psf_fwhm=np.array(psf_fwhm) / survey.pixel_scale.value,
            bkg_rms=bkg_rms,
            survey=survey,
        )
        blended_results_current["field_num"] = field_num * num_galaxies
        blended_results_current["file_num"] = file_num * num_galaxies
        blended_results_current["galaxy_num"] = np.arange(num_galaxies)
        blended_results_current = pd.DataFrame.from_dict(blended_results_current)
        blended_photometry.append(blended_results_current)

    madness_results = pd.concat(madness_results, ignore_index=True)
    # madness_results = hstack([madness_results, vstack(blend.catalog_list)])
    # madness_results = hstack([madness_results,vstack(madness_photometry)])

    # madness_results = hstack([madness_results, vstack(blend.catalog_list)])

    actual_photometry = pd.concat(actual_photometry, ignore_index=True)
    blended_photometry = pd.concat(blended_photometry, ignore_index=True)

    madness_results["r_contamination"] = (
        blended_photometry["r_phot_flux"].values
        - actual_photometry["r_phot_flux"].values
    ) / actual_photometry["r_phot_flux"].values

    # reconstruction_file = open("debblend_results" + str(rep_num)+ ".pkl", "wb")

    save_file_name = os.path.join(
        results_path,
        density_level,
        run_name,
        "madness_results" if map_solution else "debvader_results",
        str(file_num) + ".pkl",
    )
    # np.save(save_file_name,
    #     madness_results.to_records())
    madness_results.to_pickle(save_file_name)
    # hickle.dump(madness_results, save_file_name, mode="w")
    # ascii.write(madness_results, save_file_name, overwrite=True)

    save_file_name = os.path.join(
        results_path,
        density_level,
        "actual_photometry",
        str(file_num) + ".pkl",
    )

    actual_photometry.to_pickle(save_file_name)

    save_file_name = os.path.join(
        results_path,
        density_level,
        "blended_photometry",
        str(file_num) + ".pkl",
    )

    blended_photometry.to_pickle(save_file_name)
