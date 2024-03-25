"""Simulations for training models."""

import logging
import os
import sys

import btk.catalog
import btk.draw_blends
import btk.sampling_functions
import btk.survey
import btk.utils
import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from maddeb.extraction import extract_cutouts
from maddeb.utils import get_maddeb_config_path

from btksims.sampling import CustomSampling

logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

LOG.info(sys.argv)
dataset = sys.argv[1]  # should be either training or validation
if dataset not in ["training", "validation"]:
    raise ValueError(
        "The first argument (dataset) should be either training or validation"
    )


blend_type = sys.argv[2]  # set to 4 to generate blended scenes
if blend_type not in ["isolated", "blended"]:
    raise ValueError("The second argument should be either isolated or blended")

with open(get_maddeb_config_path()) as f:
    maddeb_config = yaml.safe_load(f)

survey_name = maddeb_config["survey_name"]
btksims_config = maddeb_config["btksims"]

sim_config = btksims_config["TRAIN_VAL_PARAMS"]

survey = btk.survey.get_surveys(survey_name)
SAVE_PATH = btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name]
CATALOG_PATH = btksims_config["CAT_PATH"][survey_name]
print("saving data at " + SAVE_PATH)

if type(CATALOG_PATH) == list:
    catalog = btk.catalog.CosmosCatalog.from_file(CATALOG_PATH, exclusion_level="none")
    generator = btk.draw_blends.CosmosGenerator
else:
    catalog = btk.catalog.CatsimCatalog.from_file(CATALOG_PATH)
    generator = btk.draw_blends.CatsimGenerator

catalog.table = Table.from_pandas(
    catalog.table.to_pandas().sample(frac=1, random_state=0).reset_index(drop=True)
)
survey = btk.survey.get_surveys(survey_name)

sampling_function = CustomSampling(
    index_range=sim_config[dataset][survey_name]["index_range"],
    min_number=sim_config[blend_type + "_params"]["min_number"],
    max_number=sim_config[blend_type + "_params"]["max_number"],
    maxshift=sim_config["maxshift"],
    stamp_size=sim_config["stamp_size"],
    seed=sim_config["btk_seed"],
    unique=sim_config[blend_type + "_params"]["unique_galaxies"],
)

draw_generator = generator(
    catalog,
    sampling_function,
    survey,
    batch_size=sim_config["btk_batch_size"],
    stamp_size=sim_config["stamp_size"],
    njobs=8,
    add_noise="all",
    verbose=False,
    seed=sim_config["btk_seed"],
)

total_galaxy_stamps = (
    sim_config[dataset][survey_name]["num_batches"] * sim_config["btk_batch_size"]
)
stamp_counter = 0
# shift_rng = np.random.default_rng(12345)
for batch_num in range(sim_config[dataset][survey_name]["num_batches"]):

    print("simulating file number:" + str(batch_num))

    batch = next(draw_generator)

    for blended_image_num in range(len(batch.blend_images)):

        blended_image = batch.blend_images[blended_image_num]

        for galaxy_num in range(len(batch.catalog_list[blended_image_num]["x_peak"])):

            postage_stamps = {}
            # print("Image number "+ str(blended_image_num))
            # print("Galaxy number " + str(galaxy_num))
            isolated_image = batch.isolated_images[blended_image_num][galaxy_num]
            x_pos = batch.catalog_list[blended_image_num]["y_peak"][galaxy_num]
            y_pos = batch.catalog_list[blended_image_num]["x_peak"][galaxy_num]

            # x_shift = shift_rng.random() - 0.5
            # y_shift = shift_rng.random() - 0.5
            x_shift = 0
            y_shift = 0

            pos = (x_pos + x_shift, y_pos + y_shift)
            gal_blended = extract_cutouts(
                blended_image,
                [pos],
                distances_to_center=False,
                channel_last=False,
                cutout_size=45,
            )[0][0]
            postage_stamps["blended_gal_stamps"] = [gal_blended]
            gal_isolated = extract_cutouts(
                isolated_image,
                [pos],
                distances_to_center=False,
                channel_last=False,
                cutout_size=45,
            )[0][0]
            postage_stamps["isolated_gal_stamps"] = [gal_isolated]
            postage_stamps["gal_locations_y_peak"] = [
                batch.catalog_list[blended_image_num]["y_peak"] - pos[0]
            ]
            postage_stamps["gal_locations_x_peak"] = [
                batch.catalog_list[blended_image_num]["x_peak"] - pos[1]
            ]
            if "r_band_snr" not in batch.catalog_list[blended_image_num].columns:
                postage_stamps["r_band_snr"] = 0
            else:
                postage_stamps["r_band_snr"] = [
                    batch.catalog_list[blended_image_num]["r_band_snr"]
                ]

            postage_stamps = pd.DataFrame(postage_stamps)

            np.save(
                os.path.join(
                    SAVE_PATH,
                    blend_type + "_" + dataset,
                    f"gal_{batch_num}_{blended_image_num}_{galaxy_num}.npy",
                ),
                postage_stamps.to_records(),
            )

            stamp_counter += 1
            if stamp_counter == total_galaxy_stamps:
                LOG.info(f"simulated {stamp_counter} stamps")
                sys.exit()
