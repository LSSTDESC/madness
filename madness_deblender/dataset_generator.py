"""TF Dataset generator."""

import os

import galcheat
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml

from madness_deblender.utils import get_madness_deblender_config_path

with open(get_madness_deblender_config_path()) as f:
    madness_deblender_config = yaml.safe_load(f)

survey = galcheat.get_survey(madness_deblender_config["survey_name"])

_DESCRIPTION = """
#Galaxies from CATSIM WL Deblending catalogue
"""

_CITATION = ""
_URL = "https://github.com/b-biswas/MADNESS"


def Logger(str, verbosity=0):
    """Logger.

    By default logs

    Parameters
    ----------
    str: string
        string to log
    verbosity: int
        verbosity level

    """
    if int(os.environ.get("DIFF_TRACE", 0)) > verbosity:
        print(str)


class CatsimDataset(tfds.core.GeneratorBasedBuilder):
    """Catsim galaxy dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Nothing to download. The dataset was generated at the first call."
    )

    def __init__(self, train_data_dir, val_data_dir, **kwargs):
        """Initialize the dataset.

        Parameters
        ----------
        train_data_dir: string
            path to training data directory with .npy files
        val_data_dir: string
            path to validation data directory with .npy files

        """
        super().__init__(**kwargs)
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir

    def PopulateFileList(self, data_folder):
        """Populate file list.

        Parameters
        ----------
        data_folder: string
            Path to the folder with .npy files

        Returns
        -------
        list_of_images: list
            names of all images in the data_folder.

        """
        list_of_images = []
        for root, dirs, files in os.walk(data_folder, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                if os.path.splitext(file_path)[1] != ".npy":
                    continue
                list_of_images.append(file_path)
        # Make sure that some data exists
        assert len(list_of_images) > 0

        Logger(f"File list is populated .. there are {len(list_of_images)} files")

        return list_of_images

    def _info(self) -> tfds.core.DatasetInfo:
        """Return the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            homepage=_URL,
            citation=_CITATION,
            features=tfds.features.FeaturesDict(
                {
                    "isolated_gal_stamps": tfds.features.Tensor(
                        shape=(45, 45, len(survey.available_filters)), dtype=tf.float32
                    ),
                    "blended_gal_stamps": tfds.features.Tensor(
                        shape=(45, 45, len(survey.available_filters)),
                        dtype=tf.dtypes.float32,
                    ),
                }
            ),
        )

    def _split_generators(self, dl):
        """Return generators according to split.

        Parameters
        ----------
        dl: tfds.download.DownloadManager
            not used here because nothing is to be downloaded

        """
        return {
            tfds.Split.TRAIN: self._generate_examples(self.train_data_dir),
            tfds.Split.VALIDATION: self._generate_examples(self.val_data_dir),
        }

    def _generate_examples(self, data_folder):
        """Yield examples.

        Parameters
        ----------
        data_folder: string
            path to the data folder

        """
        list_of_images = self.PopulateFileList(data_folder)

        for gal_file in list_of_images:
            # very verbose
            Logger(f"File : {gal_file} is being treated", 2)
            current_sample = np.load(gal_file, allow_pickle=True)
            key = os.path.splitext(gal_file)[0]
            example = {}
            example["isolated_gal_stamps"] = current_sample["isolated_gal_stamps"][
                0
            ].astype("float32")
            example["blended_gal_stamps"] = current_sample["blended_gal_stamps"][
                0
            ].astype("float32")
            yield key, example


def loadCATSIMDataset(
    train_data_dir,
    val_data_dir,
    output_dir,
):
    """Load/Generate CATSIM Dataset.

    If the TFDataset has already been generated (first call) it is reloaded.
    If the CATSIM tf dataset is already generated train_data_dir and val_data_dir are ignored.

    Parameters
    ----------
    train_data_dir: string
        Path to .npy files for training.
        Ignored if TF Dataset is already present in output_dir.
    val_data_dir: string
        Path to .npy files for validation.
        Ignored if TF Dataset is already present in output_dir.
    output_dir: string
        Path to save the tf Dataset

    Returns
    -------
    ds: dictionary tf datasets.
        refer to CatsimDataset._split_generators.

    """
    arg_dict = {
        "train_data_dir": train_data_dir,
        "val_data_dir": val_data_dir,
    }
    # subsets = [tfds.Split.TRAIN,tfds.Split.VALIDATION]
    ds = tfds.load("CatsimDataset", data_dir=output_dir, builder_kwargs=arg_dict)

    return ds


def batched_CATSIMDataset(
    tf_dataset_dir,
    linear_norm_coeff,
    batch_size,
    x_col_name="blended_gal_stamps",
    y_col_name="isolated_gal_stamps",
    train_data_dir=None,
    val_data_dir=None,
):
    """Load generated tf dataset.

    Parameters
    ----------
    tf_dataset_dir: string
        Path to generated tf dataset.
    linear_norm_coeff: int
        linear norm coefficient.
    batch_size: int
        size of batches to be generated.
    x_col_name: string
        column name for input to the ML models.
        Defaults to "blended_gal_stamps".
    y_col_name: string
        column name for loss computation of the ML models.
        Defaults to "isolated_gal_stamps".
    train_data_dir: string
        Path to .npy files for training.
        Ignored if TF Dataset is already present in output_dir.
    val_data_dir: string
        Path to .npy files for validation.
        Ignored if TF Dataset is already present in output_dir.

    Returns
    -------
    ds_train: tf dataset
        prefetched training dataset
    ds_val: tf dataset
        prefetched validation dataset

    """

    def preprocess_batch(ds):
        """Preprocessing function.

        Randomly flips, normalizes, shuffles the dataset

        Parameters
        ----------
        ds: tf dataset
            prefetched tf dataset

        Returns
        -------
        ds: tf dataset
            processing dataset, with (x,y) for training/validating network

        """

        def pre_process(elem):
            """Pre-processing function preparing data for denoising task.

            Parameters
            ----------
            elem: dict
                element of tf dataset.

            Returns
            -------
            (x, y): tuple
                data for training Neural Networks

            """
            x = elem[x_col_name] / linear_norm_coeff
            y = elem[y_col_name] / linear_norm_coeff

            do_flip_lr = tf.random.uniform([]) > 0.5
            if do_flip_lr:
                x = tf.image.flip_left_right(x)
                y = tf.image.flip_left_right(y)

            do_flip_ud = tf.random.uniform([]) > 0.5
            if do_flip_ud:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)

            return (x, y)

        ds = ds.shuffle(buffer_size=15 * batch_size)
        ds = ds.map(pre_process)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    ds = loadCATSIMDataset(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        output_dir=tf_dataset_dir,
    )

    ds_train = preprocess_batch(ds=ds[tfds.Split.TRAIN])

    ds_val = preprocess_batch(ds=ds[tfds.Split.VALIDATION])

    return ds_train, ds_val
