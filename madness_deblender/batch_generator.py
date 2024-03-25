"""Load batches of data to pass to TensorFlow during training."""

import random

import numpy as np
from tensorflow.keras.utils import Sequence


class COSMOSsequence(Sequence):
    """Load data for training."""

    def __init__(
        self,
        list_of_samples,
        x_col_name,
        y_col_name,
        batch_size,
        num_iterations_per_epoch,
        linear_norm_coeff=1,
        dataset="train",
        # channel_last=False,
    ):
        """Initialize the Data generator.

        Parameters
        ----------
        list_of_samples: list
            list of paths to the datafiles.
        x_col_name: String
            column name of data to be fed as input to the network.
        y_col_name: String
            column name of data to be fed as target to the network.
        batch_size: int
            sample size for each batch.
        num_iterations_per_epoch: int
            number of samples (of size = batch_size) to be drawn from the sample.
        linear_norm_coeff: int/list
            list stores the bandwise linear normalizing/scaling factor.
            if int is passed, same scaling factor is used for all.
        dataset: str
            "train" or "validation"

        """
        self.list_of_samples = list_of_samples
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.batch_size = batch_size
        self.num_iterations_per_epoch = num_iterations_per_epoch

        self.linear_norm_coeff = linear_norm_coeff
        self.dataset = dataset
        # self.channel_last = channel_last

    def __len__(self):
        """Return the number of iterations per epoch."""
        return self.num_iterations_per_epoch

    def __getitem__(self, idx):
        """Fetch the next batch for specific file.

        Parameters
        ----------
        idx: int
            File number to opened
        Returns
        -------
        x: np array
            Data corresponding to x_col_name
        y: np array
            Data corresponding to x_col_name

        """
        current_loop_file_name = self.list_of_samples[idx]
        if self.dataset == "train":
            current_loop_file_name = random.choice(self.list_of_samples)

        current_sample = np.load(current_loop_file_name, allow_pickle=True)

        batch = np.random.choice(current_sample, size=self.batch_size, replace=False)
        x = batch[self.x_col_name]
        y = batch[self.y_col_name]

        x = np.array(x.tolist())
        y = np.array(y.tolist())

        if self.linear_norm_coeff is not None:
            x = x / self.linear_norm_coeff
            y = y / self.linear_norm_coeff

        rand = np.random.randint(4)
        if rand == 1:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)

        if rand == 2:
            x = np.flip(x, axis=2)
            y = np.flip(y, axis=2)

        if rand == 3:
            x = np.flip(np.flip(x, axis=1), axis=2)
            y = np.flip(np.flip(y, axis=1), axis=2)

        # flip : flipping the image array
        # if not self.channel_last:
        # rand = np.random.randint(4)
        # if rand == 1:
        #     x = np.flip(x, axis=-1)
        #     y = np.flip(y, axis=-1)
        # elif rand == 2:
        #     x = np.swapaxes(x, -1, -2)
        #     y = np.swapaxes(y, -1, -2)
        # elif rand == 3:
        #     x = np.swapaxes(np.flip(x, axis=-1), -1, -2)
        #     y = np.swapaxes(np.flip(y, axis=-1), -1, -2)

        # Change the shape of inputs and targets to feed the network
        # x = np.transpose(x, axes=(0, 2, 3, 1))
        # y = np.transpose(y, axes=(0, 2, 3, 1))

        return x, y
