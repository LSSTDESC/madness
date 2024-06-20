"""Define utils."""

import os


def listdir_fullpath(d):
    """List of files in a directory.

    Parameters
    ----------
    d: String
        directory path

    Returns
    -------
    file_names: list
        file of file names in the directory d

    """
    file_names = [os.path.join(d, f) for f in os.listdir(d)]
    return file_names


def get_data_dir_path():
    """Fetch path to the data folder of madness_deblender.

    Returns
    -------
    data_dir: str
        path to data folder

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curdir, "data")

    return data_dir
