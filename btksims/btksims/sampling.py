"""Deifne Custom Sampling for BTK."""

import warnings

import numpy as np
from btk.sampling_functions import SamplingFunction, _get_random_center_shift
from btk.utils import DEFAULT_SEED


class CustomSampling(SamplingFunction):
    """Default sampling function used for producing blend tables."""

    def __init__(
        self,
        index_range,
        max_number=2,
        min_number=1,
        stamp_size=24.0,
        maxshift=None,
        unique=True,
        seed=DEFAULT_SEED,
    ):
        """Initialize default sampling function.

        Parameters
        ----------
        max_number: int
            maximum number of galaxies in the stamp.
        min_number: int
            minimum number of galaxies in the stamp.
        stamp_size: (=float
            Size of the desired stamp.
        index_range: tuple
            range to indexes to sample galaxies from the catalog.
        maxshift: float
            Magnitude of the maximum value of shift. If None then it
            is set as one-tenth the stamp size. (in arcseconds)
        unique: bool
            whether to have unique galaxies in different stamps.
            If true, galaxies are sampled sequentially from the catalog.
        seed: int
            Seed to initialize randomness for reproducibility.

        """
        super().__init__(max_number=max_number, min_number=min_number, seed=seed)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0
        self.index_range = index_range
        self.unique = unique
        self.indexes = list(np.arange(index_range[0], index_range[1]))

    @property
    def compatible_catalogs(self):
        """Tuple of compatible catalogs for this sampling function."""
        return "CatsimCatalog", "CosmosCatalog"

    def __call__(self, table, shifts=None, indexes=None):
        """Apply default sampling to the input CatSim-like catalog.

        Returns an astropy table with entries corresponding to a blend centered close to postage
        stamp center.

        Function selects entries from the input table that are brighter than 25.3 mag
        in the i band. Number of objects per blend is set at a random integer
        between 1 and ``self.max_number``. The blend table is then randomly sampled
        entries from the table after selection cuts. The centers are randomly
        distributed within 1/10th of the stamp size. Here even though the galaxies
        are sampled from a CatSim catalog, their spatial location are not
        representative of real blends.

        Parameters
        ----------
        table: astropy.Table
            Table containing entries corresponding to galaxies from which to sample.
        shifts: list
            Contains arbitrary shifts to be applied instead of random ones.
            Should of the form [x_peak,y_peak] where x_peak and y_peak are the lists
            containing the x and y shifts.
        indexes: list
            Contains the indexes of the galaxies to use.

        Returns
        -------
            Astropy.table with entries corresponding to one blend.

        """
        number_of_objects = self.rng.integers(self.min_number, self.max_number + 1)

        if indexes is None:
            if self.unique:
                if number_of_objects > len(self.indexes):
                    raise ValueError(
                        "Too many iterations. All galaxies have been sampled."
                    )
                current_indexes = self.indexes[:number_of_objects]
                self.indexes = self.indexes[number_of_objects:]

            else:
                current_indexes = self.rng.choice(self.indexes, size=number_of_objects)
            # print(current_indexes)
            blend_table = table[current_indexes]
        else:
            blend_table = table[indexes]
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0
        if shifts is None:
            x_peak, y_peak = _get_random_center_shift(
                number_of_objects, self.maxshift, self.rng
            )
        else:
            x_peak, y_peak = shifts
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table


# class CustomUniformSampling(SamplingFunction):
#     """Default sampling function used for producing blend tables."""

#     def __init__(
#         self,
#         max_number=2,
#         min_number=1,
#         index_range=None,
#         stamp_size=24.0,
#         maxshift=None,
#         seed=DEFAULT_SEED,
#     ):
#         """Initializes default sampling function.

#         Args:
#             max_number (int): Defined in parent class
#             stamp_size (float): Size of the desired stamp.
#             maxshift (float): Magnitude of the maximum value of shift. If None then it
#                              is set as one-tenth the stamp size. (in arcseconds)
#             seed (int): Seed to initialize randomness for reproducibility.
#         """
#         super().__init__(max_number=max_number, min_number=min_number, seed=seed)
#         self.stamp_size = stamp_size
#         self.maxshift = maxshift if maxshift else self.stamp_size / 10.0
#         self.index_range = index_range

#     @property
#     def compatible_catalogs(self):
#         """Tuple of compatible catalogs for this sampling function."""
#         return "CatsimCatalog", "CosmosCatalog"

#     def __call__(self, table, shifts=None, indexes=None):
#         """Applies default sampling to the input CatSim-like catalog.

#         Returns an astropy table with entries corresponding to a blend centered close to postage
#         stamp center.

#         Function selects entries from input table that are brighter than 25.3 mag
#         in the i band. Number of objects per blend is set at a random integer
#         between 1 and ``self.max_number``. The blend table is then randomly sampled
#         entries from the table after selection cuts. The centers are randomly
#         distributed within 1/10th of the stamp size. Here even though the galaxies
#         are sampled from a CatSim catalog, their spatial location are not
#         representative of real blends.

#         Args:
#             table (Astropy.table): Table containing entries corresponding to galaxies
#                                    from which to sample.
#             shifts (list): Contains arbitrary shifts to be applied instead of random ones.
#                            Should of the form [x_peak,y_peak] where x_peak and y_peak are the lists
#                            containing the x and y shifts.
#             indexes (list): Contains the indexes of the galaxies to use.

#         Returns:
#             Astropy.table with entries corresponding to one blend.
#         """
#         number_of_objects = self.rng.integers(1, self.max_number + 1)

#         table_region = table[self.index_range[0] : self.index_range[1]]

#         (q1,) = np.where(table_region["ref_mag"] <= 23)
#         (q2,) = np.where(
#             (table_region["ref_mag"] > 23) & (table_region["ref_mag"] <= 24)
#         )
#         (q3,) = np.where(
#             (table_region["ref_mag"] > 24) & (table_region["ref_mag"] <= 25)
#         )
#         (q4,) = np.where(table_region["ref_mag"] > 25)

#         rows = []

#         for object_num in range(number_of_objects):

#             rand_numb = self.rng.random()
#             # print("new choice ")
#             # print(rand_numb)

#             if rand_numb <= 0.25:
#                 index = self.rng.choice(q1, size=1)
#             elif rand_numb <= 0.5:
#                 index = self.rng.choice(q2, size=1)
#             elif rand_numb <= 0.75:
#                 index = self.rng.choice(q3, size=1)
#             else:
#                 index = self.rng.choice(q4, size=1)
#             rows.append(table_region[index])
#             # print(table_region[index]["ref_mag"])

#         blend_table = astropy.table.vstack(rows)
#         blend_table["ra"] = 0.0
#         blend_table["dec"] = 0.0
#         if shifts is None:
#             x_peak, y_peak = _get_random_center_shift(
#                 number_of_objects, self.maxshift, self.rng
#             )
#         else:
#             x_peak, y_peak = shifts
#         blend_table["ra"] += x_peak
#         blend_table["dec"] += y_peak

#         if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
#             blend_table["dec"] > self.stamp_size / 2.0
#         ):
#             warnings.warn("Object center lies outside the stamp")
#         return blend_table
