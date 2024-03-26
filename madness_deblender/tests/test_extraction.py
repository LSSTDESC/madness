"""Test extract cutouts."""

import numpy as np

from madness_deblender.extraction import extract_cutouts


def test_cutouts_border():
    """Test extract cutouts function."""
    field_size = 15
    nb_of_bands = 3
    cutout_size = 5

    # create image
    image = np.random.rand(field_size, field_size, nb_of_bands)

    # test when the cutout is not close to the edge
    galaxy_distances_to_center = [[-4, -3]]
    cutout_size = 5
    cutout, list_idx = extract_cutouts(
        field_image=image.copy(),
        pos=galaxy_distances_to_center,
        distances_to_center=True,
        cutout_size=cutout_size,
        channel_last=True,
    )

    np.testing.assert_array_equal(cutout[0], image[1:6, 2:7])

    # test when cutout is just contained within the boundary
    galaxy_distances_to_center = [[12, 12]]
    cutout_size = 5
    cutout, list_idx = extract_cutouts(
        field_image=image.copy(),
        pos=galaxy_distances_to_center,
        distances_to_center=False,
        cutout_size=cutout_size,
        channel_last=True,
    )
    np.testing.assert_array_equal(cutout[0], image[10:, 10:])
    assert list_idx[0] == 0

    galaxy_distances_to_center = [[-5, -5]]

    cutout_size = 5
    cutout, list_idx = extract_cutouts(
        field_image=np.transpose(image.copy(), axes=[2, 0, 1]),
        pos=galaxy_distances_to_center,
        distances_to_center=True,
        cutout_size=cutout_size,
        channel_last=False,
    )

    np.testing.assert_array_equal(cutout[0], image[:5, :5])
    assert len(list_idx) == 1

    # test when the cutout is too large
    galaxy_distances_to_center = [[6, 6]]
    cutout_size = 5
    cutout, list_idx = extract_cutouts(
        field_image=image.copy(),
        pos=galaxy_distances_to_center,
        distances_to_center=False,
        cutout_size=cutout_size,
    )

    assert len(list_idx) == 0
